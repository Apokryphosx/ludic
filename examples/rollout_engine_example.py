# examples/rollout_engine_example.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

from ludic.agent import Agent
from ludic.context.base import ContextStrategy
from ludic.types import Rollout

from ludic.training.types import (
    CtxSpec,
    EnvSpec,
    RolloutRequest,
    SamplingArgs,
)

# Adjust this import to wherever you actually placed RolloutEngine
from ludic.training.rollout_engine import RolloutEngine, EnvRegistry, CtxRegistry

from examples.envs.tictactoe import TicTacToeEnv


# ---------------------------------------------------------------------------
# Minimal context strategy (if you don't already have one handy)
# ---------------------------------------------------------------------------

class NoopContext(ContextStrategy):
    """
    Ultra-minimal ContextStrategy used just to make Tic-Tac-Toe work.

    If your codebase already has a proper ContextStrategy implementation
    (e.g. chat-style context), feel free to delete this and plug that in
    instead. This class is intentionally dumb.
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        # No state to reset
        pass

    def update(self, *, obs: str, action: str | None = None) -> None:
        # No state, so nothing to do
        return

    def build_prompt(self, *, system_prompt: str | None, obs: str) -> str:
        """
        Build a naive single-turn prompt for the agent.

        A real implementation would track conversation history etc.
        """
        parts: List[str] = []
        if system_prompt:
            parts.append(system_prompt.strip())
            parts.append("")  # blank line
        parts.append(obs)
        parts.append("")
        parts.append("Your move:")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent() -> Agent:
    """
    Construct whatever Agent you normally use in Ludic.

    This is intentionally left as a single hook so you only edit in one place.
    For example, you might do something like:

        from ludic.agent.openai import OpenAIChatAgent

        return OpenAIChatAgent(
            model="gpt-4.1-mini",
            temperature=0.0,
        )

    For now this just raises to force you to wire *something* real in.
    """
    raise RuntimeError(
        "Implement build_agent() to return a concrete ludic.Agent "
        "(OpenAI / local model / etc.)"
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def make_env_registry() -> EnvRegistry:
    """
    Two logically different Tic-Tac-Toe envs:

    - ttt_agent_first:    agent plays first (X goes first)
    - ttt_opponent_first: opponent makes a random opening move

    This gives us heterogeneous envs while reusing the same TicTacToeEnv class.
    """
    return {
        "ttt_agent_first": lambda **kwargs: TicTacToeEnv(
            agent_starts=True,
            show_opponent_move=True,
            **kwargs,
        ),
        "ttt_opponent_first": lambda **kwargs: TicTacToeEnv(
            agent_starts=False,
            show_opponent_move=True,
            **kwargs,
        ),
    }


def make_ctx_registry() -> CtxRegistry:
    """
    Two context kinds just so we can prove heterogeneity at the ctx layer too.

    In practice you'll probably just use a single chat-style context everywhere.
    """
    return {
        "noop": lambda **kwargs: NoopContext(),
        # You can add more exotic context strategies here later.
    }


# ---------------------------------------------------------------------------
# Rollout request factory
# ---------------------------------------------------------------------------

def make_rollout_requests() -> List[RolloutRequest]:
    """
    Build a small heterogeneous batch:

    - 3 episodes where the agent starts
    - 3 episodes where the opponent plays first

    Both share the same context kind ("noop") to keep the example simple.
    """
    # Shared sampling arguments; adjust if your Agent uses them.
    sargs: SamplingArgs = {"temperature": 0.0}

    req_agent_first = RolloutRequest(
        env=EnvSpec(
            kind="ttt_agent_first",
            kwargs={},   # could pass custom kwargs into TicTacToeEnv here
        ),
        ctx=CtxSpec(
            kind="noop",
            kwargs={},
        ),
        num_episodes=3,
        sampling_args=sargs,
        system_prompt=None,   # fall back to env.suggested_sysprompt inside run_episode if you do that
        meta={"label": "agent_first"},
    )

    req_opponent_first = RolloutRequest(
        env=EnvSpec(
            kind="ttt_opponent_first",
            kwargs={},
        ),
        ctx=CtxSpec(
            kind="noop",
            kwargs={},
        ),
        num_episodes=3,
        sampling_args=sargs,
        system_prompt=None,
        meta={"label": "opponent_first"},
    )

    return [req_agent_first, req_opponent_first]


# ---------------------------------------------------------------------------
# Assertions that heterogeneity actually shows up
# ---------------------------------------------------------------------------

def assert_heterogeneous_envs(rollouts: List[Rollout]) -> None:
    """
    Sanity-checks:

    - We got the expected number of episodes.
    - Both env kinds appear in the metadata.
    - The observations behave differently depending on whether the opponent
      moves first (opening obs should mention an opponent move).
    """
    assert rollouts, "No rollouts returned, something is broken."

    labels = {r.meta["request_meta"]["label"] for r in rollouts}
    assert labels == {"agent_first", "opponent_first"}, (
        f"Expected both agent_first and opponent_first, got {labels}"
    )

    env_kinds = {r.meta["engine"]["env_kind"] for r in rollouts}
    assert env_kinds == {"ttt_agent_first", "ttt_opponent_first"}, (
        f"Expected two env kinds, got {env_kinds}"
    )

    # Check that opponent-first games actually start with an opponent move
    saw_agent_first_clean = False
    saw_opponent_first_opening = False

    for r in rollouts:
        label = r.meta["request_meta"]["label"]
        # First step prev_obs is what the agent saw before its first move
        first_step = r.steps[0]
        prev_obs: str = first_step.prev_obs

        opp_tag = "Opponent (O) played at"
        has_opening_opp = opp_tag in prev_obs

        if label == "agent_first":
            # In this configuration, the opponent should NOT have played yet.
            if not has_opening_opp:
                saw_agent_first_clean = True
        elif label == "opponent_first":
            # Here the opponent should have already made a random move.
            if has_opening_opp:
                saw_opponent_first_opening = True

    assert saw_agent_first_clean, (
        "Did not find any agent_first rollout whose opening obs "
        "lacked an opponent move."
    )
    assert saw_opponent_first_opening, (
        "Did not find any opponent_first rollout whose opening obs "
        "mentioned an opponent move."
    )


def assert_episode_indices(rollouts: List[Rollout]) -> None:
    """
    Check that episode_idx is contiguous and unique across the whole batch.
    This verifies that RolloutEngine's global indexing works as intended.
    """
    indices = sorted(r.meta["episode_idx"] for r in rollouts)
    expected = list(range(len(rollouts)))
    assert indices == expected, (
        f"Episode indices {indices} do not match expected {expected}"
    )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def main() -> None:
    # 1) Build agent + registries
    agent = build_agent()  # <-- you must implement this function
    env_registry: EnvRegistry = make_env_registry()
    ctx_registry: CtxRegistry = make_ctx_registry()

    jsonl_path = Path("outputs/tictactoe_rollouts.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    engine = RolloutEngine(
        agent=agent,
        env_registry=env_registry,
        ctx_registry=ctx_registry,
        jsonl_path=str(jsonl_path),
    )

    # 2) Build heterogeneous rollout requests
    requests = make_rollout_requests()

    # 3) Run rollouts
    rollouts = await engine.generate_rollouts(
        requests=requests,
        max_steps=9,        # Tic-Tac-Toe can't be longer than 9 moves
        timeout_s=30.0,
        concurrency=4,
    )

    print(f"Generated {len(rollouts)} rollouts.")
    for r in rollouts:
        label = r.meta["request_meta"]["label"]
        env_kind = r.meta["engine"]["env_kind"]
        print(
            f"- rollout_id={r.id} | label={label} | env_kind={env_kind} "
            f"| total_reward={r.total_reward:.1f} | length={r.length}"
        )

    # 4) Assertions: check that heterogeneity & engine metadata behave as expected
    assert_heterogeneous_envs(rollouts)
    assert_episode_indices(rollouts)

    print()
    print(f"Rollouts written to: {jsonl_path.resolve()}")
    print("All assertions passed ✅")


if __name__ == "__main__":
    asyncio.run(main())
