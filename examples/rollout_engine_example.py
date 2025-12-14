from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import xml_move_parser
from ludic.training import RolloutEngine, EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout, SamplingArgs

from environments.tic_tac_toe import TicTacToeEnv


# ---------------------------------------------------------------------------
# vLLM / model config
# ---------------------------------------------------------------------------

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen3-0.6B"

JSONL_PATH = Path("outputs/tictactoe_rollouts_heterogeneous.xml.jsonl")


# ---------------------------------------------------------------------------
# System prompt glue (same pattern as your other script)
# ---------------------------------------------------------------------------

def build_system_prompt(env_cls: type[TicTacToeEnv] = TicTacToeEnv) -> str:
    """
    Take the env's suggested_sysprompt and append the XML contract line.
    Env stays parser-agnostic; XML is enforced by the parser.
    """
    env_for_prompt = env_cls()
    base = env_for_prompt.suggested_sysprompt or ""
    extra = """
When you choose a move, respond ONLY with a single XML tag containing the move,
for example:

    <move>A1</move>

Do not include any other text, commentary, or tags.
"""
    return (base.rstrip() + "\n\n" + extra.strip()).strip()

def make_env_registry():
    """
    Two logically different Tic-Tac-Toe envs:

    - ttt_agent_first:    agent plays first (X goes first)
    - ttt_opponent_first: opponent makes a random opening move

    Heterogeneity is just the `agent_starts` flag.
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


# ---------------------------------------------------------------------------
# Rollout request factory
# ---------------------------------------------------------------------------

def make_rollout_requests() -> List[RolloutRequest]:
    """
    Build a small heterogeneous batch:

    - 3 episodes where the agent starts
    - 3 episodes where the opponent plays first

    Both use FullDialog and the same XML system prompt.
    """
    sampling_args: SamplingArgs = {
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    system_prompt = build_system_prompt()

    req_agent_first = RolloutRequest(
        env=EnvSpec(
            kind="ttt_agent_first",
            kwargs={},
        ),
        protocol=ProtocolSpec(kind="single_agent", kwargs={"prompt": system_prompt}),
        num_episodes=3,
        sampling_args=sampling_args,
        meta={"label": "agent_first"},
    )

    req_opponent_first = RolloutRequest(
        env=EnvSpec(
            kind="ttt_opponent_first",
            kwargs={},
        ),
        protocol=ProtocolSpec(kind="single_agent", kwargs={"prompt": system_prompt}),
        num_episodes=3,
        sampling_args=sampling_args,
        meta={"label": "opponent_first"},
    )

    return [req_agent_first, req_opponent_first]


# ---------------------------------------------------------------------------
# Assertions that heterogeneity actually shows up
# ---------------------------------------------------------------------------

def assert_heterogeneous_envs(rollouts: List[Rollout]) -> None:
    """
    Sanity-checks:

    - Both env labels appear.
    - Both env kinds appear.
    - Opponent-first games show an opening opponent move in the first obs.
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

    # This assumes your env sticks this marker into the text when O moves.
    opp_tag = "Opponent (O) played at"
    saw_agent_first_clean = False
    saw_opponent_first_opening = False

    for r in rollouts:
        if not r.steps:
            continue

        label = r.meta["request_meta"]["label"]
        first_step = r.steps[0]
        prev_obs: str = first_step.prev_obs

        has_opening_opp = opp_tag in prev_obs

        if label == "agent_first":
            if not has_opening_opp:
                saw_agent_first_clean = True
        elif label == "opponent_first":
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
    print(
        f"Connecting to vLLM at http://{VLLM_HOST}:{VLLM_PORT} "
        f"using model {MODEL_NAME}..."
    )

    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        connection_timeout_s=300.0,
        enable_weight_updates=False,
    )

    env_registry = make_env_registry()

    def create_protocol(prompt: str | None = None):
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=MODEL_NAME,
                ctx=FullDialog(),
                parser=xml_move_parser,
            ),
            prompt=prompt,
        )

    protocol_registry = {"single_agent": create_protocol}

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(JSONL_PATH),
    )

    # 1) Build heterogeneous rollout requests (with XML parser + system prompt)
    requests = make_rollout_requests()

    # 2) Run rollouts through the engine
    rollouts = await engine.generate_rollouts(
        requests=requests,
        max_steps=9,        # Tic-Tac-Toe can't be longer than 9 moves
        timeout_s=60.0,
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

    # 3) Assertions: check that heterogeneity & engine metadata behave as expected
    assert_heterogeneous_envs(rollouts)
    assert_episode_indices(rollouts)

    print()
    print(f"Rollouts written to: {JSONL_PATH.resolve()}")
    print("All assertions passed ✅")


if __name__ == "__main__":
    # Expect vLLM server already running, e.g.:
    #   python -m ludic.inference.vllm_server --model Qwen/Qwen3-0.6B --port 8000 ...
    asyncio.run(main())
