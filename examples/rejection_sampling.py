from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.interaction import InteractionProtocol, SingleAgentSyncProtocol
from ludic.training import RolloutEngine, EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout, SamplingArgs

from environments.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Rejection Sampling Config
MIN_REWARD_THRESHOLD = 1.0  # Only keep wins (+1.0)
OUT_PATH = Path("data/tictactoe_winners_only.jsonl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_system_prompt(env_cls=TicTacToeEnv) -> str:
    """
    Take the env's suggested_sysprompt and append the XML contract line.
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


def rollout_to_json_dict(r: Rollout) -> Dict[str, Any]:
    """
    Serializable schema for a rollout.
    """
    return {
        "id": r.id,
        "meta": r.meta,
        "steps": [
            {
                "index": s.index,
                "prev_obs": s.prev_obs,
                "action": s.action,
                "next_obs": s.next_obs,
                "reward": s.reward,
                "truncated": s.truncated,
                "terminated": s.terminated,
                "info": s.info,
                "ts_ns": s.ts_ns,
            }
            for s in r.steps
        ],
        "total_reward": r.total_reward,
        "length": r.length,
        "duration_ns": r.duration_ns,
    }

# ---------------------------------------------------------------------------
# Main Generation Logic
# ---------------------------------------------------------------------------

async def generate_filtered_data(
    *,
    episodes: int = 50,
    max_steps: int = 9,
    concurrency: int = 8,
) -> None:
    print(f"Connecting to vLLM at http://{VLLM_HOST}:{VLLM_PORT}...")

    # 1. Setup Client (Shared Resource)
    # The client handles HTTP/NCCL and is stateless regarding conversation history,
    # so it can be shared across agents.
    client = VLLMChatClient(
        host=VLLM_HOST,
        port=VLLM_PORT,
        connection_timeout_s=300.0,
        enable_weight_updates=False, 
    )

    # 2. Define Factory Methods
    def create_single_agent_protocol(prompt: str) -> InteractionProtocol:
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,           # Shared client
                model=MODEL_NAME,        # Same model
                ctx=FullDialog(),  # fresh context; prompt is set via protocol
                parser=xml_move_parser,  # Stateless parser function
            ),
            prompt=prompt,
        )

    # 3. Setup Registries
    env_registry = {
        "tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)
    }
    
    protocol_registry = {
        "single_agent": create_single_agent_protocol
    }

    # 4. Initialize Engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=None, # Disable auto-logging to allow manual filtering
    )

    # 5. Create Rollout Requests
    # Note: We pass 'system_prompt' via kwargs to the protocol spec, 
    # which flows into 'create_single_agent_protocol'.
    prompt_text = build_system_prompt()
    sampling_args: SamplingArgs = {"temperature": 0.8, "max_tokens": 512}

    req_agent_starts = RolloutRequest(
        env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
        protocol=ProtocolSpec(
            kind="single_agent", 
            kwargs={"prompt": prompt_text}
        ),
        num_episodes=episodes // 2,
        sampling_args=sampling_args,
        meta={"setup": "agent_starts"},
    )

    req_opp_starts = RolloutRequest(
        env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": False}),
        protocol=ProtocolSpec(
            kind="single_agent", 
            kwargs={"prompt": prompt_text}
        ),
        num_episodes=episodes - (episodes // 2),
        sampling_args=sampling_args,
        meta={"setup": "opponent_starts"},
    )

    print(f"Running {episodes} episodes with concurrency={concurrency}...")
    
    # 6. Execute
    rollouts = await engine.generate_rollouts(
        requests=[req_agent_starts, req_opp_starts],
        max_steps=max_steps,
        concurrency=concurrency,
    )

    # 7. Rejection Sampling & Saving
    accepted_count = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rollouts:
            # Rejection Sampling Logic:
            # Only keep episodes where the agent won (Reward >= 1.0)
            if r.total_reward >= MIN_REWARD_THRESHOLD:
                payload = rollout_to_json_dict(r)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                accepted_count += 1

    # 8. Stats
    print(f"Total Rollouts: {len(rollouts)}")
    print(f"Accepted (Reward >= {MIN_REWARD_THRESHOLD}): {accepted_count}")
    print(f"Rejection Rate: {100 * (1 - accepted_count/len(rollouts)):.1f}%")
    print(f"Saved to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_filtered_data())
