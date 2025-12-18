"""
Eval a vLLM-served model on Blackjack.

Example:
    PYTHONPATH=. uv run python examples/blackjack/eval_blackjack_vllm.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --host 127.0.0.1 --port 8000 \\
        --episodes 500
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Mapping

from environments.blackjack import BlackjackEnv
from ludic.context import FullDialog, TruncatedThinkingContext
from ludic.eval.cli import (
    add_common_eval_args,
    build_single_agent_engine,
    inference_spec_from_cli,
    maybe_start_vllm,
    write_jsonl,
)
from ludic.eval.core import run_eval_sync
from ludic.inference import VLLMChatClient
from ludic.parsers import ParseResult, think_prefix_parser
from ludic.training import EnvSpec, ProtocolSpec, Reducer, RolloutRequest


_ACTION_RE = re.compile(r"\b(hit|stand|stay)\b", flags=re.IGNORECASE)


def blackjack_parser(raw: str) -> ParseResult:
    """
    Extract Blackjack actions from model output.

    - Optionally strips a leading <think>...</think> prefix (no penalty if missing)
    - Chooses the last occurrence of HIT/STAND/STAY in the remaining text
    """
    cot = think_prefix_parser(raw, success_reward=0.0, error_reward=0.0)
    text = cot.action if cot.action is not None else raw

    matches = _ACTION_RE.findall(text)
    if not matches:
        return ParseResult(
            action=None,
            reward=-1.0,
            obs="Could not parse action. Reply with HIT or STAND.",
        )

    action = matches[-1].lower()
    if action == "stay":
        action = "stand"
    return ParseResult(action=action.upper(), reward=0.0, obs=None)


def _terminal_total_reward(rec: Mapping[str, object]) -> object:
    if rec.get("terminated") or rec.get("truncated"):
        return rec.get("total_reward")
    return None


BLACKJACK_REDUCERS: Dict[str, Reducer] = {
    "win_rate": Reducer(
        kind="count_true",
        source="result",
        transform=lambda v: v in {"player_win", "dealer_bust"},
        normalize_by="rollouts",
        as_percent=True,
    ),
    "loss_rate": Reducer(
        kind="count_true",
        source="result",
        transform=lambda v: v in {"dealer_win", "player_bust"},
        normalize_by="rollouts",
        as_percent=True,
    ),
    "push_rate": Reducer(
        kind="count_true",
        source="result",
        transform=lambda v: v == "push",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "bust_rate": Reducer(
        kind="count_true",
        source="result",
        transform=lambda v: v == "player_bust",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "invalid_action_rate": Reducer(
        kind="count_true",
        source="invalid_action",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "parse_error_rate": Reducer(
        kind="count_true",
        source="parse_error",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "truncated_rate": Reducer(
        kind="count_true",
        source="truncated",
        normalize_by="rollouts",
        as_percent=True,
    ),
    "avg_steps": Reducer(
        kind="count_true",
        source="rollout_id",
        normalize_by="rollouts",
    ),
    "avg_episode_reward": Reducer(kind="mean", source=_terminal_total_reward),
    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
}


def make_requests(episodes: int, args: argparse.Namespace) -> List[RolloutRequest]:
    inf = inference_spec_from_cli(args)
    return [
        RolloutRequest(
            env=EnvSpec(
                kind="blackjack",
                kwargs={
                    "decks": int(args.decks),
                    "dealer_hits_soft_17": bool(args.dealer_hits_soft_17),
                    "system_prompt": args.system_prompt,
                },
            ),
            protocol=ProtocolSpec(kind="single_agent"),
            env_seed=seed,
            sampling_seed=seed,
            inference=inf,
            num_episodes=1,
            meta={"seed": seed},
        )
        for seed in range(int(episodes))
    ]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on Blackjack.")
    add_common_eval_args(parser)
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    parser.add_argument("--decks", type=int, default=1, help="Number of decks in the shoe.")
    parser.add_argument(
        "--dealer-hits-soft-17",
        action="store_true",
        help="If set, dealer hits on soft 17 (otherwise stands).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=BlackjackEnv.DEFAULT_SYSTEM_PROMPT + " Do not output any other text.",
        help="System prompt override (pass '' to disable).",
    )
    parser.add_argument(
        "--ctx",
        choices=["full", "truncated"],
        default="full",
        help="Context strategy: 'full' (FullDialog) or 'truncated' (TruncatedThinkingContext).",
    )
    parser.set_defaults(temperature=0.0, max_tokens=32, max_steps=20, out="blackjack_eval.jsonl")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    with maybe_start_vllm(args):
        client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=False)

        ctx_factory = (
            (lambda sp: TruncatedThinkingContext(system_prompt=sp))
            if args.ctx == "truncated"
            else (lambda sp: FullDialog(system_prompt=sp))
        )

        engine = build_single_agent_engine(
            client=client,
            model=args.model,
            parser=blackjack_parser,
            env_registry={
                "blackjack": lambda decks=1, dealer_hits_soft_17=False, system_prompt=None: BlackjackEnv(
                    decks=decks,
                    dealer_hits_soft_17=dealer_hits_soft_17,
                    system_prompt=system_prompt,
                )
            },
            system_prompt=args.system_prompt,
            context_factory=ctx_factory,
            stop_on_parse_error=True,
        )

        requests = make_requests(args.episodes, args)

        records, metrics = run_eval_sync(
            engine=engine,
            requests=requests,
            reducers=BLACKJACK_REDUCERS,
            max_steps=args.max_steps,
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
        )

        print("\n---- Blackjack Evaluation ----")
        for k, v in metrics.items():
            reducer = BLACKJACK_REDUCERS.get(k)
            if reducer is not None and reducer.as_percent:
                print(f"{k}={float(v):.2%}")
            else:
                print(f"{k}={float(v):.4g}")

        if args.out:
            write_jsonl(args.out, records)
            print(f"Wrote {len(records)} step records to {args.out}")


if __name__ == "__main__":
    main()

