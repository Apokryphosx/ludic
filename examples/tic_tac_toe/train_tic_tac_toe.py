"""
Minimal Tic-Tac-Toe training scaffold.

This wires together:
  - TicTacToeEnv single-agent episodes
  - SingleAgentSyncProtocol with a shared VLLMChatClient
  - RolloutBatchSource + GroupNormalizedReturn credit
  - Trainer with REINFORCE loss
  - Optional periodic eval of win rate
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from environments.tic_tac_toe import TicTacToeEnv
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient
from ludic.interaction import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher
from ludic.parsers import compose_parsers, think_prefix_parser, xml_tag_parser
from ludic.training import (
    RLAlgorithm,
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
    GroupNormalizedReturn,
    ReinforceLoss,
)
from ludic.training import Reducer, RichLiveLogger

# STRICT: require <think>...</think> then exactly one <move>...</move>.
TICTACTOE_PARSER = compose_parsers(think_prefix_parser, xml_tag_parser("move", exact=True))


def _load_toml_config(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "TOML config requested but `tomllib` is unavailable; use Python 3.11+ or vendor a TOML parser."
        ) from e

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Allow either a flat config file or a sectioned one.
    if isinstance(data.get("train"), dict):
        return dict(data["train"])
    return dict(data)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model on Tic-Tac-Toe using Ludic + vLLM.")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a TOML file with defaults for all flags (supports either flat keys or a [train] table).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling episode seeds.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout requests per batch source call.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Number of rollouts per GRPO group (defaults to --batch-size).",
    )
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=5)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (RL-friendly defaults from LoRA best-practice guides).",
    )
    parser.add_argument(
        "--lora-alpha-mult",
        type=float,
        default=2.0,
        help="Multiplier applied to rank to set lora_alpha (alpha = rank * mult).",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps.")
    parser.add_argument(
        "--eval-before-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run eval once before training begins.",
    )
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of episodes for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.6, help="Sampling temperature for eval passes.")
    parser.add_argument("--rollout-log", type=str, default="tictactoe_train_rollouts.jsonl")
    return parser


def _parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config-file", type=str, default=None)
    cfg_args, remaining = base.parse_known_args()

    parser = _build_arg_parser()
    if cfg_args.config_file:
        config = _load_toml_config(cfg_args.config_file)
        # CLI flags still override config values.
        parser.set_defaults(**config)

    return parser.parse_args(remaining)


async def run_eval(
    *,
    seeds: List[int],
    client: VLLMChatClient,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_steps: int,
) -> float:
    """
    Run a batch of Tic-Tac-Toe episodes and report win rate (%).
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(seed: int) -> bool:
        async with sem:
            env = TicTacToeEnv(agent_starts=True)
            base_prompt = env.suggested_sysprompt or ""
            sys_prompt = (
                base_prompt
                + "\n\nThink through the board in <think>...</think>. After </think>, output exactly one XML tag of the form <move>A1</move> and nothing else."
            )
            protocol = SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=model,
                    ctx=FullDialog(),
                    parser=TICTACTOE_PARSER,
                ),
                prompt=sys_prompt,
            )
            rollouts = await protocol.run(
                env=env,
                max_steps=max_steps,
                seed=seed,
                sampling_args={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extras": {"extra_body": {"return_token_ids": True}},
                },
            )
            info = rollouts[0].steps[-1].info
            return info.get("result") == "win"

    results = await asyncio.gather(*[_run_one(s) for s in seeds])
    wins = sum(1 for r in results if r)
    return 100.0 * wins / len(seeds) if seeds else 0.0


def build_requests_fn(
    rng: torch.Generator,
    batch_size: int,
    sampling_args: Dict[str, Any],
):
    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        # GRPO-style grouping: sample multiple rollouts for the same env seed/prompt
        # so GroupNormalizedReturn can compute a within-group baseline.
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        group_id = f"tictactoe_seed_{seed}"
        for _ in range(batch_size):
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    seed=int(seed),
                    sampling_args=sampling_args,
                    meta={"group_id": group_id},
                )
            )
        return reqs

    return _fn


def main():
    args = _parse_args()

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Seeds for deterministic episode resets
    rng = torch.Generator()
    rng.manual_seed(args.seed if args.seed is not None else 0)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Apply a lightweight LoRA adapter to train only a small subset of params.
    # Apply LoRA to all linear projections (per “LoRA Without Regret” guidance: all-linear > attention-only).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=int(args.lora_rank * args.lora_alpha_mult),
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Registries
    env_registry = {"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)}

    def protocol_factory():
        # Extend the env's suggested system prompt with explicit CoT + XML move instructions.
        base_prompt = TicTacToeEnv().suggested_sysprompt or ""
        prompt = (
            base_prompt
            + "\n\nThink through the board in <think>...</think>. After </think>, output exactly one XML tag of the form <move>A1</move> and nothing else."
        )
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(),
                parser=TICTACTOE_PARSER,
            ),
            prompt=prompt,
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm
    group_size = int(args.group_size) if args.group_size is not None else int(args.batch_size)
    algo = RLAlgorithm(
        name="grpo",
        credit_assigner=GroupNormalizedReturn(group_size=group_size, normalize_adv=True),
        loss=ReinforceLoss(length_normalize=True),
    )

    # Engine + batch source
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )
    sampling_args = {
        "temperature": args.train_temperature,
        "max_tokens": 250,
        "extras": {"extra_body": {"return_token_ids": True}},
    }
    requests_fn = build_requests_fn(rng, args.batch_size, sampling_args)
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=args.max_steps_per_episode,
        concurrency=args.concurrency,
        retokenize=False,
    )

    # Trainer
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        grad_accum_steps=8,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
        lr=5e-5,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_tictactoe",
        every_n_steps=25,
        max_to_keep=2,
        save_optimizer=True,
    )
    reducers = {
        "win_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "win",
            normalize_by="rollouts",
        ),
        "loss_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "loss",
            normalize_by="rollouts",
        ),
        "draw_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "draw",
            normalize_by="rollouts",
        ),
        "illegal_rate": Reducer(
            kind="count_true",
            source="illegal_move",
            normalize_by="samples",
        ),
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }

    train_logger = RichLiveLogger(
        keys=[
            "loss",
            "avg_total_reward",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "illegal_rate",
            "avg_completion_length",
            "total_completion_tokens",
            "num_rollouts",
            "num_samples",
        ],
        spark_key="avg_total_reward",
        history=100,
        precision=4,
    )

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
    )

    async def train_loop():
        train_step = 0
        eval_seeds = list(range(args.eval_episodes))
        if args.eval_before_start and eval_seeds:
            acc = await run_eval(
                seeds=eval_seeds,
                client=client,
                model=args.model,
                concurrency=args.eval_concurrency,
                max_tokens=250,
                temperature=args.eval_temperature,
                max_steps=args.max_steps_per_episode,
            )
            print(f"[eval @ step 0] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")

        for _ in range(args.train_steps):
            stats = await trainer.train_step()
            train_step = int(stats["train_step"])
            if args.eval_every > 0 and train_step % args.eval_every == 0 and eval_seeds:
                acc = await run_eval(
                    seeds=eval_seeds,
                    client=client,
                    model=args.model,
                    concurrency=args.eval_concurrency,
                    max_tokens=250,
                    temperature=args.eval_temperature,
                    max_steps=args.max_steps_per_episode,
                )
                print(f"[eval @ step {train_step}] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")

    asyncio.run(train_loop())


if __name__ == "__main__":
    main()
