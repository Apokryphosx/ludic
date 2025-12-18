"""
Minimal Blackjack training scaffold (LoRA + GRPO-style grouped advantages).

Intended setup for 2 GPUs:
  - GPU0: vLLM server (inference)
  - GPU1: this script (training)

Example:
  # Terminal 1 (GPU0): start vLLM
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m ludic.inference.vllm_server \\
    --model Qwen/Qwen2.5-7B-Instruct

  # Terminal 2 (GPU1): train LoRA on Blackjack
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/blackjack/train_blackjack.py \\
    --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Callable, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from environments.blackjack import BlackjackEnv
from ludic.agent import Agent
from ludic.context import FullDialog, TruncatedThinkingContext
from ludic.distributed.adapters import create_vllm_publisher
from ludic.eval import EngineEvaluator
from ludic.inference import InferenceSpec, ReturnSpec, SamplingParams, VLLMChatClient
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import ParseResult, think_prefix_parser
from ludic.training import (
    CheckpointConfig,
    EnvSpec,
    GRPORequestStrategy,
    GroupNormalizedReturn,
    ProtocolSpec,
    RLAlgorithm,
    Reducer,
    ReinforceLoss,
    RichLiveLogger,
    RolloutBatchSource,
    RolloutEngine,
    RolloutRequest,
    Trainer,
    TrainerConfig,
)


_ACTION_ANYWHERE_RE = re.compile(r"\b(hit|stand|stay)\b", flags=re.IGNORECASE)
_ACTION_EXACT_RE = re.compile(r"^\s*(hit|stand|stay)\s*$", flags=re.IGNORECASE)


def blackjack_parser(raw: str) -> ParseResult:
    """
    Parse HIT/STAND actions.

    - If output is exactly HIT/STAND(/STAY), treat as success (no penalty).
    - Else, if HIT/STAND(/STAY) appears anywhere, extract the last occurrence and
      apply a small format penalty to discourage extra text.
    - Else, return a parse failure with a larger penalty.
    """
    cot = think_prefix_parser(raw, success_reward=0.0, error_reward=0.0)
    text = cot.action if cot.action is not None else raw

    exact = _ACTION_EXACT_RE.match(text.strip())
    if exact:
        action = exact.group(1).lower()
        return ParseResult(action=("STAND" if action == "stay" else action.upper()), reward=0.0, obs=None)

    matches = _ACTION_ANYWHERE_RE.findall(text)
    if matches:
        action = matches[-1].lower()
        action_out = "STAND" if action == "stay" else action.upper()
        return ParseResult(action=action_out, reward=-0.2, obs=None)

    return ParseResult(
        action=None,
        reward=-1.0,
        obs="Invalid action. Reply with HIT or STAND.",
    )


def build_requests_fn(
    rng: torch.Generator,
    batch_size: int,
    inference: InferenceSpec,
    *,
    decks: int,
    dealer_hits_soft_17: bool,
    system_prompt: str | None,
) -> Callable[[], List[RolloutRequest]]:
    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        for _ in range(int(batch_size)):
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(
                        kind="blackjack",
                        kwargs={
                            "decks": int(decks),
                            "dealer_hits_soft_17": bool(dealer_hits_soft_17),
                            "system_prompt": system_prompt,
                        },
                    ),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    env_seed=int(seed),
                    sampling_seed=int(seed),
                    inference=inference,
                    meta={"decks": int(decks), "dealer_hits_soft_17": bool(dealer_hits_soft_17)},
                )
            )
        return reqs

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model on Blackjack using Ludic + vLLM.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling episode seeds.")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1, help="Rollout requests per batch source call.")
    parser.add_argument("--train-steps", type=int, default=200, help="Number of trainer steps.")
    parser.add_argument("--max-steps-per-episode", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8, help="Group size for grouped advantages (GRPO-style).")
    parser.add_argument("--decks", type=int, default=1, help="Number of decks in the shoe.")
    parser.add_argument("--dealer-hits-soft-17", action="store_true", help="If set, dealer hits on soft 17.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (RL-friendly defaults).")
    parser.add_argument("--lora-alpha-mult", type=float, default=2.0, help="alpha = rank * mult.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--train-max-tokens", type=int, default=32, help="Max completion tokens per step (training).")
    parser.add_argument("--eval-every", type=int, default=20, help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True, help="Run eval once before training begins.")
    parser.add_argument("--eval-episodes", type=int, default=500, help="Number of episodes for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="Sampling temperature for eval passes.")
    parser.add_argument("--eval-max-tokens", type=int, default=32, help="Max completion tokens per step (eval).")
    parser.add_argument("--rollout-log", type=str, default="blackjack_train_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_blackjack", help="Checkpoint output directory.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Checkpoint every N steps (0 to disable).")
    parser.add_argument("--max-to-keep", type=int, default=2, help="Max checkpoints to keep.")
    parser.add_argument(
        "--ctx",
        choices=["full", "truncated"],
        default="full",
        help="Context strategy: 'full' (FullDialog) or 'truncated' (TruncatedThinkingContext).",
    )
    parser.add_argument("--final-save", action="store_true", help="Save a final checkpoint after training completes.")
    parser.add_argument("--positive-only", action="store_true", help="Only learn from positive advantages; clip negative ones to 0.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=BlackjackEnv.DEFAULT_SYSTEM_PROMPT + " Do not output any other text.",
        help="System prompt override (pass '' to disable).",
    )

    args = parser.parse_args()

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Seeds for deterministic episode resets
    rng = torch.Generator()
    rng.manual_seed(args.seed if args.seed is not None else 0)

    # Tokenizer + model (train on GPU1; vLLM runs separately on GPU0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(args.lora_rank),
        lora_alpha=int(int(args.lora_rank) * float(args.lora_alpha_mult)),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules="all-linear",
    )

    model = get_peft_model(base_model, lora_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()

    # Shared client for inference + weight updates
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    system_prompt = args.system_prompt if args.system_prompt != "" else None

    env_registry = {
        "blackjack": lambda decks=1, dealer_hits_soft_17=False, system_prompt=None: BlackjackEnv(
            decks=decks,
            dealer_hits_soft_17=dealer_hits_soft_17,
            system_prompt=system_prompt,
        )
    }

    def protocol_factory() -> SingleAgentSyncProtocol:
        ctx = (
            TruncatedThinkingContext(system_prompt=system_prompt)
            if args.ctx == "truncated"
            else FullDialog(system_prompt=system_prompt)
        )
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=ctx,
                parser=blackjack_parser,
            ),
            stop_on_parse_error=True,
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm: GRPO-style group baseline with REINFORCE loss
    algo = RLAlgorithm(
        name="grpo",
        credit_assigner=GroupNormalizedReturn(
            group_size=int(args.group_size),
            normalize_adv=True,
            positive_only=bool(args.positive_only),
        ),
        loss=ReinforceLoss(length_normalize=True),
    )

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )

    train_inference = InferenceSpec(
        sampling=SamplingParams(temperature=float(args.train_temperature), max_tokens=int(args.train_max_tokens)),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )
    base_requests_fn = build_requests_fn(
        rng,
        args.batch_size,
        train_inference,
        decks=args.decks,
        dealer_hits_soft_17=args.dealer_hits_soft_17,
        system_prompt=system_prompt,
    )

    def requests_fn() -> List[RolloutRequest]:
        return GRPORequestStrategy(group_size=int(args.group_size)).expand(base_requests_fn())

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=int(args.max_steps_per_episode),
        concurrency=int(args.concurrency),
    )

    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        grad_accum_steps=6,
        max_grad_norm=0.5,
        pad_token_id=int(tokenizer.pad_token_id),
        lr=5e-5,
        eval_at_start=bool(args.eval_before_start and args.eval_episodes and args.eval_episodes > 0),
        eval_every_n_steps=(int(args.eval_every) if args.eval_every and args.eval_every > 0 else None),
        eval_concurrency=int(args.eval_concurrency),
        eval_max_steps=int(args.max_steps_per_episode),
    )

    checkpoint_cfg = CheckpointConfig(
        output_dir=str(args.checkpoint_dir),
        every_n_steps=int(args.checkpoint_every),
        max_to_keep=int(args.max_to_keep),
        save_optimizer=True,
    )

    reducers = {
        "win_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v in {"player_win", "dealer_bust"},
            normalize_by="rollouts",
        ),
        "loss_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v in {"dealer_win", "player_bust"},
            normalize_by="rollouts",
        ),
        "push_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "push",
            normalize_by="rollouts",
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
        "avg_prompt_length": Reducer(
            kind="mean",
            source="prompt_length",
        ),
        "avg_completion_length": Reducer(
            kind="mean",
            source="completion_length",
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
            "push_rate",
            "bust_rate",
            "invalid_action_rate",
            "parse_error_rate",
            "truncated_rate",
            "avg_prompt_length",
            "avg_completion_length",
            "total_completion_tokens",
            "eval_win_rate",
            "eval_loss_rate",
            "eval_push_rate",
            "eval_bust_rate",
            "eval_invalid_action_rate",
            "eval_parse_error_rate",
            "eval_truncated_rate",
            "eval_avg_completion_tokens",
            "num_rollouts",
            "num_samples",
        ],
        spark_key="avg_total_reward",
        history=100,
        precision=4,
    )

    eval_reducers = {
        "win_rate": Reducer(kind="count_true", source="result", transform=lambda v: v in {"player_win", "dealer_bust"}, normalize_by="rollouts", as_percent=True),
        "loss_rate": Reducer(kind="count_true", source="result", transform=lambda v: v in {"dealer_win", "player_bust"}, normalize_by="rollouts", as_percent=True),
        "push_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "push", normalize_by="rollouts", as_percent=True),
        "bust_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "player_bust", normalize_by="rollouts", as_percent=True),
        "invalid_action_rate": Reducer(kind="count_true", source="invalid_action", normalize_by="rollouts", as_percent=True),
        "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="rollouts", as_percent=True),
        "truncated_rate": Reducer(kind="count_true", source="truncated", normalize_by="rollouts", as_percent=True),
        "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
    }

    evaluator = (
        None
        if not args.eval_episodes or args.eval_episodes <= 0
        else EngineEvaluator(
            engine=RolloutEngine(env_registry=env_registry, protocol_registry=protocol_registry),
            requests_fn=lambda: [
                RolloutRequest(
                    env=EnvSpec(
                        kind="blackjack",
                        kwargs={
                            "decks": int(args.decks),
                            "dealer_hits_soft_17": bool(args.dealer_hits_soft_17),
                            "system_prompt": system_prompt,
                        },
                    ),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    env_seed=int(seed),
                    sampling_seed=int(seed),
                    inference=InferenceSpec(
                        sampling=SamplingParams(temperature=float(args.eval_temperature), max_tokens=int(args.eval_max_tokens)),
                        return_=ReturnSpec.for_eval(return_token_ids=True),
                    ),
                    meta={"eval_seed": int(seed)},
                )
                for seed in range(int(args.eval_episodes))
            ],
            reducers=eval_reducers,
            max_steps=int(cfg.eval_max_steps),
            timeout_s=cfg.eval_timeout_s,
            concurrency=int(cfg.eval_concurrency),
        )
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
        evaluator=evaluator,
    )

    trainer.train_sync(int(args.train_steps))
    if args.final_save:
        try:
            trainer.save_checkpoint(metadata={"final": True})
        except RuntimeError:
            pass  # No checkpointer configured


if __name__ == "__main__":
    main()
