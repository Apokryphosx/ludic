"""
Tic-Tac-Toe LoRA RL training via Ludic + a running vLLM server (FSDP2-ready).

Launch (2 GPUs):
  torchrun --nproc_per_node=2 examples/tictactoe/train_tictactoe_lora_fsdp2_vllm.py

Config:
  By default this reads `examples/tictactoe/config.toml`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from torch import nn
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from environments.tic_tac_toe import TicTacToeEnv
from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.distributed.adapters import create_vllm_publisher
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import Parser, compose_parsers, cot_prefix_parser, xml_move_parser
from ludic.training.algorithm import RLAlgorithm
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.batching.synced_batching import RolloutBatchSource
from ludic.training.checkpoint import CheckpointConfig
from ludic.training.config import TrainerConfig
from ludic.training.credit_assignment import MonteCarloReturn
from ludic.training.loss import ReinforceBaselineLoss
from ludic.training.loggers import PrintLogger, RichLiveLogger
from ludic.training.stats import Reducer
from ludic.training.trainer import Trainer
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest


log = logging.getLogger("tictactoe_train")


class _NoopPublisher:
    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        return


def _load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table/dict, got {type(data)}")
    return data


def _get(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _normalize_resume_from(v: Any) -> int | str | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "null":
            return None
        if s.isdigit():
            return int(s)
        return s
    raise TypeError(f"checkpoint.resume_from must be int, str, or empty; got {type(v)}")


def _normalize_optional_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "null":
            return None
        return float(s)
    raise TypeError(f"Expected float/int/str/empty, got {type(v)}")


def _is_lora_param_name(name: str) -> bool:
    # PEFT LoRA params typically include "lora_A"/"lora_B"/"lora_" in the name.
    return "lora_" in name or "lora_A" in name or "lora_B" in name


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _model_param_stats(model: nn.Module) -> Dict[str, int]:
    bytes_total = 0
    bytes_base = 0
    bytes_lora = 0
    bytes_trainable = 0
    bytes_trainable_lora = 0
    bytes_trainable_base = 0

    count_total = 0
    count_base = 0
    count_lora = 0
    count_trainable = 0
    count_trainable_lora = 0
    count_trainable_base = 0

    for name, p in model.named_parameters():
        if not isinstance(p, torch.Tensor) or p.device.type != "cuda":
            continue
        n = int(p.numel())
        b = _tensor_nbytes(p)

        bytes_total += b
        count_total += n

        is_lora = _is_lora_param_name(name)
        if is_lora:
            bytes_lora += b
            count_lora += n
            if p.requires_grad:
                bytes_trainable_lora += b
                count_trainable_lora += n
        else:
            bytes_base += b
            count_base += n
            if p.requires_grad:
                bytes_trainable_base += b
                count_trainable_base += n

        if p.requires_grad:
            bytes_trainable += b
            count_trainable += n

    return {
        "bytes_total": bytes_total,
        "bytes_base": bytes_base,
        "bytes_lora": bytes_lora,
        "bytes_trainable": bytes_trainable,
        "bytes_trainable_base": bytes_trainable_base,
        "bytes_trainable_lora": bytes_trainable_lora,
        "count_total": count_total,
        "count_base": count_base,
        "count_lora": count_lora,
        "count_trainable": count_trainable,
        "count_trainable_base": count_trainable_base,
        "count_trainable_lora": count_trainable_lora,
    }


def _model_buffer_bytes(model: nn.Module) -> int:
    total = 0
    for b in model.buffers():
        if isinstance(b, torch.Tensor) and b.device.type == "cuda":
            total += _tensor_nbytes(b)
    return total


def _model_grad_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        g = getattr(p, "grad", None)
        if isinstance(g, torch.Tensor) and g.device.type == "cuda":
            total += _tensor_nbytes(g)
    return total


def _optimizer_state_bytes(optim: Optimizer) -> Dict[str, int]:
    total = 0
    exp_avg = 0
    exp_avg_sq = 0
    tensors = 0

    for state in optim.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in state.items():
            if not isinstance(v, torch.Tensor) or v.device.type != "cuda":
                continue
            b = _tensor_nbytes(v)
            total += b
            tensors += 1
            if k == "exp_avg":
                exp_avg += b
            elif k == "exp_avg_sq":
                exp_avg_sq += b

    return {
        "total": total,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "tensors": tensors,
    }


def _vram_breakdown_stats_mb(*, model: nn.Module, optimizer: Optimizer, device: torch.device) -> Dict[str, float]:
    mb = 1024**2
    out: Dict[str, float] = {}

    if device.type != "cuda" or not torch.cuda.is_available():
        return out

    p = _model_param_stats(model)
    bbytes = _model_buffer_bytes(model)
    gbytes = _model_grad_bytes(model)
    obytes = _optimizer_state_bytes(optimizer)

    alloc = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    free = None
    total = None
    try:
        free, total = torch.cuda.mem_get_info(device)  # type: ignore[arg-type]
    except Exception:
        pass

    out.update(
        {
            "model_param_mb_total": p["bytes_total"] / mb,
            "model_param_mb_base": p["bytes_base"] / mb,
            "model_param_mb_lora": p["bytes_lora"] / mb,
            "model_param_mb_trainable": p["bytes_trainable"] / mb,
            "model_param_count_total": float(p["count_total"]),
            "model_param_count_base": float(p["count_base"]),
            "model_param_count_lora": float(p["count_lora"]),
            "model_param_count_trainable": float(p["count_trainable"]),
            "model_buffer_mb_total": bbytes / mb,
            # Grad tensors are usually freed by Trainer.zero_grad(set_to_none=True) before logging,
            # so this is often 0; keep it for debugging.
            "grad_mb_allocated": gbytes / mb,
            # Expected gradient footprint if all trainable params had grads allocated.
            "grad_mb_expected_trainable": p["bytes_trainable"] / mb,
            "optim_state_mb_total": obytes["total"] / mb,
            "optim_state_mb_exp_avg": obytes["exp_avg"] / mb,
            "optim_state_mb_exp_avg_sq": obytes["exp_avg_sq"] / mb,
            "optim_state_tensors": float(obytes["tensors"]),
            "cuda_alloc_mb": alloc / mb,
            "cuda_reserved_mb": reserved / mb,
        }
    )

    if free is not None and total is not None:
        out["cuda_free_mb"] = float(free) / mb
        out["cuda_total_mb"] = float(total) / mb

    accounted = p["bytes_total"] + bbytes + gbytes + obytes["total"]
    out["cuda_other_alloc_mb"] = max(0.0, float(alloc - accounted) / mb)
    return out


class _VRAMBreakdownLogger:
    def __init__(self, inner: Any, *, model: nn.Module, optimizer: Optimizer, device: torch.device) -> None:
        self._inner = inner
        self._model = model
        self._optimizer = optimizer
        self._device = device

    def log(self, step: int, stats: Dict[str, float]) -> None:
        merged = dict(stats)
        merged.update(_vram_breakdown_stats_mb(model=self._model, optimizer=self._optimizer, device=self._device))
        self._inner.log(step, merged)


def _preflight_vram_check(
    *,
    cfg: Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
    rank: int,
) -> None:
    if rank != 0:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    mb = 1024**2
    param = _model_param_stats(model)
    buffer_bytes = _model_buffer_bytes(model)
    param_mb = param["bytes_total"] / mb
    buffer_mb = buffer_bytes / mb

    cuda_alloc_mb = float(torch.cuda.memory_allocated(device)) / mb
    cuda_reserved_mb = float(torch.cuda.memory_reserved(device)) / mb

    max_model_param_gb = _get(cfg, "preflight.max_model_param_gb", 20)
    max_model_param_mb = float(max_model_param_gb) * 1024.0

    print(
        "[preflight] "
        f"model_param_mb_total={param_mb:.1f} "
        f"(base={param['bytes_base']/mb:.1f}, lora={param['bytes_lora']/mb:.1f}) "
        f"model_param_count_total={param['count_total']} "
        f"(base={param['count_base']}, lora={param['count_lora']}, trainable={param['count_trainable']}) "
        f"model_buffer_mb_total={buffer_mb:.1f} "
        f"cuda_alloc_mb={cuda_alloc_mb:.1f} cuda_reserved_mb={cuda_reserved_mb:.1f}"
    )

    if param_mb > max_model_param_mb:
        raise SystemExit(
            f"Preflight failed: model parameters use {param_mb/1024.0:.2f} GiB on CUDA, "
            f"exceeds limit {float(max_model_param_gb):.2f} GiB. "
            "Choose a smaller model / lower precision / different placement."
        )


def _configure_logging(*, rank: int, level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for noisy in ("urllib3", "aiohttp", "httpx", "openai", "transformers"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def _silence_nonzero_ranks(*, rank: int) -> None:
    if rank == 0:
        return
    logging.getLogger().setLevel(logging.ERROR)
    for noisy in ("urllib3", "aiohttp", "httpx", "openai", "transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    sys.stdout = open(os.devnull, "w")


def _init_dist_if_needed(*, local_rank: int) -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        return 0, 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _wait_for_vllm_health(*, host: str, port: int, timeout_s: float) -> None:
    import requests

    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5)
    raise RuntimeError(f"vLLM health check failed at {url} (timeout={timeout_s}s). Last error: {last_err}")


def _build_tictactoe_parser(max_tokens: int) -> Parser:
    # Repo no longer ships token_guard_parser; vLLM-side `max_tokens` already bounds length.
    # Keep parsing strict on format (<think>...</think> + <move>...</move>).
    return compose_parsers(cot_prefix_parser, xml_move_parser)


async def _run_eval(
    *,
    seeds: List[int],
    host: str,
    port: int,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    max_steps: int,
    timeout_s: float | None,
) -> float:
    sem = asyncio.Semaphore(max(1, concurrency))
    parser = _build_tictactoe_parser(max_tokens)
    client = VLLMChatClient(host=host, port=port, enable_weight_updates=False)

    async def _run_one(seed: int) -> bool:
        async with sem:
            env = TicTacToeEnv(agent_starts=True)
            base_prompt = env.suggested_sysprompt or ""
            sys_prompt = (
                base_prompt
                + "\n\nThink in <think>...</think>, then output your move as exactly one XML tag, e.g. <move>A1</move>."
            )
            protocol = SingleAgentSyncProtocol(
                agent=Agent(
                    client=client,
                    model=model,
                    ctx=FullDialog(),
                    parser=parser,
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
                timeout_s=timeout_s,
            )
            info = rollouts[0].steps[-1].info
            return info.get("result") == "win"

    results = await asyncio.gather(*[_run_one(s) for s in seeds])
    wins = sum(1 for r in results if r)
    return 100.0 * wins / len(seeds) if seeds else 0.0


def _build_requests_fn(
    *,
    rng: torch.Generator,
    batch_size: int,
    sampling_args: Dict[str, Any],
) -> Any:
    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        for _ in range(batch_size):
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
                    protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                    num_episodes=1,
                    seed=seed,
                    sampling_args=sampling_args,
                )
            )
        return reqs

    return _fn


def _find_transformer_blocks(model: torch.nn.Module) -> Optional[Any]:
    m = model
    for attr in ("base_model", "model", "transformer"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "layers"):
        return getattr(m, "layers")
    if hasattr(m, "model") and hasattr(getattr(m, "model"), "layers"):
        return getattr(getattr(m, "model"), "layers")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "examples" / "tictactoe" / "config.toml"),
        help="Path to TOML config. Defaults to examples/tictactoe/config.toml",
    )
    args = parser.parse_args()

    cfg = _load_toml(args.config)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank, world_size = _init_dist_if_needed(local_rank=local_rank)

    _configure_logging(rank=rank, level=str(_get(cfg, "logging.level", "INFO")))
    if bool(_get(cfg, "logging.rank0_only_output", True)):
        _silence_nonzero_ranks(rank=rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if rank == 0:
        log.info("Loaded config: %s", args.config)
        log.info("Distributed: world_size=%s local_rank=%s device=%s", world_size, local_rank, device)

    # ---- vLLM server check (rank0) ----
    vllm_host = str(_get(cfg, "vllm.host", "127.0.0.1"))
    vllm_port = int(_get(cfg, "vllm.port", 8000))
    vllm_health_timeout_s = float(_get(cfg, "vllm.health_timeout_s", 30.0))
    if rank == 0:
        _wait_for_vllm_health(host=vllm_host, port=vllm_port, timeout_s=vllm_health_timeout_s)
        log.info("vLLM server healthy at http://%s:%s", vllm_host, vllm_port)
    _barrier()

    # ---- output paths ----
    rollout_log = str(_get(cfg, "rollouts.jsonl_path", "tictactoe_train_rollouts.jsonl"))
    rollout_log_path = Path(rollout_log)
    if world_size > 1:
        rollout_log_path = rollout_log_path.with_suffix(f".rank{rank}.jsonl")
    if rank == 0:
        rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
    _barrier()
    rollout_log_path.touch(exist_ok=True)

    checkpoint_dir = str(_get(cfg, "checkpoint.output_dir", "checkpoints_tictactoe_lora"))
    if rank == 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    _barrier()

    # ---- RNG (per-rank) ----
    base_seed = int(_get(cfg, "rollouts.seed", 0))
    rng = torch.Generator()
    rng.manual_seed(base_seed + rank)

    # ---- model + tokenizer ----
    model_name = str(_get(cfg, "model.name", "Qwen/Qwen2.5-7B-Instruct"))
    trust_remote_code = bool(_get(cfg, "model.trust_remote_code", True))
    dtype_str = str(_get(cfg, "model.dtype", "bf16")).lower()
    dtype = torch.bfloat16 if dtype_str in ("bf16", "bfloat16") else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": "cpu"} if (world_size > 1 and device.type == "cuda") else None,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    lora_rank = int(_get(cfg, "lora.rank", 16))
    lora_alpha_mult = float(_get(cfg, "lora.alpha_mult", 2.0))
    lora_dropout = float(_get(cfg, "lora.dropout", 0.0))
    lora_target_modules = _get(cfg, "lora.target_modules", "all-linear")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=int(lora_rank * lora_alpha_mult),
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(base_model, lora)
    if rank == 0:
        model.print_trainable_parameters()

    # ---- FSDP2 wrapping (when torchrun world_size>1) ----
    if world_size > 1 and device.type == "cuda":
        mp_policy = fsdp.MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
        )
        blocks = _find_transformer_blocks(model)
        if blocks is not None:
            for layer in blocks:
                fsdp.fully_shard(layer, mp_policy=mp_policy)
        fsdp.fully_shard(model, mp_policy=mp_policy)
    else:
        model.to(device)

    _preflight_vram_check(cfg=cfg, model=model, device=device, rank=rank)

    # ---- vLLM client + publisher ----
    vllm_group_port = int(_get(cfg, "vllm.group_port", 51216))
    client = VLLMChatClient(
        host=vllm_host,
        port=vllm_port,
        group_port=vllm_group_port,
        enable_weight_updates=(rank == 0 and bool(_get(cfg, "vllm.enable_weight_updates", True))),
        device=str(device),
    )
    publisher = create_vllm_publisher(client) if (rank == 0 and client.enable_weight_updates) else _NoopPublisher()

    # ---- env/protocol registries ----
    max_tokens = int(_get(cfg, "rollouts.max_tokens", 250))
    action_parser = _build_tictactoe_parser(max_tokens)

    env_registry = {"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)}

    def protocol_factory() -> SingleAgentSyncProtocol:
        base_prompt = TicTacToeEnv().suggested_sysprompt or ""
        prompt = (
            base_prompt
            + "\n\nThink in <think>...</think>, then output your move as exactly one XML tag, e.g. <move>A1</move>."
        )
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=model_name,
                ctx=FullDialog(),
                parser=action_parser,
            ),
            prompt=prompt,
        )

    protocol_registry = {"single_agent": protocol_factory}

    # ---- algorithm ----
    gamma = float(_get(cfg, "algo.gamma", 1.0))
    normalize_adv = bool(_get(cfg, "algo.normalize_advantages", True))
    algo = RLAlgorithm(
        name="reinforce_baseline",
        credit_assigner=MonteCarloReturn(gamma=gamma),
        loss=ReinforceBaselineLoss(
            normalize=normalize_adv,
            length_normalize=True,
        ),
    )

    # ---- rollouts ----
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(rollout_log_path),
    )
    train_temperature = float(_get(cfg, "rollouts.temperature", 1.0))
    sampling_args = {
        "temperature": train_temperature,
        "max_tokens": max_tokens,
        "extras": {"extra_body": {"return_token_ids": True}},
    }
    batch_size = int(_get(cfg, "rollouts.batch_size", 4))
    concurrency = int(_get(cfg, "rollouts.concurrency", 8))
    max_steps_per_episode = int(_get(cfg, "rollouts.max_steps_per_episode", 5))
    requests_fn = _build_requests_fn(rng=rng, batch_size=batch_size, sampling_args=sampling_args)
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=max_steps_per_episode,
        concurrency=concurrency,
        retokenize=False,
    )

    # ---- trainer ----
    trainer_cfg = TrainerConfig(
        model_device=str(device),
        runtime_device=str(device),
        grad_accum_steps=int(_get(cfg, "trainer.grad_accum_steps", 8)),
        max_grad_norm=float(_get(cfg, "trainer.max_grad_norm", 0.5)),
        pad_token_id=int(tokenizer.pad_token_id),
        lr=float(_get(cfg, "trainer.lr", 5e-5)),
        weight_decay=float(_get(cfg, "trainer.weight_decay", 0.01)),
        sync_every_steps=int(_get(cfg, "trainer.sync_every_steps", 5)),
        reduce_stats_across_ranks=(world_size > 1),
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir=checkpoint_dir,
        every_n_steps=int(_get(cfg, "checkpoint.every_n_steps", 25)),
        max_to_keep=int(_get(cfg, "checkpoint.max_to_keep", 2)),
        save_optimizer=bool(_get(cfg, "checkpoint.save_optimizer", True)),
    )

    reducers = {
        "win_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "win", normalize_by="rollouts"),
        "loss_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "loss", normalize_by="rollouts"),
        "draw_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "draw", normalize_by="rollouts"),
        "illegal_rate": Reducer(kind="count_true", source="illegal_move", normalize_by="samples"),
        "parse_err_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples"),
        "total_completion_tokens": Reducer(kind="sum", source="completion_length"),
    }

    train_logger = None
    if rank == 0:
        logger_kind = str(_get(cfg, "logging.logger", "print")).lower()
        include_gpu_memory = bool(_get(cfg, "logging.gpu_memory", True))
        include_vram_breakdown = bool(_get(cfg, "logging.vram_breakdown", True))
        keys = [
            "loss",
            "avg_total_reward",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "illegal_rate",
            "parse_err_rate",
            "avg_completion_length",
            "total_completion_tokens",
            "num_rollouts",
            "num_samples",
        ]
        if include_gpu_memory:
            keys.extend(
                [
                    "gpu_mem_alloc_mb",
                    "gpu_mem_reserved_mb",
                    "gpu_mem_peak_mb",
                    "gpu_activation_peak_mb",
                    "gpu_forward_peak_mb",
                    "gpu_forward_activation_peak_mb",
                    "gpu_backward_peak_mb",
                    "gpu_backward_activation_peak_mb",
                ]
            )
        if include_vram_breakdown:
            keys.extend(
                [
                    "model_param_mb_total",
                    "model_param_mb_base",
                    "model_param_mb_lora",
                    "model_param_mb_trainable",
                    "model_param_count_total",
                    "model_param_count_base",
                    "model_param_count_lora",
                    "model_param_count_trainable",
                    "model_buffer_mb_total",
                    "grad_mb_allocated",
                    "grad_mb_expected_trainable",
                    "optim_state_mb_total",
                    "optim_state_mb_exp_avg",
                    "optim_state_mb_exp_avg_sq",
                    "optim_state_tensors",
                    "cuda_alloc_mb",
                    "cuda_reserved_mb",
                    "cuda_free_mb",
                    "cuda_total_mb",
                    "cuda_other_alloc_mb",
                ]
            )

        if logger_kind == "none":
            train_logger = None
        elif logger_kind == "rich" and sys.stdout.isatty():
            train_logger = RichLiveLogger(
                keys=keys,
                spark_key="avg_total_reward",
                history=int(_get(cfg, "logging.history", 100)),
                precision=int(_get(cfg, "logging.precision", 4)),
            )
        else:
            train_logger = PrintLogger(
                prefix="[trainer]",
                keys=keys,
                precision=int(_get(cfg, "logging.precision", 4)),
            )

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        cfg=trainer_cfg,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
        resume_from=_normalize_resume_from(_get(cfg, "checkpoint.resume_from", None)),
    )
    if rank == 0 and trainer.train_logger is not None and bool(_get(cfg, "logging.vram_breakdown", True)):
        trainer.train_logger = _VRAMBreakdownLogger(
            trainer.train_logger,
            model=model,
            optimizer=trainer.optimizer,
            device=device,
        )

    # ---- eval config ----
    do_eval = bool(_get(cfg, "eval.enabled", True))
    eval_every = int(_get(cfg, "eval.every_n_steps", 10))
    eval_before = bool(_get(cfg, "eval.before_start", True))
    eval_episodes = int(_get(cfg, "eval.episodes", 200))
    eval_concurrency = int(_get(cfg, "eval.concurrency", 32))
    eval_temperature = float(_get(cfg, "eval.temperature", 0.6))
    eval_timeout_s = _normalize_optional_float(_get(cfg, "eval.timeout_s", None))

    train_steps = int(_get(cfg, "trainer.train_steps", 100))

    async def train_loop() -> None:
        eval_seeds = list(range(eval_episodes))

        if do_eval and eval_before and rank == 0 and eval_seeds:
            acc = await _run_eval(
                seeds=eval_seeds,
                host=vllm_host,
                port=vllm_port,
                model=model_name,
                concurrency=eval_concurrency,
                max_tokens=max_tokens,
                temperature=eval_temperature,
                max_steps=max_steps_per_episode,
                timeout_s=eval_timeout_s,
            )
            print(f"[eval @ step 0] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")
        _barrier()

        for _ in range(train_steps):
            stats = await trainer.train_step()
            train_step = int(stats["train_step"])

            if do_eval and eval_every > 0 and eval_seeds and train_step % eval_every == 0:
                _barrier()
                if rank == 0:
                    acc = await _run_eval(
                        seeds=eval_seeds,
                        host=vllm_host,
                        port=vllm_port,
                        model=model_name,
                        concurrency=eval_concurrency,
                        max_tokens=max_tokens,
                        temperature=eval_temperature,
                        max_steps=max_steps_per_episode,
                        timeout_s=eval_timeout_s,
                    )
                    print(f"[eval @ step {train_step}] win_rate={acc:.2f}% on {len(eval_seeds)} episodes")
                _barrier()

    try:
        asyncio.run(train_loop())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
