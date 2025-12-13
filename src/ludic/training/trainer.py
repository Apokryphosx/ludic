from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Mapping

import torch
from torch import nn, optim, Tensor
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

from ludic.distributed.interfaces import PolicyPublisher
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.training.algorithm import RLAlgorithm
from ludic.training.loggers import TrainingLogger
from ludic.training.config import TrainerConfig
from ludic.training.stats import aggregate_stats, Reducer
from ludic.training.types import SAWBatch, SAWItem, BatchSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collation: SAWItems -> tensor batch
# ---------------------------------------------------------------------------

def _collate_saw_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Collate a list of SAWItem into a simple dense batch of tensors.

    - Left-aligns sequences and pads to max length in this batch.
    - Returns a dict suitable for RLAlgorithm.loss.compute():

          {
              "input_ids":      [B, T] long,
              "attention_mask": [B, T] long,
              "action_mask":    [B, T] float,
              "weight":         [B]    float,
          }

    If you want microbatching or a DataLoader, you can refactor this later.
    """
    if not items:
        raise ValueError("Cannot collate empty list of SAWItems")

    if max_seq_len is not None and max_seq_len > 0:
        lengths = [min(len(it.input_ids), max_seq_len) for it in items]
    else:
        lengths = [len(it.input_ids) for it in items]
    max_len = max(lengths)

    input_ids_list: List[Tensor] = []
    attn_mask_list: List[Tensor] = []
    action_mask_list: List[Tensor] = []
    weights_list: List[Tensor] = []

    for it in items:
        # Optional truncation: keep the most recent tokens (right side).
        # This preserves the action tail in [state || action].
        if max_seq_len is not None and max_seq_len > 0 and len(it.input_ids) > max_seq_len:
            input_ids = it.input_ids[-max_seq_len:]
            attention_mask = it.attention_mask[-max_seq_len:]
            action_mask = it.action_mask[-max_seq_len:]
        else:
            input_ids = it.input_ids
            attention_mask = it.attention_mask
            action_mask = it.action_mask

        L = len(input_ids)

        ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
        am = torch.zeros((max_len,), dtype=torch.long)
        actm = torch.zeros((max_len,), dtype=torch.float32)

        ids[:L] = torch.tensor(input_ids, dtype=torch.long)
        am[:L] = torch.tensor(attention_mask, dtype=torch.long)
        actm[:L] = torch.tensor(action_mask, dtype=torch.float32)

        input_ids_list.append(ids)
        attn_mask_list.append(am)
        action_mask_list.append(actm)
        weights_list.append(torch.tensor(it.weight, dtype=torch.float32))

    batch = {
        "input_ids": torch.stack(input_ids_list, dim=0).to(device),
        "attention_mask": torch.stack(attn_mask_list, dim=0).to(device),
        "action_mask": torch.stack(action_mask_list, dim=0).to(device),
        "weight": torch.stack(weights_list, dim=0).to(device),
    }
    return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates the training loop with gradient accumulation:

        Loop `grad_accum_steps`:
            BatchSource.next_batch() → micro-batch
              ↓
            collate → tensors
              ↓
            RLAlgorithm.compute_loss(model, batch)
              ↓
            scaled_loss.backward() (with FSDP.no_sync())
          ↓
        optimizer.step()
          ↓
        optimizer.zero_grad()  # <-- Grads freed
          ↓
        publisher.publish(...)     # <-- Sync

    Trainer is agnostic to envs, contexts, rollouts, and tokenization.

    This variant is FSDP-aware:

      - If `model` is wrapped in FSDP, it will:
          * on rank 0 only:
                - switch to FULL_STATE_DICT
                - gather a full state dict (no CPU offload by default)
                - push the full (unsharded) params to the runtime

      - On non-FSDP models, it just uses `named_parameters()` as before.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        publisher: PolicyPublisher,
        cfg: TrainerConfig = TrainerConfig(),
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        enable_gradient_checkpointing: bool = False,
        checkpointer: Optional[CheckpointManager] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        resume_from: Optional[int | str] = None,
        train_logger: Optional[TrainingLogger] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
    ) -> None:
        """
        Args:
            model:
                Trainable policy model (typically a HF CausalLM-like module).
                Must accept (input_ids, attention_mask) and expose .logits.
                May be wrapped in FSDP.

            algo:
                RLAlgorithm = (CreditAssigner + Loss).

            batch_source:
                Any object implementing BatchSource.next_batch() -> SAWBatch.
                This is where rollouts, replay, branching, curricula live.

            publisher:
                Abstract interface to push weights to inference workers.

            cfg:
                TrainerConfig for device, optimizer hyperparams, pad_token_id,
                grad_accum_steps, and sync_every_steps.

            param_filter:
                Optional predicate (name, Tensor) -> bool deciding which
                parameters get pushed into the runtime.

            enable_gradient_checkpointing:
                If True, enables activation checkpointing on the model to save VRAM.
                Also automatically disables KV cache and enables input gradients.

            checkpointer:
                Optional CheckpointManager for periodic HF-format saves. If not
                provided, one will be constructed from `checkpoint_config`.
                Prefer `checkpoint_config` for standard usage; `checkpointer`
                is an escape hatch to inject a custom implementation.

            checkpoint_config:
                Convenience config to build a default CheckpointManager.

            resume_from:
                Optional checkpoint to load on startup. Accepts a step number
                or an explicit checkpoint directory path. Requires a
                checkpointer (explicit or via checkpoint_config).

            train_logger:
                Optional stats logger (e.g., PrintLogger, WandbLogger). Called
                once per train_step with (step, stats).

            reducers:
                Optional aggregation reducers passed to aggregate_stats to
                compute custom metrics from SAWItem meta.
        """
        self.cfg = cfg
        self.train_logger = train_logger
        self.reducers = reducers

        # Assume caller has already done any FSDP wrapping / device placement.
        # We do NOT unconditionally .to(device) for FSDP; that’s the caller’s job.
        self.model = model.to(cfg.model_device) if not isinstance(model, FSDP) else model  # type: ignore[arg-type]

        self.algo = algo
        self.publisher = publisher
        self._batch_source = batch_source
        self.sync_every_steps = self.cfg.sync_every_steps
        self.param_filter = param_filter
        self._train_step_idx = 0
        self._checkpointer = checkpointer or (
            CheckpointManager(checkpoint_config) if checkpoint_config is not None else None
        )

        # ---- Gradient Checkpointing Setup ----------------------------
        if enable_gradient_checkpointing:
            logger.info("🛡️ Enabling Gradient Checkpointing (Activation Checkpointing)...")
            
            # 1. Enable on the model (HuggingFace standard API)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("⚠️ Model does not have 'gradient_checkpointing_enable' method.")

            # 2. Disable KV Cache (incompatible with checkpointing + training)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # 3. Enable input gradients
            # This is often required so that gradients can flow through the checkpointed segments.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # Fallback for generic torch modules if needed, though usually HF specific
                pass
        # --------------------------------------------------------------
        self.model.config.use_cache = False
        
        # Initialize optimizer
        self.optimizer = self.initialize_optimizer()

        # Assume gradients are zeroed at init
        self.optimizer.zero_grad(set_to_none=True)

        # Optionally resume from a checkpoint (weights + optimizer)
        if resume_from is not None:
            if self._checkpointer is None:
                raise ValueError("resume_from requires a CheckpointManager or checkpoint_config")
            load_kwargs: Dict[str, Optional[object]] = {}
            if isinstance(resume_from, int):
                load_kwargs["step"] = resume_from
            elif isinstance(resume_from, str):
                # Prefer treating a real path as explicit target; fall back to int-like strings
                if Path(resume_from).exists():
                    load_kwargs["path"] = resume_from
                elif resume_from.isdigit():
                    load_kwargs["step"] = int(resume_from)
            meta = self._checkpointer.load(
                self.model,
                optimizer=self.optimizer,
                step=load_kwargs.get("step"),  # type: ignore[arg-type]
                path=load_kwargs.get("path"),  # type: ignore[arg-type]
            )
            # Default to the saved step, else keep current.
            self._train_step_idx = int(meta.get("step", self._train_step_idx))
            logger.info("✅ Resumed from checkpoint at step %s", self._train_step_idx)

    # ------------------------------------------------------------------
    # Optimizer initialization
    # ------------------------------------------------------------------

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize and return the optimizer.

        - We create AdamW over all model parameters with hyperparams
          from TrainerConfig:

                lr         = cfg.lr
                weight_decay= cfg.weight_decay
                betas       = cfg.betas
                eps         = cfg.eps
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    # ------------------------------------------------------------------
    # Core async train step (now a "macro-step")
    # ------------------------------------------------------------------

    async def train_step(self) -> Dict[str, float]:
        """
        One full "macro" RL step with gradient accumulation:

            - Accumulate gradients over `cfg.grad_accum_steps` micro-batches
            - Perform one optimizer step
            - Free gradients
            - Optionally push updated params into runtime

        Returns:
            Aggregated stats dict from all micro-batches.
        """
        device = torch.device(self.cfg.model_device)
        grad_accum_steps = getattr(self.cfg, "grad_accum_steps", 1)
        if grad_accum_steps < 1:
            grad_accum_steps = 1

        all_micro_stats: List[Dict[str, float]] = []
        all_saw_batches: List[SAWBatch] = []

        # ---- 1) Accumulation Loop (Micro-Steps) ------------------------

        # Note: Gradients are *not* zeroed here. They are accumulated
        # from the previous state (which should be zero).
        self.model.train()

        log_memory = bool(getattr(self.cfg, "log_memory", False))
        log_memory_every_steps = int(getattr(self.cfg, "log_memory_every_steps", 1) or 1)
        log_memory_per_micro_step = bool(getattr(self.cfg, "log_memory_per_micro_step", False))
        should_log_step = log_memory and (self._train_step_idx % log_memory_every_steps == 0)

        if should_log_step:
            self._log_memory("train_step/start")

        for micro_step_idx in range(grad_accum_steps):
            
            # ---- 1a) Sample Valid Micro-batch (with Rejection Loop) ----
            # For PipelineRL, we might receive stale data from the queue.
            # We loop until we get a batch containing at least one fresh item.
            while True:
                saw_batch: SAWBatch = await self._batch_source.next_batch()
                
                # If configured, filter out items that exceed max_lag.
                if getattr(self.cfg, "max_lag", None) is not None:
                    current_time = self._train_step_idx
                    limit = self.cfg.max_lag
                    
                    fresh_items = []
                    for item in saw_batch.items:
                        # Default to current_time (0 lag) if tag is missing
                        item_ver = item.meta.get("policy_version", current_time)
                        if (current_time - item_ver) <= limit:
                            fresh_items.append(item)
                    
                    # Update the batch with only fresh items
                    saw_batch.items = fresh_items

                # If the batch is empty (e.g. all stale), drop it and fetch another.
                if not saw_batch.items:
                    continue

                # Batch has valid items, proceed to collation
                break

            all_saw_batches.append(saw_batch)

            # Debug: Check batch size before collation to diagnose OOM or filtering impact
            item_count = len(saw_batch.items)
            logger.info(
                f"[Micro-step {micro_step_idx+1}/{grad_accum_steps}] "
                f"Processing {item_count} SAWItems."
            )

            # ---- 1b) Collate into tensors ------------------------------
            max_seq_len = getattr(self.cfg, "max_seq_len", None)
            if isinstance(max_seq_len, int) and max_seq_len <= 0:
                max_seq_len = None

            batch_tensors = _collate_saw_items(
                saw_batch.items,
                pad_token_id=self.cfg.pad_token_id,
                device=device,
                max_seq_len=max_seq_len,
            )

            # Debug: Check tensor shape [Batch, Time]
            input_shape = batch_tensors["input_ids"].shape
            logger.info(
                f"    -> Collated Tensor Shape: {input_shape} "
                f"(Batch={input_shape[0]}, SeqLen={input_shape[1]})"
            )

            # ---- 1c) FSDP: context for no_sync -------------------------
            # We only sync (all-reduce) gradients on the *last* micro-batch
            is_last_micro = (micro_step_idx == grad_accum_steps - 1)
            no_sync_context = (
                self.model.no_sync()
                if (isinstance(self.model, FSDP) and not is_last_micro)
                else contextlib.nullcontext()
            )

            # ---- 1d) Loss + backward (scaled) --------------------------
            if should_log_step and log_memory_per_micro_step:
                self._log_memory(f"train_step/micro{micro_step_idx+1}/pre_fwd")

            try:
                if should_log_step and log_memory_per_micro_step and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                with no_sync_context:
                    loss, stats = self.algo.compute_loss(self.model, batch_tensors)

                    if should_log_step and log_memory_per_micro_step:
                        self._log_memory(f"train_step/micro{micro_step_idx+1}/post_fwd")

                    # Scale loss for accumulation
                    scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

                if should_log_step and log_memory_per_micro_step:
                    self._log_memory(f"train_step/micro{micro_step_idx+1}/post_bwd")

            except torch.cuda.OutOfMemoryError:
                if bool(getattr(self.cfg, "log_memory_summary_on_oom", True)) and torch.cuda.is_available():
                    logger.error("CUDA OOM in train_step at step=%s micro=%s", self._train_step_idx, micro_step_idx)
                    logger.error(self._cuda_memory_summary())
                raise

            all_micro_stats.append(stats)

        # ---- 2) Gradient Clipping (after loop) -------------------------
        if self.cfg.max_grad_norm is not None:
            # FSDP requires calling clip_grad_norm_ *on the model*
            # to handle unsharding grads before clipping.
            if isinstance(self.model, FSDP):
                self.model.clip_grad_norm_(self.cfg.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.max_grad_norm,
                )

        # ---- 3) Optimizer Step (one step for the macro-batch) ----------
        self.optimizer.step()

        # ---- 4) Free Gradients (as requested) --------------------------
        # Grads are freed *after* step and *before* weight sync
        self.optimizer.zero_grad(set_to_none=True)

        self._train_step_idx += 1

        # ---- 5) Push policy update into runtime ------------------------
        if self._train_step_idx % self.sync_every_steps == 0:
            if should_log_step:
                self._log_memory("train_step/pre_publish")
            self._push_weights_to_runtime()
            if should_log_step:
                self._log_memory("train_step/post_publish")

        # ---- 6) Enrich stats -------------------------------------------
        final_stats = aggregate_stats(all_micro_stats, all_saw_batches, reducers=self.reducers)
        final_stats["train_step"] = float(self._train_step_idx)

        # ---- 7) Optional Checkpoint ------------------------------------
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(
                self.model,
                optimizer=self.optimizer,
                step=self._train_step_idx,
                metadata={"algorithm": self.algo.name},
            )

        # ---- 8) Optional logging ---------------------------------------
        if self.train_logger is not None:
            try:
                self.train_logger.log(self._train_step_idx, final_stats)
            except Exception:
                logger.exception("Stats logger failed at step %s", self._train_step_idx)

        if should_log_step:
            self._log_memory("train_step/end")

        return final_stats

    # ------------------------------------------------------------------
    # Memory logging helpers
    # ------------------------------------------------------------------

    def _log_memory(self, tag: str) -> None:
        parts = [f"[mem] {tag}"]
        rss = self._process_rss_bytes()
        if rss is not None:
            parts.append(f"rss={rss/1024/1024:.0f}MiB")
        if torch.cuda.is_available():
            try:
                parts.append(self._cuda_memory_brief())
            except Exception:
                logger.exception("Failed to collect CUDA memory stats")
        logger.info(" ".join(parts))

    def _process_rss_bytes(self) -> Optional[int]:
        # Linux-only (this repo runs primarily in Linux containers).
        status_path = "/proc/self/status"
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # e.g. "VmRSS:\t  123456 kB"
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024
                        return None
        except OSError:
            return None
        return None

    def _cuda_memory_brief(self) -> str:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        return (
            f"cuda_alloc={allocated/1024/1024:.0f}MiB "
            f"cuda_reserved={reserved/1024/1024:.0f}MiB "
            f"cuda_peak_alloc={max_alloc/1024/1024:.0f}MiB "
            f"cuda_peak_reserved={max_reserved/1024/1024:.0f}MiB"
        )

    def _cuda_memory_summary(self) -> str:
        # Use abbreviated=True to keep logs readable.
        return torch.cuda.memory_summary(abbreviated=True)

    # ------------------------------------------------------------------
    # Sync wrappers for non-async callers
    # ------------------------------------------------------------------

    def train_step_sync(self) -> Dict[str, float]:
        """
        Synchronous wrapper around `train_step()`.

        Use this from plain scripts/CLIs. Do NOT call inside an existing
        running event loop.
        """
        return asyncio.run(self.train_step())

    async def train(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Run `num_steps` training iterations.

        Args:
            num_steps:
                Number of trainer steps (each step pulls a fresh SAWBatch).

            log_every:
                Frequency (in steps) at which to call `log_fn`.

            log_fn:
                Optional callback(stats_dict) for logging / progress reporting.
        """
        for _ in range(num_steps):
            stats = await self.train_step()
            if log_fn is not None and (self._train_step_idx % log_every == 0):
                log_fn(stats)

    def train_sync(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Synchronous wrapper around `train(...)`.
        """
        asyncio.run(self.train(num_steps, log_every=log_every, log_fn=log_fn))

    # ------------------------------------------------------------------
    # Weight sync into runtime via Agent (FSDP-aware + LoRA-aware)
    # ------------------------------------------------------------------

    def _push_weights_to_runtime(self) -> None:
        """
        Gather weights (handling FSDP if needed) and publish them.

        If the model is a PEFT/LoRA model, we strictly follow the
        Merge -> Publish -> Unmerge pattern so vLLM receives dense weights
        but training continues on adapters.
        """
        # Helper to get the underlying model if wrapped in FSDP
        # FSDP wraps the actual model in .module
        inner_model = self.model.module if isinstance(self.model, FSDP) else self.model

        # 1. Check if this is a LoRA/PEFT model
        #    (We use getattr so we don't need to import peft here)
        merge_fn = getattr(inner_model, "merge_adapter", None)
        unmerge_fn = getattr(inner_model, "unmerge_adapter", None)
        is_peft = callable(merge_fn) and callable(unmerge_fn)

        # 2. If LoRA, merge weights before gathering
        if is_peft:
            merge_fn()

        try:
            # Only rank 0 talks to the runtime in distributed mode
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() != 0:
                    return

            runtime_device = torch.device(
                self.cfg.runtime_device or self.cfg.model_device
            )

            # We use the current training step as the authoritative version for PipelineRL
            current_version = self._train_step_idx

            raw_params: Dict[str, Tensor] = {}

            # --- Gather Raw Params (FSDP or Standard) ---
            if isinstance(self.model, FSDP):
                # Gather full, unsharded state dict on the model device.
                full_cfg = FullStateDictConfig(
                    offload_to_cpu=False,  # full model stays on GPU; rollouts dominate anyway
                    rank0_only=True,
                )
                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
                    full_cfg,
                ):
                    full_state = self.model.state_dict()
                    for k, v in full_state.items():
                        if self.param_filter is not None and not self.param_filter(k, v):
                            continue
                        raw_params[k] = v.detach().to(runtime_device)
            else:
                # Standard model
                for name, p in self.model.named_parameters():
                    # Optimization: In LoRA, only send what we touched (plus what needs fusion)
                    # If not PEFT, only send requires_grad.
                    # If PEFT (merged), we theoretically need to send the whole base layer 
                    # because it changed.
                    
                    # If standard training: send only requires_grad=True
                    # If LoRA (merged): send everything (because base weights need updating in vLLM)
                    if not is_peft and not p.requires_grad:
                        continue
                        
                    if self.param_filter is not None:
                        if not self.param_filter(name, p):
                            continue

                    raw_params[name] = p.detach().to(runtime_device)

            if not raw_params:
                return

            # --- Publish ---
            # NOTE: We simply hand off the raw state dict and rely on the Publisher to handle fusion and renaming.
            self.publisher.publish(raw_params, version=current_version)

        finally:
            # 3. CRITICAL: Unmerge adapters immediately after publishing.
            #    If we don't do this, the optimizer state will become invalid
            #    (gradients calculated on merged weights vs adapter weights).
            if is_peft:
                unmerge_fn()
