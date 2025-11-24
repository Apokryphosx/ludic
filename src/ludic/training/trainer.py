from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional

import torch
from torch import nn, optim, Tensor
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

from ludic.agent import Agent
from ludic.training.algorithm import RLAlgorithm
from ludic.training.config import TrainerConfig
from ludic.training.types import SAWBatch, SAWItem, BatchSource


# ---------------------------------------------------------------------------
# Collation: SAWItems -> tensor batch
# ---------------------------------------------------------------------------


def _collate_saw_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
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

    lengths = [len(it.input_ids) for it in items]
    max_len = max(lengths)

    input_ids_list: List[Tensor] = []
    attn_mask_list: List[Tensor] = []
    action_mask_list: List[Tensor] = []
    weights_list: List[Tensor] = []

    for it in items:
        L = len(it.input_ids)
        pad_len = max_len - L

        ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
        am = torch.zeros((max_len,), dtype=torch.long)
        actm = torch.zeros((max_len,), dtype=torch.float32)

        ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
        am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
        actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)

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
    Orchestrates the training loop:

        BatchSource.next_batch() → SAWBatch
          ↓
        collate SAWBatch.items → tensors
          ↓
        RLAlgorithm.compute_loss(model, batch)
          ↓
        optimizer.step()
          ↓
        Agent.push_policy_update(...)   # optional, online weight sync

    Trainer is agnostic to envs, contexts, rollouts, and tokenization.

    This variant is FSDP-aware:

      - If `model` is wrapped in FSDP/FSDP2, it will:
            * on rank 0 only:
                - switch to FULL_STATE_DICT
                - gather a full state dict (no CPU offload by default)
                - push the full (unsharded) params to the Agent's runtime

      - On non-FSDP models, it just uses `named_parameters()` as before.

    By default, we sync to the runtime **every train step** (`sync_to_runtime=True`),
    i.e. strictly on-policy if you use the runtime for rollouts.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        agent: Agent,
        cfg: TrainerConfig = TrainerConfig(),
        sync_every_steps: int = 1,
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
    ) -> None:
        """
        Args:
            model:
                Trainable policy model (typically a HF CausalLM-like module).
                Must accept (input_ids, attention_mask) and expose .logits.
                May be wrapped in FSDP/FSDP2.

            algo:
                RLAlgorithm = (WeightingStrategy + Loss).

            batch_source:
                Any object implementing BatchSource.next_batch() -> SAWBatch.
                This is where rollouts, replay, branching, curricula live.

            agent:
                Ludic Agent wrapping a ChatClient (e.g., VLLMChatClient).
                Used for pushing updated weights into the runtime.

            cfg:
                TrainerConfig for device, optimizer hyperparams, pad_token_id, etc.

            sync_every_steps:
                Push the updated weights every n steps to the inference engine.

            param_filter:
                Optional predicate (name, Tensor) -> bool deciding which
                parameters get pushed into the runtime. If None, defaults to
                `p.requires_grad` for non-FSDP models; for FSDP, no default
                filter (caller should pass one if they want to restrict).
        """
        self.cfg = cfg

        # Assume caller has already done any FSDP wrapping / device placement.
        # We do NOT unconditionally .to(device) for FSDP; that’s the caller’s job.
        self.model = model.to(cfg.model_device) if not isinstance(model, FSDP) else model  # type: ignore[arg-type]

        self.algo = algo
        self.agent = agent
        self._batch_source = batch_source
        self.sync_every_steps = sync_every_steps
        self.param_filter = param_filter
        self._train_step_idx = 0

        # Initialize optimizer
        self.optimizer = self.initialize_optimizer()

    # ------------------------------------------------------------------
    # Optimizer initialization
    # ------------------------------------------------------------------

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize and return the optimizer.

        - We create AdamW over all model parameters with hyperparams
          from TrainerConfig:

              lr          = cfg.lr
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
    # Core async train step
    # ------------------------------------------------------------------

    async def train_step(self) -> Dict[str, float]:
        """
        One full RL step:

            - sample SAWBatch from BatchSource
            - collate to tensors
            - compute loss via algo.compute_loss
            - backward + optimizer step
            - optionally push updated params into runtime

        Returns:
            stats dict with at least:
                - "loss"
                - "train_step"
                - "batch_items"
                - "batch_size"
                - "avg_total_reward"
            plus whatever the Loss implementation adds.
        """
        device = torch.device(self.cfg.model_device)

        # ---- 1) Sample batch from source -------------------------------
        saw_batch: SAWBatch = await self._batch_source.next_batch()

        # ---- 2) Collate into tensors -----------------------------------
        batch_tensors = _collate_saw_items(
            saw_batch.items,
            pad_token_id=self.cfg.pad_token_id,
            device=device,
        )

        # ---- 3) Loss + backward ----------------------------------------
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, stats = self.algo.compute_loss(self.model, batch_tensors)
        loss.backward()

        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_grad_norm,
            )

        self.optimizer.step()

        self._train_step_idx += 1

        # ---- 4) Push policy update into runtime ------------------------
        if self._train_step_idx % self.sync_every_steps == 0:
            self._push_weights_to_runtime()

        # ---- 5) Enrich stats -------------------------------------------
        stats = dict(stats)
        stats["train_step"] = float(self._train_step_idx)
        stats["batch_items"] = float(len(saw_batch.items))
        stats["batch_size"] = float(saw_batch.meta.get("batch_size", 0.0))
        stats["avg_total_reward"] = float(saw_batch.meta.get("avg_total_reward", 0.0))

        return stats

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
    # Weight sync into runtime via Agent (FSDP-aware)
    # ------------------------------------------------------------------

    def _push_weights_to_runtime(self) -> None:
        """
        Push updated policy parameters into the serving runtime via Agent.

        Requires that `agent` implements:

            def push_policy_update(self, params: Mapping[str, Tensor], ...) -> str

        FSDP/FSDP2-aware behavior:

            - If model is FSDP:
                * only rank 0 (if dist initialized) does anything
                * uses FULL_STATE_DICT with rank0_only=True
                * gathers a full, unsharded state_dict on model_device
                * optionally filters params, then hands the dict to Agent

            - If model is not FSDP:
                * uses named_parameters() with the same filter policy.

        Note: this assumes that the runtime (e.g. vLLM client) expects tensors
        on the device implied by cfg.runtime_device (or cfg.model_device if None),
        and will use NCCL from there. If you care, set cfg.runtime_device
        accordingly.
        """
        if not hasattr(self.agent, "push_policy_update"):
            raise RuntimeError(
                "Agent does not support policy weight updates "
                "(missing push_policy_update)."
            )

        # Only rank 0 talks to the runtime in distributed mode
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return

        runtime_device = torch.device(
            self.cfg.runtime_device or self.cfg.model_device
        )

        # ---------------- FSDP path ----------------
        if isinstance(self.model, FSDP):
            # Gather full, unsharded state dict on the model device.
            full_cfg = FullStateDictConfig(
                offload_to_cpu=False,   # full model stays on GPU; rollouts dominate anyway
                rank0_only=True,
            )
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                full_cfg,
            ):
                full_state = self.model.state_dict()

            params: Dict[str, Tensor] = {}
            for name, tensor in full_state.items():
                if self.param_filter is not None and not self.param_filter(name, tensor):
                    continue
                params[name] = tensor.detach().to(runtime_device)

            if not params:
                return

            self.agent.push_policy_update(params)
            return

        # ---------------- non-FSDP path ----------------
        params: Dict[str, Tensor] = {}
        for name, p in self.model.named_parameters():
            if self.param_filter is not None:
                if not self.param_filter(name, p):
                    continue
            else:
                if not p.requires_grad:
                    continue

            params[name] = p.detach().to(runtime_device)

        if not params:
            return

        # Delegate to Agent → ChatClient → vLLM
        self.agent.push_policy_update(params)
