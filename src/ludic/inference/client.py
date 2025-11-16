from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import torch  # type: ignore

from ludic.types import Message
from ludic.inference.sampling import SamplingConfig

@dataclass
class ChatResponse:
    """
    Normalized inference output for training/logging.
    Keep this minimal. Put transport/vendor junk in the returned `info` dict.
    """
    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None

class ChatClient(Protocol):
    """
    Backend contract.
      - accepts a fully-resolved SamplingConfig (defaults already applied)
      - maps SamplingConfig -> backend kwargs
      - executes the call and returns (ChatResponse, info)
      - can atomically push a set of parameter tensors to the runtime
    """

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        ...

    def push_update_atomic(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
    ) -> str:
        """
        Atomically apply a set of parameter updates.
        Returns the committed version string.
        Should raise specific exceptions on timeout/reject/broadcast failure.
        """
        ...
