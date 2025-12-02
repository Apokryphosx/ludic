from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from ludic.parsers import Parser
from ludic.types import JSON, Rollout, SamplingArgs, Step


@dataclass
class EnvSpec:
    """
    Serializable description of an environment to instantiate.

    - kind: string key into an env registry
    - kwargs: JSON-serializable constructor/config kwargs
    """
    kind: str
    kwargs: Dict[str, JSON] = field(default_factory=dict)


@dataclass
class CtxSpec:
    """
    Serializable description of a context strategy.

    - kind: string key into a ctx registry
    - kwargs: JSON-serializable constructor/config kwargs
    """
    kind: str
    kwargs: Dict[str, JSON] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rollout-level configuration / identification
# ---------------------------------------------------------------------------

# (rollout_id, step_index)
RolloutStepKey = Tuple[str, int]

@dataclass
class RolloutRequest:
    """
    Template for one or more rollouts.

    This is *pure data*; RolloutEngine will:

        - resolve env via registry from (env.kind)
        - call the factory with env.kwargs
        - run `num_episodes` independent episodes using the
          engine's pre-configured InteractionProtocol.

    Fields:
      - env:
            EnvSpec, resolved via env_registry.
            
      - sampling_args:
            Passed directly to Agent via protocol.run().

      - num_episodes:
            How many episodes to run with this configuration.

      - meta:
            Arbitrary JSON metadata that gets merged into Rollout.meta["request_meta"].
    """
    env: EnvSpec
    seed: Optional[int] = None
    sampling_args: Optional[SamplingArgs] = None
    num_episodes: int = 1
    meta: Dict[str, JSON] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------


class CreditAssigner(Protocol):
    """
    Computes a scalar weight for each (rollout, step) in a batch.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        ...


# ---------------------------------------------------------------------------
# State–Action–Weight representation
# ---------------------------------------------------------------------------


@dataclass
class SAWItem:
    """
    State–Action–Weight sample with masks.

    - input_ids: tokenized [state || action]
    - attention_mask: 1/0 attention mask to tell tokens from padding
    - action_mask: 1 on action tokens, 0 on state tokens
    - weight: scalar credit for this sample
    - meta: arbitrary rollout/step metadata
    """
    input_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]
    weight: float
    meta: Dict[str, JSON]

@dataclass
class SAWBatch:
    """
    Logical batch of State–Action–Weight samples.

    - items: the SAWItem samples
    - meta: batch-level metadata (reward stats, timing, env info, etc.)
    """
    items: list[SAWItem]
    meta: dict[str, JSON] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Batch source abstraction
# ---------------------------------------------------------------------------


class BatchSource(Protocol):
    """
    Abstract source of SAWBatch samples.

    Trainer only depends on this interface and does not care where the
    data comes from (online rollouts, replay buffer, branching search, etc.).
    """

    async def next_batch(self) -> SAWBatch:
        ...

# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------

TokenizeFn = Callable[[str], List[int]]
StateFromStepFn = Callable[[Rollout, int, Step], str]