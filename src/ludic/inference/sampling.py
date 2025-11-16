from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ludic.types import SamplingArgs


@dataclass(frozen=True)
class SamplingConfig:
    """
    Fully resolved sampling configuration.

    This is the "post-defaulting" object: every field has a concrete value.
    No missing keys, no guessing at call sites.
    """
    seed: int
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, List[str]]  # can be "" / [] / actual stops, but never missing
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """
        Map this config to OpenAI-compatible keyword arguments.
        Vendor-specific `extras` are layered on top and can override fields.
        """
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
        }

        # `stop` being "falsy" means "don't send a stop field at all"
        if self.stop:
            kwargs["stop"] = self.stop

        # extras can override anything if the caller really wants to
        kwargs.update(self.extras or {})

        return kwargs


# Project-wide default sampling config.
# Central place for policy instead of random literals in clients.
_DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    seed=0,
    temperature=0.7,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=[],
    extras={},
)


def get_default_sampling_config() -> SamplingConfig:
    """
    Return the default sampling configuration.

    The returned object is immutable (frozen dataclass), so callers who want
    to tweak it should create a new SamplingConfig instance rather than
    mutating this one.
    """
    return _DEFAULT_SAMPLING_CONFIG


def resolve_sampling_args(
    partial: Optional[SamplingArgs],
    base: SamplingConfig = _DEFAULT_SAMPLING_CONFIG,
) -> SamplingConfig:
    """
    Take a (possibly partial) SamplingArgs dict and resolve it into a fully
    specified SamplingConfig by overlaying it on top of `base`.

    - If `partial` is None, you just get `base`.
    - For `extras`, we shallow-merge base.extras and partial["extras"].
    """

    if partial is None:
        return base

    extras = partial.get("extras")
    merged_extras = {
        **(base.extras or {}),
        **(extras or {}),
    }

    return SamplingConfig(
        seed=partial.get("seed", base.seed),
        temperature=partial.get("temperature", base.temperature),
        max_tokens=partial.get("max_tokens", base.max_tokens),
        top_p=partial.get("top_p", base.top_p),
        frequency_penalty=partial.get("frequency_penalty", base.frequency_penalty),
        presence_penalty=partial.get("presence_penalty", base.presence_penalty),
        stop=partial.get("stop", base.stop),
        extras=merged_extras,
    )
