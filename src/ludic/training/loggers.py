from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence


class TrainingLogger(Protocol):
    """
    Interface for logging training stats to arbitrary backends.
    """

    def log(self, step: int, stats: Dict[str, float]) -> None:
        ...


class PrintLogger:
    """
    Lightweight console logger (plain stdout).
    """

    def __init__(
        self,
        *,
        prefix: str = "[trainer]",
        keys: Sequence[str] | None = None,
        precision: int = 4,
        max_items_per_line: int = 6,
    ) -> None:
        self.prefix = prefix
        self.keys = list(keys) if keys is not None else None
        self.precision = precision
        self.max_items_per_line = max_items_per_line

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if self.keys is not None:
            keys = [k for k in self.keys if k in stats]
        else:
            keys = sorted(stats.keys())
        pairs = [f"{k}={self._fmt_val(stats[k])}" for k in keys]
        if not pairs:
            print(f"{self.prefix} step={step}")
            return

        header = f"{self.prefix} step={step}"
        lines = []
        for i in range(0, len(pairs), self.max_items_per_line):
            lines.append(" ".join(pairs[i : i + self.max_items_per_line]))

        print(f"{header} {lines[0]}")
        indent = " " * len(header)
        for line in lines[1:]:
            print(f"{indent} {line}")


class WandbLogger:
    """
    Minimal Weights & Biases logger. Lazily imports wandb.
    """

    def __init__(self, *, run: Any | None = None, init_kwargs: Dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise ImportError("WandbLogger requires the 'wandb' package installed.") from exc

        self._wandb = wandb
        self._run = run or wandb.init(**(init_kwargs or {}))

    def log(self, step: int, stats: Dict[str, float]) -> None:
        self._wandb.log(stats, step=step)
