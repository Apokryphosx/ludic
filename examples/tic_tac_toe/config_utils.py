from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


def load_toml(path: str | Path) -> Dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        raise RuntimeError("TOML support requires Python 3.11+ (tomllib).")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid TOML root type: {type(data)}")
    return data


def select_section(config: Mapping[str, Any], section: str) -> Dict[str, Any]:
    """
    Merge top-level scalar keys with an optional named section.

    - Scalars at the TOML root apply to both scripts.
    - Tables under [train] or [eval] override those scalars.
    """
    base: Dict[str, Any] = {k: v for k, v in config.items() if not isinstance(v, dict)}
    sec = config.get(section)
    if isinstance(sec, dict):
        base.update(sec)
    return base


def cli_provided(argv: Sequence[str], option_strings: Iterable[str]) -> bool:
    opts = set(option_strings)
    for tok in argv:
        if tok in opts:
            return True
    return False


def apply_config_to_args(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    argv: Optional[Sequence[str]] = None,
    option_strings_by_dest: Mapping[str, Sequence[str]],
) -> argparse.Namespace:
    """
    For each dest in option_strings_by_dest:
      - if config contains dest
      - and CLI did NOT provide any of its option strings
    then set args.dest = config[dest].
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for dest, option_strings in option_strings_by_dest.items():
        if dest not in config:
            continue
        if cli_provided(argv, option_strings):
            continue
        setattr(args, dest, config[dest])
    return args

