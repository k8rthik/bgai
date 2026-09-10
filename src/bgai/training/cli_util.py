"""argparse helpers for resumable CLIs.

``--resume RUN_DIR`` takes every setting from the checkpoint's stored
config and lets the command line override only what was EXPLICITLY
typed. argparse cannot tell a default from an explicit value, so the
parser is rebuilt with every default suppressed and parsed again: what
survives in the namespace is exactly what the user passed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any


def explicit_cli_args(
    build_parser: Callable[[], argparse.ArgumentParser], argv: list[str] | None
) -> dict[str, Any]:
    """``{dest: value}`` for the flags actually present in ``argv``."""
    parser = build_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    return vars(parser.parse_args(argv))


def config_kwargs_from_stored(config_type: type, stored: dict, path_fields: set[str]) -> dict:
    """Pick the dataclass's fields out of a checkpoint's stored config,
    turning the string-serialised paths back into ``Path``s."""
    known = {f.name for f in fields(config_type)}
    out: dict[str, Any] = {}
    for key, value in stored.items():
        if key not in known:
            continue
        if key in path_fields:
            out[key] = Path(value) if value is not None else None
        else:
            out[key] = value
    return out
