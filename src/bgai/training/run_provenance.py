"""Code provenance stamped on every checkpoint config and metrics row.

A self-play run outlives code versions: actors get upgraded one at a
time, the learner last, and a games/hour change has to be attributable
to the upgrade rather than to the cluster. So every row carries the
short git sha of the code that wrote it, whether the tree was dirty,
the state-encoding version and the host. On resume a differing
``encoding_version`` is refused (the shards would not decode) and a
differing ``git_sha`` emits a ``code_changed`` event row.

Git is queried for the package's OWN repository (not the cwd), with
short timeouts; anything that fails degrades to ``"unknown"`` rather
than blocking a training run.
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from bgai.training.vocab import ENCODING_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_TIMEOUT = 5.0


@dataclass(frozen=True)
class CodeProvenance:
    git_sha: str
    code_dirty: bool
    encoding_version: int
    hostname: str

    def as_dict(self) -> dict:
        return asdict(self)


def _git(args: list[str], root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git_sha(root: Path) -> str:
    out = _git(["rev-parse", "--short", "HEAD"], root)
    return out.strip() if out else "unknown"


def _git_dirty(root: Path) -> bool:
    out = _git(["status", "--porcelain"], root)
    return bool(out and out.strip())


def code_provenance(root: Path = _REPO_ROOT) -> CodeProvenance:
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unknown"
    return CodeProvenance(
        git_sha=_git_sha(root),
        code_dirty=_git_dirty(root),
        encoding_version=ENCODING_VERSION,
        hostname=hostname,
    )
