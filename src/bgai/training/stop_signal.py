"""Graceful stop for long-running training processes.

Two triggers, one flag: a ``STOP`` file in the run directory (anyone
can create it from any host; rsync propagates it) or SIGTERM/SIGINT to
the process (a tmux kill, a scheduler pre-emption). Loops check
``requested()`` at iteration/step boundaries only, so the worst case is
finishing the in-flight iteration -- never a half-written checkpoint.

``clear()`` removes the file once the process has honoured it, so a
later restart in the same directory does not stop immediately.
"""

from __future__ import annotations

import contextlib
import logging
import signal
from pathlib import Path

log = logging.getLogger(__name__)

STOP_FILE = "STOP"
_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopSignal:
    """Not a dataclass: the flag is process state a signal handler mutates."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._flag = False
        self._previous: dict[int, object] = {}

    @property
    def stop_file(self) -> Path:
        return self.run_dir / STOP_FILE

    def _handler(self, signum: int, _frame) -> None:
        log.warning("stop: received signal %d; finishing the in-flight step", signum)
        self._flag = True

    def install(self) -> None:
        """Route SIGTERM/SIGINT to the flag (main thread only; elsewhere
        the file check still works)."""
        for sig in _SIGNALS:
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except ValueError:  # not the main thread
                return

    def uninstall(self) -> None:
        for sig, previous in self._previous.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, previous)  # type: ignore[arg-type]
        self._previous = {}

    def requested(self) -> bool:
        return self._flag or self.stop_file.exists()

    def clear(self) -> None:
        self.stop_file.unlink(missing_ok=True)
        self._flag = False
