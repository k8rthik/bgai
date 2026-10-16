"""Graceful stop for long-running training processes.

Two triggers, one latched flag: a ``STOP`` file in the run directory
(anyone can create it from any host; rsync propagates it) or
SIGTERM/SIGINT to the process (a tmux kill, a scheduler pre-emption).
Loops check ``requested()`` at iteration/step/game boundaries only, so
the worst case is finishing the in-flight unit -- never a half-written
checkpoint or shard.

Ownership rules (they matter with several processes on one directory):

- ``requested()`` LATCHES: once a stop has been seen it stays requested
  even if the file disappears, so an actor that noticed ``STOP`` at one
  game's end cannot be "un-stopped" by a cleanup racing ahead of it.
- A ``STOP`` whose mtime predates this process's start is STALE -- left
  over from a previous run that was stopped and never cleaned up -- and
  is removed and ignored on first sight. Hosts sharing a mirrored run
  directory therefore need roughly synchronised clocks (NTP), the same
  requirement rsync's mtime-based sync has anyway.
- Who deletes a live ``STOP``: the process that owns the WHOLE run. The
  serial loop (``clear()``), or the pipelined launcher after every actor
  has exited. A standalone learner or actor never deletes one it
  honoured, because the other processes may not have seen it yet.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import time
from pathlib import Path

log = logging.getLogger(__name__)

STOP_FILE = "STOP"
_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopSignal:
    """Not a dataclass: the flag is process state a signal handler mutates."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.started_at = time.time()
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

    def _file_requests_stop(self) -> bool:
        try:
            mtime = os.stat(self.stop_file).st_mtime
        except FileNotFoundError:
            return False
        if mtime < self.started_at:
            log.warning("stop: removing stale %s (older than this process)", self.stop_file)
            self.stop_file.unlink(missing_ok=True)
            return False
        return True

    def requested(self) -> bool:
        if not self._flag and self._file_requests_stop():
            self._flag = True
        return self._flag

    def clear(self) -> None:
        """Remove the STOP file and reset the latch -- run owners only."""
        self.stop_file.unlink(missing_ok=True)
        self._flag = False
