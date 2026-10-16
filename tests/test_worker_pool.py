"""Pools of SIGTERM-ignoring workers must shut down without hanging.

``Pool.__exit__`` calls ``terminate()``, which SIGTERMs each worker and
joins it. Self-play workers ignore SIGTERM by design, so an idle worker
that missed the pool's shutdown sentinel was joined forever -- the
actor-restart test deadlocked at 0% CPU."""

from __future__ import annotations

import threading
from multiprocessing import get_context

from bgai.training.selfplay_actor import ignore_stop_signals
from bgai.training.worker_pool import shielded_pool

DEADLINE_SECS = 60.0


def _square(x: int) -> int:
    return x * x


def _within_deadline(fn) -> dict:
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- surfaced to the test
            outcome["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(DEADLINE_SECS)
    assert not t.is_alive(), "pool shutdown hung"
    return outcome


def _pool_kwargs() -> dict:
    return {"processes": 2, "initializer": ignore_stop_signals}


def test_shielded_pool_exits_cleanly_after_work() -> None:
    def work() -> list[int]:
        for _ in range(5):  # the race is per shutdown; repeat it
            with shielded_pool(get_context("spawn"), **_pool_kwargs()) as pool:
                out = pool.map(_square, range(4))
        return out

    assert _within_deadline(work) == {"value": [0, 1, 4, 9]}


def test_shielded_pool_kills_workers_and_reraises_on_error() -> None:
    seen: list = []

    def work() -> None:
        with shielded_pool(get_context("spawn"), **_pool_kwargs()) as pool:
            pool.map(_square, range(2))
            seen.extend(pool._pool)  # noqa: SLF001
            raise RuntimeError("boom")

    outcome = _within_deadline(work)
    assert isinstance(outcome.get("error"), RuntimeError)
    assert seen and all(not p.is_alive() for p in seen)

