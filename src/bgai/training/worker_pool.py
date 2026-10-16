"""Process pools whose workers ignore SIGTERM (self-play game workers).

Those workers ignore SIGINT/SIGTERM so a process-group signal cannot kill
a game mid-flight; the parent decides when to stop. That makes the stock
``with Pool(...)`` exit unsafe: ``Pool.__exit__`` calls ``terminate()``,
which SIGTERMs every worker and then joins it, and an idle worker that
missed the pool's shutdown sentinel ignores the signal and is joined
forever (0% CPU, parent stuck in ``Process.join``).

``shielded_pool`` exits cleanly instead: on success ``close()`` +
``join()`` lets every worker take its sentinel and return; on an error it
runs ``terminate()`` with SIGKILL -- the one signal they cannot ignore --
in place of SIGTERM, then re-raises."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from multiprocessing.pool import Pool
from typing import Any


@contextmanager
def shielded_pool(ctx: BaseContext, **pool_kwargs: Any) -> Iterator[Pool]:
    pool = ctx.Pool(**pool_kwargs)
    try:
        yield pool
    except BaseException:
        _terminate_with_kill(pool, ctx)
        raise
    pool.close()
    pool.join()


def _terminate_with_kill(pool: Pool, ctx: BaseContext) -> None:
    """``Pool.terminate()`` with SIGKILL where it would send SIGTERM.

    Killing workers *before* terminate() races the pool: its handler thread
    respawns replacements (which ignore SIGTERM too), and a worker killed
    while idle can die holding the task queue's read lock that terminate()
    then waits on. terminate() itself stops respawning and drains the queue
    before it signals workers, so only the signal is swapped."""
    proc_cls = ctx.Process
    original = proc_cls.terminate
    proc_cls.terminate = proc_cls.kill  # type: ignore[method-assign]
    try:
        pool.terminate()
    finally:
        proc_cls.terminate = original  # type: ignore[method-assign]
