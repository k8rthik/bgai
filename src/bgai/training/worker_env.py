"""Thread pinning for CPU-bound worker processes.

``torch.set_num_threads(1)`` inside a worker is not enough: on Linux
aarch64 torch's OpenMP pool was already spun up by the time it ran, and
each self-play worker showed five runnable threads with the game thread
getting about a third of a core. The BLAS/OpenMP thread counts are read
from the environment at library import, so the PARENT must export them
before spawning; ``spawn``ed children inherit ``os.environ`` and import
torch single-threaded from the start.

``setdefault`` only: an operator who deliberately exported a different
count keeps it.
"""

from __future__ import annotations

import os

SINGLE_THREAD_ENV: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def pin_single_thread_env() -> dict[str, str]:
    """Export the single-thread variables (without overriding explicit
    values) and return the resulting mapping for logging."""
    for key in SINGLE_THREAD_ENV:
        os.environ.setdefault(key, "1")
    return {key: os.environ[key] for key in SINGLE_THREAD_ENV}
