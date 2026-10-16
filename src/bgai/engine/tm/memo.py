"""Identity-keyed memoisation for pure queries over immutable engine objects.

``GameState``/``FactionState`` are frozen dataclasses whose fields
include plain dicts, so they are not hashable and cannot key an
``lru_cache``. They are, however, never mutated after construction --
every engine step builds a new object -- so a query's answer is fixed
for the lifetime of the object it was asked about. That makes object
identity a sound cache key, provided the id cannot be recycled while
the entry lives: each entry pins a reference to its keyed object, and
the table is bounded so pinning stays cheap (a few hundred states).

Search is the consumer: MCTS asks ``reachable``/``clusters``/... several
times per node about the very same state object (once per legal-move
family, again on the child's expansion), and every one of those calls
is pure geometry over a state that cannot have changed.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

_R = TypeVar("_R")

DEFAULT_MAXSIZE = 512


def state_memo(maxsize: int = DEFAULT_MAXSIZE) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """Memoise ``fn(obj, *args)`` on ``(id(obj), *args)``.

    ``args`` must be hashable (faction names, flags). Eviction is FIFO:
    search touches states in roughly generation order, so the oldest
    entry is the one least likely to be asked about again.
    """

    def decorate(fn: Callable[..., _R]) -> Callable[..., _R]:
        table: dict[tuple[Any, ...], tuple[Any, _R]] = {}

        @wraps(fn)
        def wrapper(obj: Any, *args: Any) -> _R:
            key = (id(obj), *args)
            hit = table.get(key)
            if hit is not None and hit[0] is obj:
                return hit[1]
            value = fn(obj, *args)
            if len(table) >= maxsize:
                del table[next(iter(table))]
            table[key] = (obj, value)
            return value

        wrapper.cache_clear = table.clear  # type: ignore[attr-defined]
        return wrapper

    return decorate
