"""tests/test_engine_import_isolation.py

Regression test for a circular-import bug (Task 8 code review, Critical
1): ``actions_build.py`` did ``from bgai.engine.tm.leech import
offers_for_build, queue_leech`` at module top, and ``apply.py`` imports
``leech``/``actions_build`` for their handler-registration side effects
at its own module bottom. Importing ``bgai.engine.tm.leech`` (or
``bgai.engine.tm.actions_build``) as the *first* touch of the `tm`
package triggered: leech -> apply -> actions_build -> leech (partially
initialized, missing ``offers_for_build`` yet) -> ``ImportError``.

The existing test suite only ever imported ``bgai.engine.tm.apply``
first (directly or transitively via ``bgai.engine.tm.state``), which
happens to sidestep the cycle -- masking the bug ("import-order
accident"). This test imports each module below **first**, in a fresh
interpreter (subprocess), so no earlier import in the test process can
hide the failure. ``_MODULES`` must be kept in sync with every module
``apply.py`` bottom-imports for its handler-registration side effects
(``actions_build``/``actions_terraform``/``leech``/``actions_power`` as
of Task 10) -- each is a fresh potential cycle root, same as the
original ``leech``/``actions_build`` pair.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_MODULES = [
    "bgai.engine.tm.leech",
    "bgai.engine.tm.actions_build",
    "bgai.engine.tm.actions_terraform",
    "bgai.engine.tm.actions_power",
    "bgai.engine.tm.apply",
]


@pytest.mark.parametrize("module", _MODULES)
def test_module_imports_cleanly_as_the_first_tm_import(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`import {module}` failed as the first import in a fresh interpreter:\n{result.stderr}"
    )


def test_all_verb_families_register_regardless_of_import_order() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import bgai.engine.tm.leech\n"
                "from bgai.engine.tm.apply import HANDLERS\n"
                "for verb in ('leech', 'decline', 'build', 'upgrade', 'bridge', "
                "'gain_favor', 'gain_town', 'dig', 'transform', 'lose_spade', "
                "'action', 'lose_marker'):\n"
                "    assert verb in HANDLERS, verb\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
