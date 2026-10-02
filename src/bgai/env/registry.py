"""Gymnasium registration, so TM-Env is reachable by id.

    import bgai.env.registry as tm; tm.register_envs()
    env = gymnasium.make("bgai/TerraMystica-v0", player_count=4)

Registration is explicit rather than an import side effect: ``bgai.env``
must import without gymnasium (it is a dev-group dependency), and a package
that quietly mutates a global registry on import is hard to reason about in
a test suite.

The AEC env has no equivalent: PettingZoo environments are constructed by
calling a module's ``env()``, which ``bgai.env.aec`` provides.
"""

from __future__ import annotations

ENV_ID = "bgai/TerraMystica-v0"
"""Gymnasium id for the single-agent wrapper."""

AEC_NAME = "terra_mystica_v0"
"""``metadata['name']`` of the PettingZoo AEC env."""


def register_envs() -> None:
    """Register :data:`ENV_ID` with Gymnasium. Idempotent."""
    import gymnasium

    if ENV_ID in gymnasium.registry:
        return
    gymnasium.register(
        id=ENV_ID,
        entry_point="bgai.env.single:make_single_env",
        disable_env_checker=False,
        order_enforce=True,
    )
