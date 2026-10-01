"""The ``bgai.env`` package surface.

The import-time contract matters: the engine-facing half of TM-Env must work
without gymnasium or pettingzoo installed (they are dev-group dependencies,
not runtime ones), so the API wrappers are imported lazily.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import bgai.env as env_pkg


def test_public_names_all_resolve() -> None:
    for name in env_pkg.__all__:
        assert hasattr(env_pkg, name), name


def test_lazy_accessors_build_the_real_envs() -> None:
    import gymnasium
    import pettingzoo

    assert isinstance(env_pkg.aec_env(), pettingzoo.utils.wrappers.OrderEnforcingWrapper)
    assert isinstance(env_pkg.single_env(), gymnasium.Env)


def test_engine_facing_modules_import_without_the_api_packages() -> None:
    """Blocks ``gymnasium`` and ``pettingzoo`` at import time and checks the
    engine-facing half still works end to end.
    """
    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                root = name.split(".")[0]
                if root in {"gymnasium", "pettingzoo"}:
                    raise ImportError(f"{root} is blocked for this test")
                return None

        sys.meta_path.insert(0, Blocker())
        for module in list(sys.modules):
            if module.split(".")[0] in {"gymnasium", "pettingzoo"}:
                del sys.modules[module]

        import random

        from bgai.env.codec import sim_from_bytes, sim_to_bytes
        from bgai.env.config import EnvConfig
        from bgai.env.core import Episode
        from bgai.env.reward import terminal_rewards
        from bgai.env.serialize import obs_from_bytes, obs_to_bytes

        episode = Episode.start(1, EnvConfig(player_count=3))
        rng = random.Random(1)
        while not episode.finished:
            faction, offer = episode.pending()
            obs = episode.observe(faction)
            assert obs_from_bytes(obs_to_bytes(obs)) == obs
            episode, _ = episode.step(rng.randrange(len(offer)))
        assert sim_from_bytes(sim_to_bytes(episode.sim)) == episode.sim
        assert terminal_rewards(EnvConfig().reward_mode, episode.vps())
        assert "gymnasium" not in sys.modules
        assert "pettingzoo" not in sys.modules
        print("OK")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout


def test_spaces_module_needs_gymnasium_and_says_so() -> None:
    """``bgai.env.spaces`` is the one module that genuinely requires
    gymnasium; it imports it at the top rather than pretending otherwise.
    """
    module = importlib.import_module("bgai.env.spaces")
    assert module.__doc__ is not None
    assert "lazily" in module.__doc__


def test_gymnasium_registration_is_idempotent_and_makeable() -> None:
    import gymnasium
    import numpy as np

    from bgai.env.registry import AEC_NAME, ENV_ID, register_envs

    register_envs()
    register_envs()  # idempotent
    assert ENV_ID in gymnasium.registry

    env = gymnasium.make(ENV_ID, player_count=3, opponent="greedy")
    try:
        obs, info = env.reset(seed=1)
        assert set(obs) == {"observation", "action_mask"}
        assert info["opponent"] == "greedy"
        obs, reward, terminated, truncated, _ = env.step(
            int(np.flatnonzero(obs["action_mask"])[0])
        )
        assert not terminated and not truncated
        assert env.spec is not None and env.spec.id == ENV_ID
    finally:
        env.close()

    from bgai.env.aec import TerraMysticaAECEnv

    assert TerraMysticaAECEnv.metadata["name"] == AEC_NAME


def test_check_env_through_gymnasium_make_is_warning_free() -> None:
    """With a registered spec, Gymnasium can also exercise the alternative
    render modes, which is the one thing a bare instance cannot be checked for.
    """
    import warnings

    import gymnasium
    from gymnasium.utils.env_checker import check_env

    from bgai.env.registry import ENV_ID, register_envs

    register_envs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(gymnasium.make(ENV_ID, player_count=4).unwrapped, skip_render_check=False)
    assert [str(w.message) for w in caught] == []
