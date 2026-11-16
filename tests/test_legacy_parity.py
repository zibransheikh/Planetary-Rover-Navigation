"""New env == original env on every transition where the original is internally consistent.

Parity is asserted on identical injected maps. Known, intentional divergences (documented in
docs/AUDIT_NOTES.md) are excluded: boundary bumps and moves that exceed the remaining battery.
"""
import unittest

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from tests.helpers import fresh, load_legacy_env_module


class TestLegacyParity(unittest.TestCase):
    def test_transitions_match_original_until_known_divergences(self):
        L = load_legacy_env_module()
        legacy = L.RoverEnv(size=10, battery=100)
        env, ev = fresh(EnvConfig(obs_mode="legacy17", reward_privileged_shaping=True))
        rng = np.random.default_rng(0)
        compared, episodes_full, resyncs = 0, 0, 0
        for seed in range(200):
            env.reset(seed=seed)
            grid = ev.grid
            # install the same hidden map in the original implementation
            legacy.grid = grid.astype(float)
            legacy.position, legacy.battery, legacy.steps = [0, 0], 100, 0
            legacy.optimal_cost = ev.optimal_cost
            np.testing.assert_allclose(env._observe(), legacy.get_state(), atol=1e-12)
            done = False
            while not done:
                a = int(rng.integers(4))
                x, y = env.position
                dx, dy = T.DELTAS[a]
                nx, ny = x + dx, y + dy
                oob = not (0 <= nx < 10 and 0 <= ny < 10)
                too_costly = (not oob) and grid[nx, ny] != T.OBSTACLE and T.ENERGY[int(grid[nx, ny])] > env.battery
                if oob or too_costly:  # intentional divergence (audited behaviour): step only the new env, then resync
                    _, _, done, _ = env.step(a)
                    legacy.position, legacy.battery, legacy.steps = list(env.position), env.battery, env.steps
                    resyncs += 1
                    continue
                o_new, r_new, d_new, _ = env.step(a)
                o_old, r_old, d_old = legacy.step(a)
                np.testing.assert_allclose(o_new, o_old, atol=1e-12)
                self.assertAlmostEqual(r_new, r_old, places=9)
                self.assertEqual(d_new, d_old)
                self.assertEqual(env.battery, legacy.battery)
                compared += 1
                done = d_new
            else:
                episodes_full += 1
        self.assertGreater(compared, 3000)  # test has statistical power
        self.assertGreater(resyncs, 100)  # divergent events were actually exercised

    def test_onboard_obs_is_legacy_obs_minus_two_privileged_features(self):
        L = load_legacy_env_module()
        legacy = L.RoverEnv(size=10, battery=100)
        env, ev = fresh()
        env.reset(seed=11)
        legacy.grid = ev.grid.astype(float)
        legacy.position, legacy.battery, legacy.steps = [0, 0], 100, 0
        legacy.optimal_cost = ev.optimal_cost
        np.testing.assert_allclose(env._observe(), np.delete(legacy.get_state(), [5, 6]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
