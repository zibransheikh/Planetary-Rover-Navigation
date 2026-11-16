import hashlib
import unittest

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from src.evaluation.runner import run_episode
from src.policies.baseline_policies import GreedyLocalPolicy, RandomPolicy
from tests.helpers import fresh


def digest(env_seed, policy_cls, policy_seed, episodes=25):
    env, ev = fresh(seed=env_seed)
    rows = [run_episode(env, ev, policy_cls(policy_seed + i), seed=i) for i in range(episodes)]
    return hashlib.sha256(repr([(r["map"], r["steps"], round(r["reward"], 6), r["battery_left"]) for r in rows]).encode()).hexdigest()


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_map_different_seed_different_map(self):
        env, ev = fresh()
        env.reset(seed=7); a = ev.grid
        env.reset(seed=7); b = ev.grid
        env.reset(seed=8); c = ev.grid
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())

    def test_map_stream_reproducible_across_env_instances(self):
        e1, v1 = fresh(seed=3); e2, v2 = fresh(seed=3); e3, v3 = fresh(seed=4)
        for _ in range(5):
            e1.reset(); e2.reset(); e3.reset()
            self.assertTrue((v1.grid == v2.grid).all())
        self.assertFalse((v1.grid == v3.grid).all())

    def test_trajectories_reproducible(self):
        for cls in (RandomPolicy, GreedyLocalPolicy):
            self.assertEqual(digest(0, cls, 100), digest(0, cls, 100))
        self.assertNotEqual(digest(0, RandomPolicy, 100), digest(0, RandomPolicy, 200))

    def test_all_generated_maps_are_solvable_within_battery(self):
        env, ev = fresh()
        for s in range(300):
            env.reset(seed=s)
            self.assertLess(ev.optimal_cost, T.INF)
            self.assertLessEqual(ev.optimal_cost, env.max_battery)
            g = ev.grid
            self.assertEqual((g[0, 0], g[-1, -1]), (T.NORMAL, T.NORMAL))

    def test_terrain_distribution_matches_config(self):
        rng = np.random.default_rng(0)
        g = T.generate_grid(rng, 400, 0.55, 0.15, 0.15)
        freq = np.bincount(g.ravel(), minlength=4) / g.size
        np.testing.assert_allclose(freq, [0.55, 0.15, 0.15, 0.15], atol=0.01)


if __name__ == "__main__":
    unittest.main()
