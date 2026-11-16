import unittest

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from tests.helpers import fresh, flat_grid


class TestAccounting(unittest.TestCase):
    def test_battery_never_negative_and_reasons_consistent(self):
        env, ev = fresh()
        rng = np.random.default_rng(1)
        for ep in range(300):
            env.reset(seed=ep)
            done, info = False, {}
            while not done:
                _, _, done, info = env.step(int(rng.integers(4)))
                self.assertGreaterEqual(env.battery, 0)
                self.assertEqual(info["steps"], env.steps)
            self.assertIn(info["reason"], ("goal", "battery_depleted", "timeout"))
            self.assertEqual(info["terminated"] or info["truncated"], True)
            if info["reason"] == "battery_depleted":
                self.assertEqual(env.battery, 0)

    def test_optimal_path_reaches_goal_with_exact_energy(self):
        env, ev = fresh()
        for seed in range(100):
            env.reset(seed=seed)
            path, opt = ev.optimal_path(), ev.optimal_cost
            total_r, info = 0.0, {}
            for (a, b) in zip(path[:-1], path[1:]):
                action = T.DELTAS.index((b[0] - a[0], b[1] - a[1]))
                _, r, done, info = env.step(action)
            self.assertEqual(info["reason"], "goal")
            self.assertEqual(env.battery, env.max_battery - opt)
            self.assertEqual(env.steps, len(path) - 1)
            self.assertEqual(r, 100.0 + 50.0 * min(env.battery / opt, 1.0))

    def test_collision_costs_one_unit_regardless_of_terrain(self):
        env, ev = fresh()
        g = flat_grid(); g[0, 1] = T.ROCK
        ev.reset_with_map(g, start=(0, 1))        # standing on rock
        _, r, done, info = env.step(0)            # bump the top boundary
        self.assertEqual((env.battery, r, done, env.position), (99, -2.0, False, (0, 1)))
        self.assertEqual(info["event"], "boundary_collision")
        g[1, 1] = T.OBSTACLE
        ev.reset_with_map(g, start=(0, 1))
        _, r, _, info = env.step(1)               # walk into an obstacle
        self.assertEqual((env.battery, r, info["event"]), (99, -2.0, "obstacle_collision"))

    def test_insufficient_energy_stalls_and_clamps_to_zero(self):
        env, ev = fresh()
        g = flat_grid(); g[0, 1] = T.ROCK
        ev.reset_with_map(g)
        env.battery = 3
        _, r, done, info = env.step(3)            # rock costs 5 > 3
        self.assertEqual((env.battery, r, done, env.position), (0, -20.0, True, (0, 0)))
        self.assertEqual((info["event"], info["reason"], info["energy_used"]), ("stalled", "battery_depleted", 3))

    def test_arriving_on_goal_with_exactly_zero_battery_is_success(self):
        env, ev = fresh()
        g = flat_grid(4)
        env.size = 4  # tiny map via injection
        ev.reset_with_map(g, start=(3, 2), goal=(3, 3))
        env.battery = 1
        _, r, done, info = env.step(3)
        self.assertEqual((info["reason"], env.battery, done), ("goal", 0, True))
        self.assertGreaterEqual(r, 100.0)

    def test_step_after_done_raises(self):
        env, ev = fresh()
        env.reset(seed=0); env.battery = 1
        env.step(0)  # boundary bump -> depleted
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_timeout_is_truncation_not_termination(self):
        env, ev = fresh(EnvConfig(max_steps=3))
        env.reset(seed=0)
        for _ in range(3):
            _, _, done, info = env.step(0)        # bump boundary repeatedly (corner start)
        self.assertEqual((done, info["truncated"], info["terminated"], info["reason"]), (True, True, False, "timeout"))

    def test_every_action_consumes_energy_so_episodes_are_finite(self):
        env, ev = fresh()
        for seed in range(50):
            env.reset(seed=seed)
            for _ in range(env.max_battery):
                _, _, done, info = env.step(seed % 4)
                self.assertGreaterEqual(info["energy_used"], 1 if info["event"] != "stalled" else 0)
                if done:
                    break
            self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
