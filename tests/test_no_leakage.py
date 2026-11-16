"""Policy-facing interface must not expose hidden full-map information."""
import ast
import unittest
from pathlib import Path

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from tests.helpers import ROOT, fresh, flat_grid

POLICY_SIDE = [ROOT / "src" / "agents", ROOT / "src" / "policies"]
FORBIDDEN_NAMES = {"Evaluator", "make_env", "evaluator", "_world", "_install_world", "dijkstra", "dijkstra_path",
                   "reset_with_map", "optimal_cost"}
FORBIDDEN_MODULES = ("src.evaluation", "src.environment.terrain")
ALLOWED_INFO_KEYS = {"event", "reason", "terminated", "truncated", "energy_used", "steps", "battery"}


class TestNoLeakage(unittest.TestCase):
    def test_onboard_observation_is_15d_without_privileged_features(self):
        env, _ = fresh()
        self.assertEqual(env.reset(seed=0).shape, (15,))
        self.assertFalse(any("PRIVILEGED" in n for n in env.feature_names))
        self.assertEqual(len(env.feature_names), env.obs_dim)

    def test_observation_invariant_to_terrain_outside_local_patch(self):
        """Core leakage test: change the hidden map everywhere the rover cannot see -> identical observation."""
        env, ev = fresh()
        rng = np.random.default_rng(0)
        checked = 0
        for seed in range(150):
            env.reset(seed=seed)
            base = ev.grid
            pos = (int(rng.integers(0, 10)), int(rng.integers(0, 10)))
            if base[pos] == T.OBSTACLE:
                continue
            ev.reset_with_map(base, start=pos); o1 = env._observe()
            mutated = base.copy()
            mask = np.ones_like(mutated, dtype=bool)
            mask[max(0, pos[0] - 1):pos[0] + 2, max(0, pos[1] - 1):pos[1] + 2] = False
            mask[9, 9] = False  # goal cell must stay valid
            mutated[mask] = rng.integers(0, 4, size=int(mask.sum()))
            ev.reset_with_map(mutated, start=pos); o2 = env._observe()
            np.testing.assert_array_equal(o1, o2)
            checked += 1
        self.assertGreater(checked, 100)

    def test_legacy17_mode_DOES_leak_so_the_test_above_has_power(self):
        env, ev = fresh(EnvConfig(obs_mode="legacy17"))
        open_map = flat_grid()
        wall = open_map.copy(); wall[5, :] = T.ROCK  # far from (0,0), outside the 3x3 window, but raises min cost
        ev.reset_with_map(open_map); o1 = env._observe()
        ev.reset_with_map(wall); o2 = env._observe()
        self.assertTrue((o1[8:] == o2[8:]).all())          # local patch identical
        self.assertFalse(np.isclose(o1[5], o2[5]))         # cost_to_goal_ratio differs => leak detected
        self.assertFalse(np.isclose(o1[6], o2[6]))         # budget_margin differs

    def test_info_dict_contains_only_non_privileged_keys(self):
        env, _ = fresh()
        env.reset(seed=0)
        for a in (1, 3, 0, 2, 1, 1):
            _, _, done, info = env.step(a)
            self.assertLessEqual(set(info), ALLOWED_INFO_KEYS)
            if done:
                break

    def test_public_env_api_exposes_no_ground_truth(self):
        env, _ = fresh()
        env.reset(seed=0)
        public = [a for a in dir(env) if not a.startswith("_")]
        for bad in ("grid", "goal", "world", "optimal_cost", "start"):
            self.assertNotIn(bad, public)
        for a in public:
            v = getattr(env, a)
            self.assertFalse(isinstance(v, np.ndarray) and v.shape == (env.size, env.size), a)

    def test_agent_and_policy_code_never_touches_privileged_symbols(self):
        offenders = []
        for folder in POLICY_SIDE:
            for f in folder.rglob("*.py"):
                tree = ast.parse(f.read_text())
                for node in ast.walk(tree):
                    names = set()
                    if isinstance(node, ast.Name):
                        names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        names.add(node.attr)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        mod = getattr(node, "module", "") or ""
                        names |= {a.name for a in node.names}
                        if any(mod.startswith(m) for m in FORBIDDEN_MODULES):
                            offenders.append((f.name, mod))
                    bad = names & FORBIDDEN_NAMES
                    if bad:
                        offenders.append((f.name, sorted(bad)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
