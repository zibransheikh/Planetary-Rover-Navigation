"""Task-2 simulator tests: generation, movement, sensing, battery, collisions, goal, seeds, leakage, presets, runner."""
import ast
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import Config, EnvConfig, load_config
from src.environment import procgen as G
from src.environment import terrain as T
from src.environment.advanced_env import FEATURES, R, W, AdvancedRoverEnv
from src.evaluation.evaluator import AdvancedEvaluator, make_env
from src.evaluation.replay import load_record, plot_episode, record_episode, save_record, trajectory_hash
from src.evaluation.runner import evaluate, run_episode
from src.policies.sim32_heuristic import ReactiveExplorer

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ["id", "terrain_shift", "sensor_shift", "dynamics_shift", "combined_shift"]
WIN0 = FEATURES.index("win_r-2_c-2")
VIS0 = FEATURES.index("visits_r-2_c-2")
ACT = {"up": 0, "down": 1, "left": 2, "right": 3}


def cfg32(**kw):
    base = dict(kind="advanced", size=32, battery=250, p_normal=.60, p_sand=.15, p_rock=.10)
    base.update(kw)
    return EnvConfig(**base)


def quiet(**kw):
    """No slip, perfect sensor: isolates the movement/energy rules."""
    return cfg32(slip_prob=(0, 0, 0), sensor_miss=0.0, sensor_false=0.0, sensor_sigma=0.0, **kw)


def flat(n=32, fill=T.NORMAL):
    g = np.full((n, n), fill, dtype=np.int64)
    return g


def window(obs):
    return obs[WIN0:WIN0 + W * W].reshape(W, W)


class TestGeneration(unittest.TestCase):
    def test_composition_and_clustering(self):
        comp = np.zeros(4)
        for s in range(40):
            g = G.generate_terrain(np.random.default_rng(s), 32, .6, .15, .10, 2)
            comp += np.bincount(g.ravel(), minlength=4) / g.size
        comp /= 40
        np.testing.assert_allclose(comp, [.6, .15, .10, .15], atol=0.01)

    def test_clustered_vs_iid(self):
        def adj_same(g):
            return float((g[:, 1:] == g[:, :-1]).mean())
        smooth = np.mean([adj_same(G.generate_terrain(np.random.default_rng(s), 32, .5, .2, .15, 3)) for s in range(20)])
        iid = np.mean([adj_same(G.generate_terrain(np.random.default_rng(s), 32, .5, .2, .15, 0)) for s in range(20)])
        self.assertGreater(smooth, iid + 0.1)

    def test_generated_maps_valid_for_every_preset(self):
        for name in PRESETS:
            cfg = load_config(ROOT / f"configs/sim32_{name}.yaml").env
            env, ev = make_env(cfg, 0)
            for s in range(20000, 20015):
                env.reset(seed=s)
                g = ev.grid
                self.assertEqual(g.shape, (32, 32))
                self.assertTrue((g[:3, :3] == T.NORMAL).all() and (g[-3:, -3:] == T.NORMAL).all())
                self.assertLessEqual(ev.optimal_cost, cfg.battery)
                self.assertLessEqual(G.dijkstra_lut(g, (0, 0), G.energy_lut(cfg.ref_energy_cost))[31, 31],
                                     cfg.max_opt_cost_frac * cfg.battery)

    def test_energy_table_validation(self):
        with self.assertRaises(ValueError):
            G.energy_lut((0, 3, 5))
        with self.assertRaises(ValueError):
            G.energy_lut((1, 3))


class TestSeeds(unittest.TestCase):
    def _traj_hash(self, cfg, seed, policy_seed=0):
        env, ev = make_env(cfg, 0)
        rec = record_episode(env, ev, ReactiveExplorer(policy_seed), seed)
        return trajectory_hash(rec), rec

    def test_same_seed_same_world_and_trajectory(self):
        a, _ = self._traj_hash(cfg32(), 20007)
        b, _ = self._traj_hash(cfg32(), 20007)
        self.assertEqual(a, b)

    def test_different_seed_different_map(self):
        e1, v1 = make_env(cfg32(), 0)
        e1.reset(seed=1); g1 = v1.grid
        e1.reset(seed=2)
        self.assertFalse(np.array_equal(g1, v1.grid))

    def test_stream_reproducible(self):
        maps = []
        for _ in range(2):
            env, ev = make_env(cfg32(), 3)
            maps.append([ev.grid for _ in range(3) if env.reset() is not None])
        for x, y in zip(*maps):
            np.testing.assert_array_equal(x, y)

    def test_sensor_and_dynamics_presets_keep_the_same_maps(self):
        ref, _ = make_env(load_config(ROOT / "configs/sim32_id.yaml").env, 0)
        ref_ev = AdvancedEvaluator(ref)
        for name in ("sensor_shift", "dynamics_shift"):
            env, ev = make_env(load_config(ROOT / f"configs/sim32_{name}.yaml").env, 0)
            for s in range(20000, 20025):
                ref.reset(seed=s); env.reset(seed=s)
                np.testing.assert_array_equal(ref_ev.grid, ev.grid, err_msg=f"{name} seed {s}")

    def test_sensor_stream_independent_of_dynamics(self):
        """Same seed, different slip tables -> identical first observation noise (sensor stream untouched)."""
        o1 = make_env(cfg32(), 0)[0].reset(seed=5)
        o2 = make_env(cfg32(slip_prob=(.3, .3, .3)), 0)[0].reset(seed=5)
        np.testing.assert_array_equal(o1, o2)


class TestMovementAndEnergy(unittest.TestCase):
    def setUp(self):
        self.env, self.ev = make_env(quiet(), 0)

    def _world(self, grid, start=(5, 5), goal=(31, 31)):
        return self.ev.reset_with_map(grid, start=start, goal=goal)

    def test_actions_move_the_right_way(self):
        for name, (dr, dc) in zip(("up", "down", "left", "right"), T.DELTAS):
            self._world(flat())
            self.env.step(ACT[name])
            self.assertEqual(self.env.position, (5 + dr, 5 + dc), name)

    def test_energy_by_terrain(self):
        for terr, cost in ((T.NORMAL, 1), (T.SAND, 3), (T.ROCK, 5)):
            g = flat(); g[5, 6] = terr
            self._world(g)
            _, _, _, info = self.env.step(ACT["right"])
            self.assertEqual((info["energy_used"], self.env.battery), (cost, 250 - cost))

    def test_dynamics_shift_energy_table_is_used(self):
        env, ev = make_env(quiet(energy_cost=(1, 4, 6)), 0)
        g = flat(); g[5, 6] = T.SAND
        ev.reset_with_map(g, start=(5, 5))
        env.step(ACT["right"])
        self.assertEqual(env.battery, 250 - 4)

    def test_obstacle_collision(self):
        g = flat(); g[5, 6] = T.OBSTACLE
        self._world(g)
        _, r, done, info = self.env.step(ACT["right"])
        self.assertEqual((self.env.position, self.env.battery, r, info["event"], done), ((5, 5), 249, -2.0, "obstacle_collision", False))

    def test_boundary_collision_costs_one_even_on_rock(self):
        g = flat(); g[0, 0] = T.ROCK
        self._world(g, start=(0, 0))
        _, r, _, info = self.env.step(ACT["up"])
        self.assertEqual((self.env.battery, r, info["event"]), (249, -2.0, "boundary_collision"))

    def test_goal_reward_and_reason(self):
        self._world(flat(), start=(30, 31), goal=(31, 31))
        _, r, done, info = self.env.step(ACT["down"])
        self.assertTrue(done)
        self.assertEqual(info["reason"], "goal")
        self.assertAlmostEqual(r, 100 + 50 * 249 / 250)

    def test_goal_on_exactly_zero_battery_is_success(self):
        self._world(flat(), start=(30, 31), goal=(31, 31))
        self.env.battery = 1
        _, _, done, info = self.env.step(ACT["down"])
        self.assertEqual((done, info["reason"], self.env.battery), (True, "goal", 0))

    def test_battery_depletion_by_moves_and_stall(self):
        g = flat(fill=T.ROCK)
        self._world(g)
        self.env.battery = 7
        _, _, done, info = self.env.step(ACT["right"])      # 7 -> 2
        self.assertFalse(done)
        _, r, done, info = self.env.step(ACT["right"])      # costs 5 > 2: stall
        self.assertEqual((done, info["reason"], info["event"], self.env.battery, r), (True, "battery_depleted", "stalled", 0, -20.0))
        self.assertEqual(self.env.position, (5, 6))

    def test_depletion_by_collision(self):
        g = flat(); g[5, 6] = T.OBSTACLE
        self._world(g)
        self.env.battery = 1
        _, r, done, info = self.env.step(ACT["right"])
        self.assertEqual((done, info["reason"], self.env.battery, r), (True, "battery_depleted", 0, -10.0))

    def test_step_after_done_raises(self):
        self._world(flat(), start=(30, 31), goal=(31, 31))
        self.env.step(ACT["down"])
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_timeout_when_max_steps_below_battery(self):
        env, ev = make_env(quiet(max_steps=3), 0)
        ev.reset_with_map(flat(), start=(5, 5))
        for _ in range(3):
            _, _, done, info = env.step(ACT["right"] if _ % 2 == 0 else ACT["left"])
        self.assertEqual((done, info["reason"], info["truncated"], info["terminated"]), (True, "timeout", True, False))

    def test_battery_never_negative_and_accounting(self):
        for name in PRESETS:
            env, ev = make_env(load_config(ROOT / f"configs/sim32_{name}.yaml").env, 0)
            rng = np.random.default_rng(0)

            class Rnd:
                def act(self, o):
                    return int(rng.integers(4))
            rows, s = evaluate(env, ev, Rnd(), range(20000, 20006))
            self.assertGreaterEqual(s["min_battery_seen"], 0)
            for r in rows:
                self.assertEqual(r["success"], int(r["reason"] == "goal"))
                self.assertLessEqual(r["energy_used"], 250)


class TestSlip(unittest.TestCase):
    def test_slip_rate_matches_config_and_is_lateral(self):
        env, ev = make_env(cfg32(slip_prob=(0.3, 0.3, 0.3), sensor_miss=0, sensor_false=0, sensor_sigma=0), 0)
        ev.reset_with_map(flat(), start=(15, 15))
        n, slips, lateral = 600, 0, 0
        for _ in range(n):
            if env._done or env.battery < 5:
                ev.reset_with_map(flat(), start=(15, 15))
            before = env.position
            _, _, _, info = env.step(ACT["up"])  # commanded: vertical
            if info["slipped"]:
                slips += 1
                lateral += env.position[0] == before[0] and env.position != before
            # keep the rover in the middle so walls do not interfere
            if abs(env.position[0] - 15) > 4 or abs(env.position[1] - 15) > 4:
                ev.reset_with_map(flat(), start=(15, 15))
        self.assertAlmostEqual(slips / n, 0.3, delta=0.06)
        self.assertEqual(lateral, slips)  # every slip slid sideways (vertical command -> horizontal move)

    def test_slip_can_cause_collision_next_to_obstacle(self):
        env, ev = make_env(cfg32(slip_prob=(1.0, 1.0, 1.0), sensor_miss=0, sensor_false=0, sensor_sigma=0), 0)
        g = flat(); g[15, 14] = T.OBSTACLE; g[15, 16] = T.OBSTACLE
        ev.reset_with_map(g, start=(15, 15))
        _, _, _, info = env.step(ACT["up"])  # free cell above, but slip is certain -> sideways into an obstacle
        self.assertEqual((info["event"], info["slipped"], env.position), ("obstacle_collision", True, (15, 15)))

    def test_no_slip_when_probability_zero(self):
        env, ev = make_env(quiet(), 0)
        ev.reset_with_map(flat(), start=(15, 15))
        for _ in range(30):
            _, _, _, info = env.step(ACT["up"] if _ % 2 else ACT["down"])
            self.assertFalse(info["slipped"])


class TestSensing(unittest.TestCase):
    def _env(self, **kw):
        return make_env(cfg32(slip_prob=(0, 0, 0), **kw), 0)

    def test_perfect_sensor_returns_true_patch(self):
        env, ev = self._env(sensor_miss=0, sensor_false=0, sensor_sigma=0)
        g = flat(); g[10, 12] = T.SAND; g[11, 11] = T.ROCK; g[9, 9] = T.OBSTACLE
        obs = ev.reset_with_map(g, start=(10, 10))
        w = window(obs)  # centre = (10,10); rows 8..12, cols 8..12
        self.assertEqual(w[2, 4], 0.3); self.assertEqual(w[3, 3], 0.6); self.assertEqual(w[1, 1], 1.0); self.assertEqual(w[2, 2], 0.0)

    def test_map_edge_reads_as_blocked(self):
        env, ev = self._env(sensor_miss=0, sensor_false=0, sensor_sigma=0)
        w = window(ev.reset_with_map(flat(), start=(0, 0)))
        self.assertTrue((w[:2, :] == 1.0).all() and (w[:, :2] == 1.0).all())
        self.assertTrue((w[2:, 2:] < 1.0).all())

    def test_miss_and_false_rates(self):
        env, ev = self._env(sensor_miss=0.25, sensor_false=0.10, sensor_sigma=0.0)
        g = flat(); g[10:13, 8:13] = T.OBSTACLE  # only row 10 falls inside the window
        misses = falses = n_obs = n_free = 0
        for k in range(300):
            w = window(ev.reset_with_map(g.copy(), start=(8, 10)))  # window = rows 6..10, cols 8..12
            truth = g[6:11, 8:13] == T.OBSTACLE
            det = w >= 0.999
            misses += int((truth & ~det).sum()); n_obs += int(truth.sum())
            falses += int((~truth & det).sum()); n_free += int((~truth).sum())
        self.assertAlmostEqual(misses / n_obs, 0.25, delta=0.06)
        self.assertAlmostEqual(falses / n_free, 0.10, delta=0.03)

    def test_reading_noise_sigma(self):
        env, ev = self._env(sensor_miss=0, sensor_false=0, sensor_sigma=0.1)
        vals = np.concatenate([window(ev.reset_with_map(flat(), start=(15, 15))).ravel() for _ in range(200)])
        free = vals[vals < 0.9]  # clipping at 0 truncates normal terrain, so check the sand variant too
        g = flat(fill=T.SAND)
        sand = np.concatenate([window(ev.reset_with_map(g, start=(15, 15))).ravel() for _ in range(200)])
        self.assertAlmostEqual(float(sand.mean()), 0.3, delta=0.01)
        self.assertAlmostEqual(float(sand.std()), 0.1, delta=0.01)
        self.assertGreater(len(free), 0)

    def test_window_is_exactly_5x5_and_ignores_cells_beyond(self):
        env, ev = self._env(sensor_miss=0, sensor_false=0, sensor_sigma=0)
        g = flat(); g[10, 13] = T.OBSTACLE  # 3 columns away: outside the window
        w = window(ev.reset_with_map(g, start=(10, 10)))
        self.assertTrue((w < 1.0).all())
        self.assertEqual(len(w.ravel()), 25)


class TestExploredMap(unittest.TestCase):
    def setUp(self):
        self.env, self.ev = make_env(quiet(), 0)

    def test_coverage_grows_and_known_matches_window(self):
        self.ev.reset_with_map(flat(), start=(10, 10))
        m = self.env.explored_map()
        self.assertEqual(int(m.known.sum()), 25)
        self.assertTrue(m.known[8:13, 8:13].all())
        self.env.step(ACT["right"])
        self.assertEqual(int(self.env.explored_map().known.sum()), 30)  # 5 new cells in the new column

    def test_visits_and_obstacle_memory(self):
        g = flat(); g[10, 11] = T.OBSTACLE
        self.ev.reset_with_map(g, start=(10, 10))
        m = self.env.explored_map()
        self.assertEqual(m.obstacle_frac[10, 11], 1.0)
        self.assertEqual(m.visits[10, 10], 1)
        self.env.step(ACT["right"])  # bump
        m = self.env.explored_map()
        self.assertTrue(m.bumped[10, 11])
        self.assertEqual(m.visits[10, 10], 1)

    def test_bump_marks_obstacle_the_sensor_missed(self):
        env, ev = make_env(cfg32(slip_prob=(0, 0, 0), sensor_miss=1.0, sensor_false=0, sensor_sigma=0), 0)
        g = flat(); g[10, 11] = T.OBSTACLE
        ev.reset_with_map(g, start=(10, 10))
        self.assertEqual(env.explored_map().obstacle_frac[10, 11], 0.0)   # sensor missed it
        env.step(ACT["right"])
        self.assertEqual(env.explored_map().obstacle_frac[10, 11], 1.0)   # touch is certain

    def test_snapshot_is_a_copy(self):
        self.ev.reset_with_map(flat(), start=(10, 10))
        m = self.env.explored_map()
        m.seen[:] = 99
        self.assertEqual(int(self.env.explored_map().seen.max()), 1)

    def test_slip_estimate_and_confidence_in_range(self):
        env, ev = make_env(cfg32(slip_prob=(.5, .5, .5)), 0)
        ev.reset_with_map(flat(), start=(15, 15))
        for t in range(40):
            if env._done:
                break
            o, *_ = env.step(t % 4)
            self.assertTrue(0 < o[FEATURES.index("slip_estimate")] < 1)
            self.assertTrue(0 <= o[FEATURES.index("sensor_confidence")] <= 1)
        self.assertGreater(o[FEATURES.index("slip_estimate")], 0.25)

    def test_noisier_sensor_lowers_confidence(self):
        def mean_conf(**kw):
            env, ev = make_env(cfg32(slip_prob=(0, 0, 0), **kw), 0)
            cs = []
            for s in range(20000, 20006):
                o = env.reset(seed=s); done = False; k = 0
                while not done and k < 40:
                    o, _, done, _ = env.step((k // 3) % 4 if k % 2 else 3); k += 1
                    cs.append(o[FEATURES.index("sensor_confidence")])
            return float(np.mean(cs))
        self.assertGreater(mean_conf(sensor_miss=0, sensor_false=0, sensor_sigma=0), mean_conf(sensor_miss=.25, sensor_false=.15, sensor_sigma=.2) + 0.1)


class TestLeakage(unittest.TestCase):
    def test_observation_depends_only_on_window(self):
        """Terrain changes OUTSIDE the 5x5 window must not change the observation (fixed RNG consumption per step)."""
        rng = np.random.default_rng(0)
        for trial in range(60):
            cfg = cfg32()
            e1, v1 = make_env(cfg, trial); e2, v2 = make_env(cfg, trial)
            g1 = G.generate_terrain(rng, 32, .6, .15, .1, 2); G.clear_blocks(g1, (0, 0), (31, 31))
            g2 = g1.copy()
            pos = (int(rng.integers(3, 29)), int(rng.integers(3, 29)))
            mask = np.ones((32, 32), bool); mask[pos[0] - R:pos[0] + R + 1, pos[1] - R:pos[1] + R + 1] = False
            g2[mask] = rng.integers(0, 4, size=int(mask.sum()))
            g1[pos] = g2[pos] = T.NORMAL
            np.testing.assert_array_equal(v1.reset_with_map(g1, start=pos), v2.reset_with_map(g2, start=pos))
            a = int(rng.integers(4))
            o1, o2 = e1.step(a)[0], e2.step(a)[0]
            # the next window overlaps terrain that differs between the two maps, so only the scalars must agree
            if e1.position == e2.position:
                np.testing.assert_array_equal(o1[:4], o2[:4])

    def test_rng_consumption_does_not_depend_on_terrain(self):
        """Maps that differ INSIDE the window must leave both random streams in the same state (keeps paired runs aligned)."""
        rng = np.random.default_rng(1)
        e1, v1 = make_env(cfg32(), 4); e2, v2 = make_env(cfg32(), 4)
        g1 = flat(); g2 = rng.integers(0, 4, size=(32, 32)).astype(np.int64); g2[14:17, 14:17] = T.NORMAL
        v1.reset_with_map(g1, start=(15, 15)); v2.reset_with_map(g2, start=(15, 15))
        for k in range(6):
            e1.step(k % 4); e2.step(k % 4)
            self.assertEqual(e1._sens_rng.bit_generator.state, e2._sens_rng.bit_generator.state)
            self.assertEqual(e1._slip_rng.bit_generator.state, e2._slip_rng.bit_generator.state)

    def test_info_keys_are_whitelisted(self):
        env, ev = make_env(cfg32(), 0)
        env.reset(seed=1)
        _, _, _, info = env.step(0)
        self.assertEqual(set(info), {"event", "reason", "terminated", "truncated", "energy_used", "steps", "battery", "slipped"})

    def test_public_api_exposes_no_ground_truth(self):
        env, _ = make_env(cfg32(), 0)
        env.reset(seed=1)
        public = [n for n in dir(env) if not n.startswith("_")]
        for n in public:
            v = getattr(env, n)
            self.assertFalse(isinstance(v, np.ndarray) and v.shape == (32, 32), f"public ndarray attribute {n}")
        for banned in ("grid", "world", "optimal_cost", "slip", "lut", "energy_table", "true"):
            self.assertFalse([n for n in public if banned in n.lower()], banned)
        m = env.explored_map()
        self.assertFalse(hasattr(m, "grid"))

    def test_policy_modules_do_not_import_evaluator_or_generator(self):
        for f in (ROOT / "src/policies").glob("*.py"):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for m in mods:
                    self.assertFalse(m.startswith("src.evaluation") or m.endswith("procgen"), f"{f.name} imports {m}")

    def test_observation_has_documented_layout(self):
        env, _ = make_env(cfg32(), 0)
        o = env.reset(seed=1)
        self.assertEqual((o.shape, env.obs_dim, len(FEATURES)), ((59,), 59, 59))
        self.assertTrue(np.isfinite(o).all())
        self.assertTrue(((o[WIN0:WIN0 + 25] >= 0) & (o[WIN0:WIN0 + 25] <= 1)).all())


class TestPresets(unittest.TestCase):
    GROUPS = {"terrain": {"p_normal", "p_sand", "p_rock", "gen_corr"},
              "sensor": {"sensor_miss", "sensor_false", "sensor_sigma"},
              "dynamics": {"energy_cost", "slip_prob"}}

    def test_each_shift_changes_only_its_own_parameters(self):
        base = load_config(ROOT / "configs/sim32_id.yaml").env
        for name, allowed in (("terrain_shift", {"terrain"}), ("sensor_shift", {"sensor"}), ("dynamics_shift", {"dynamics"}),
                              ("combined_shift", {"terrain", "sensor", "dynamics"})):
            cfg = load_config(ROOT / f"configs/sim32_{name}.yaml").env
            changed = {f for f in base.__dataclass_fields__ if str(getattr(base, f)) != str(getattr(cfg, f)) and
                       list(np.ravel(getattr(base, f))) != list(np.ravel(getattr(cfg, f)))}
            allowed_fields = set().union(*(self.GROUPS[g] for g in allowed))
            self.assertTrue(changed <= allowed_fields, f"{name}: unexpected changes {changed - allowed_fields}")
            for g in allowed:
                self.assertTrue(changed & self.GROUPS[g], f"{name} does not change any {g} parameter")

    def test_all_presets_share_size_battery_and_kind(self):
        for name in PRESETS:
            c = load_config(ROOT / f"configs/sim32_{name}.yaml").env
            self.assertEqual((c.kind, c.size, c.battery), ("advanced", 32, 250))

    def test_shifts_are_harder_in_the_intended_direction(self):
        def stats(name):
            env, ev = make_env(load_config(ROOT / f"configs/sim32_{name}.yaml").env, 0)
            costs = []
            for s in range(20000, 20020):
                env.reset(seed=s); costs.append(ev.optimal_cost)
            return float(np.mean(costs))
        self.assertGreater(stats("terrain_shift"), stats("id") + 10)
        self.assertGreater(stats("dynamics_shift"), stats("id"))
        self.assertAlmostEqual(stats("sensor_shift"), stats("id"))

    def test_id_preset_is_solvable_from_onboard_observations(self):
        env, ev = make_env(load_config(ROOT / "configs/sim32_id.yaml").env, 0)
        _, s = evaluate(env, ev, ReactiveExplorer(0), range(20000, 20040))
        self.assertGreater(s["success_rate"], 0.4)


class TestEvaluatorAndRunner(unittest.TestCase):
    def test_evaluator_costs_use_true_energy_table(self):
        env, ev = make_env(quiet(energy_cost=(1, 4, 6)), 0)
        g = flat(); g[5, :] = T.SAND
        ev.reset_with_map(g, start=(5, 0), goal=(5, 5))
        self.assertEqual(ev.optimal_cost, ev.cost_to_goal())
        self.assertEqual(ev.optimal_cost, G.dijkstra_lut(g, (5, 0), G.energy_lut((1, 4, 6)))[5, 5])
        self.assertEqual(ev.optimal_path()[0], (5, 0)); self.assertEqual(ev.optimal_path()[-1], (5, 5))

    def test_budget_margin_sign(self):
        env, ev = make_env(quiet(), 0)
        ev.reset_with_map(flat(), start=(0, 0))
        self.assertGreater(ev.budget_margin(), 0)
        env.battery = 10
        self.assertLess(ev.budget_margin(), 0)

    def test_run_episode_row_is_consistent(self):
        env, ev = make_env(load_config(ROOT / "configs/sim32_id.yaml").env, 0)
        row = run_episode(env, ev, ReactiveExplorer(0), 20001)
        self.assertEqual(row["success"], int(row["reason"] == "goal"))
        self.assertEqual(row["steps"], env.steps)

    def test_training_loop_runs_on_advanced_env(self):
        """The existing agent-agnostic PPO loop works unchanged on the 59-d observation (stand-in agent, no torch)."""
        from src.training.loops import train_ppo

        class Stub:
            def __init__(self): self.rng = np.random.default_rng(0); self.n = 0
            def sample(self, o): return int(self.rng.integers(4)), 0.0, 0.0
            def act(self, o, greedy=True): return int(self.rng.integers(4))
            def value(self, o): return 0.0
            def store(self, *a): self.n += 1
            def update(self, v): pass
        cfg = Config(env=cfg32(), run=replace(Config().run, episodes=3, eval_every=3, eval_episodes=2))
        with tempfile.TemporaryDirectory() as d:
            s = train_ppo(Stub(), cfg, d)
        self.assertEqual(s["episodes"], 3)
        self.assertEqual(s["greedy_val_200"]["episodes"], 200)


class TestReplay(unittest.TestCase):
    def test_record_roundtrip_and_redraw(self):
        env, ev = make_env(load_config(ROOT / "configs/sim32_id.yaml").env, 0)
        rec = record_episode(env, ev, ReactiveExplorer(0), 20003)
        with tempfile.TemporaryDirectory() as d:
            save_record(rec, Path(d) / "e.json")
            back = load_record(Path(d) / "e.json")
            self.assertEqual(trajectory_hash(back), trajectory_hash(rec))
            self.assertEqual(len(back["steps"]), len(rec["steps"]))
            fig = plot_episode(back, Path(d) / "f.png", t=len(back["steps"]) // 2)
            self.assertTrue((Path(d) / "f.png").stat().st_size > 5000)
            import matplotlib.pyplot as plt; plt.close(fig)

    def test_rerun_reproduces_record(self):
        hashes = []
        for _ in range(2):
            env, ev = make_env(load_config(ROOT / "configs/sim32_combined_shift.yaml").env, 0)
            hashes.append(trajectory_hash(record_episode(env, ev, ReactiveExplorer(7), 20005)))
        self.assertEqual(*hashes)


if __name__ == "__main__":
    unittest.main()
