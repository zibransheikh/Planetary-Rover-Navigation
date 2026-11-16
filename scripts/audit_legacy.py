"""Quantify defects in the ORIGINAL code (legacy/) and its committed result files. Needs only numpy.

Usage: python scripts/audit_legacy.py            -> prints a report, writes results/audit/legacy_audit.json
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import EnvConfig  # noqa: E402
from src.evaluation.evaluator import make_env  # noqa: E402
from src.policies.baseline_policies import GreedyLocalPolicy  # noqa: E402
from tests.helpers import load_legacy_env_module  # noqa: E402

L = load_legacy_env_module()
OUT: dict = {}


def seed_all(s):
    random.seed(s); np.random.seed(s)


# ---------------------------------------------------------------- A. map generation
def audit_maps(n=2000):
    seed_all(0)
    env = L.RoverEnv()
    unsolvable = unreachable_hidden = over_capacity = 0
    for _ in range(n):
        env.reset()
        dist = env._dijkstra(env.start)
        if not env._is_solvable():
            unsolvable += 1
        if env.goal not in dist:
            unreachable_hidden += 1          # optimal_cost silently falls back to max_battery
        elif dist[env.goal] > env.max_battery:
            over_capacity += 1               # reachable, but costs more than a full battery
    OUT["maps"] = {"n": n, "unsolvable_after_200_retries": unsolvable,
                   "unreachable_goal_hidden_by_fallback": unreachable_hidden, "cost_exceeds_battery": over_capacity}


# ---------------------------------------------------------------- B. accounting under a competent-ish policy
def audit_accounting(n=600):
    seed_all(1)
    env = L.RoverEnv()
    pol = GreedyLocalPolicy(0)
    neg = succ_pos = succ_nonpos = reward_rule_miss = n_goal = 0
    for _ in range(n):
        s = env.reset()
        done = False
        while not done:
            s, r, done = env.step(pol.act(s))
        at_goal = tuple(env.position) == env.goal
        reward_success = r >= 100
        neg += env.battery < 0
        n_goal += at_goal
        succ_nonpos += at_goal and env.battery <= 0
        reward_rule_miss += at_goal and not reward_success   # arrived but NOT counted as success by `reward >= 100`
    OUT["accounting"] = {"episodes": n, "reached_goal": n_goal, "ended_with_negative_battery": int(neg),
                         "goal_reached_with_battery<=0": int(succ_nonpos),
                         "goal_reached_but_not_counted_success_by_reward>=100_rule": int(reward_rule_miss)}


# ---------------------------------------------------------------- C. boundary bump vs obstacle collision
def audit_bump():
    env = L.RoverEnv(); env.grid[:] = 0; env.grid[0, 1] = 2  # rock under the rover
    env.position, env.battery, env.steps = [0, 1], 100, 0
    _, r_wall, _ = env.step(0)
    wall_cost = 100 - env.battery
    env.grid[1, 1] = 3; env.position, env.battery = [0, 1], 100
    _, r_obs, _ = env.step(1)
    OUT["bump"] = {"boundary_bump_on_rock_battery_cost": wall_cost, "boundary_bump_reward": r_wall,
                   "obstacle_collision_battery_cost": 100 - env.battery, "obstacle_collision_reward": r_obs}


# ---------------------------------------------------------------- D. privileged information in the observation
def audit_leak():
    env = L.RoverEnv(); env.grid[:] = 0
    env.position, env.battery, env.steps = [0, 0], 100, 0
    s1 = env.get_state().copy()
    env.grid[5, :] = 2  # rock band far outside the 3x3 patch
    s2 = env.get_state().copy()
    OUT["leak"] = {"features_changed_by_terrain_outside_3x3": [i for i in range(17) if not np.isclose(s1[i], s2[i])],
                   "interpretation": "indices 5 (cost_to_goal_ratio) and 6 (budget_margin) read the full hidden map"}


# ---------------------------------------------------------------- E. cost: Dijkstra calls and time per step
def audit_speed(n_steps=4000):
    calls = {"n": 0}
    orig = L.RoverEnv._dijkstra
    def counted(self, start):
        calls["n"] += 1
        return orig(self, start)
    L.RoverEnv._dijkstra = counted
    seed_all(2); env = L.RoverEnv(); env.reset(); calls["n"] = 0
    t0, steps = time.perf_counter(), 0
    rng = np.random.default_rng(0)
    while steps < n_steps:
        _, _, d = env.step(int(rng.integers(4))); steps += 1
        if d: env.reset()
    legacy_us = (time.perf_counter() - t0) / steps * 1e6
    per_step = calls["n"] / steps
    L.RoverEnv._dijkstra = orig
    new, _ = make_env(EnvConfig()); new.reset(seed=0)
    t0, steps = time.perf_counter(), 0
    while steps < n_steps:
        _, _, d, _ = new.step(int(rng.integers(4))); steps += 1
        if d: new.reset(seed=steps)
    new_us = (time.perf_counter() - t0) / steps * 1e6
    OUT["speed"] = {"legacy_dijkstra_calls_per_step": round(per_step, 2), "legacy_us_per_step": round(legacy_us, 1),
                    "audited_onboard_us_per_step": round(new_us, 1), "speedup_x": round(legacy_us / new_us, 1)}


# ---------------------------------------------------------------- F. seeding
def audit_seed():
    a = L.RoverEnv(); b = L.RoverEnv()
    unseeded_equal = bool((a.grid == b.grid).all())
    seed_all(5); a = L.RoverEnv(); seed_all(5); b = L.RoverEnv()
    OUT["seeding"] = {"two_unseeded_envs_same_map": unseeded_equal, "same_global_seed_same_map": bool((a.grid == b.grid).all()),
                      "note": "no seed argument or per-episode seed; map stream depends on global random state"}


# ---------------------------------------------------------------- G. static repo/notebook defects
def audit_static():
    nb = json.load(open(ROOT / "legacy" / "DQN_PPO_Agents.ipynb"))
    cells = [''.join(c["source"]) for c in nb["cells"]]
    first_use = next(i for i, c in enumerate(cells) if "MarsGridGenerator(" in c)
    defined = next(i for i, c in enumerate(cells) if "class MarsGridGenerator" in c)
    train_ppo = (ROOT / "legacy" / "train_ppo.py").read_text()
    try:
        (0.5).item()
        item_bug = False
    except AttributeError:
        item_bug = True
    dqn_placeholder = any('"steps":   [env.max_battery] * len(reward_history)' in c for c in cells)
    OUT["static"] = {
        "notebook_env_cell_uses_MarsGridGenerator_before_definition": first_use < defined,
        "notebook_cells_use_vs_define": [first_use, defined],
        "train_dqn.py_listed_in_README_but_missing": not (ROOT / "legacy" / "train_dqn.py").exists(),
        "train_ppo.py_calls_item_on_python_float": ("_, _, next_val = agent.act(state)" in train_ppo and "next_val.item()" in train_ppo and item_bug),
        "dqn_steps_saved_as_constant_placeholder": dqn_placeholder,
        "success_defined_as_reward>=100": all("reward >= 100" in c for c in (cells[4], cells[8])),
        "reported_success_is_training_time_epsilon_greedy": "agent.act(state)" in cells[4] and "epsilon" in cells[4],
    }


# ---------------------------------------------------------------- H. committed result files
def audit_saved_results():
    import pickle
    res = {}
    for name in ("dqn", "ppo"):
        r = pickle.load(open(ROOT / "legacy" / f"{name}_results.pkl", "rb"))
        s, b, st = np.array(r["success"]), np.array(r["battery"]), np.array(r["steps"])
        roll = np.array([s[max(0, i - 99):i + 1].mean() for i in range(len(s))])
        first80 = int(np.argmax(roll >= 0.8)) + 1 if (roll >= 0.8).any() else None
        res[name] = {"episodes": len(s), "SR_first100": float(s[:100].mean()), "SR_all": float(s.mean()),
                     "SR_last100": float(s[-100:].mean()), "battery_last100": float(b[-100:].mean()),
                     "episodes_with_negative_battery": int((b < 0).sum()), "min_battery": int(b.min()),
                     "episode_rolling_SR>=0.8": first80, "steps_unique_values": int(len(np.unique(st))),
                     "steps_mean": float(st.mean())}
    nb = json.load(open(ROOT / "legacy" / "DQN_PPO_Agents.ipynb"))
    log = {}
    for idx, key, pat in ((4, "dqn", r"Final success rate\s+\(last 100 ep\):\s*([\d.]+)"), (8, "ppo", r"Final SR \(last 100\):\s*([\d.]+)")):
        txt = "".join("".join(o.get("text", [])) for o in nb["cells"][idx].get("outputs", []))
        m = re.findall(pat, txt)
        log[key] = float(m[-1]) if m else None
    for k in res:
        res[k]["notebook_logged_SR_last100"] = log[k]
        res[k]["pkl_matches_notebook_log"] = log[k] is not None and abs(log[k] - res[k]["SR_last100"]) < 0.005
    OUT["saved_results"] = res


if __name__ == "__main__":
    for f in (audit_maps, audit_accounting, audit_bump, audit_leak, audit_speed, audit_seed, audit_static, audit_saved_results):
        f()
    out = ROOT / "results" / "audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "legacy_audit.json").write_text(json.dumps(OUT, indent=2, default=float))
    print(json.dumps(OUT, indent=2, default=float))
