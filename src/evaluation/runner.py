"""Episode runner + metrics. Correct-by-construction accounting: success == terminal reason 'goal'."""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.evaluator import Evaluator
from src.environment.rover_env import RoverEnv


def run_episode(env: RoverEnv, ev: Evaluator, policy, seed: int, logger=None, episode_id: int = 0,
                log_margin: bool = False) -> dict:
    obs = env.reset(seed=seed)
    opt = ev.optimal_cost
    total_r, energy, collisions, margins, info = 0.0, 0, 0, [], {}
    done = False
    while not done:
        a = int(policy.act(obs))
        pos_before = env.position
        obs, r, done, info = env.step(a)
        total_r += r
        energy += info["energy_used"]
        collisions += info["event"].endswith("collision")
        if log_margin:
            margins.append(ev.budget_margin())
        if logger is not None:
            logger.log(episode=episode_id, map=env.map_id, step=info["steps"], pos=pos_before, action=a,
                       reward=round(r, 4), battery=info["battery"], event=info["event"],
                       energy_used=info["energy_used"], reason=info["reason"])
    success = info["reason"] == "goal"
    return {
        "map": env.map_id, "success": int(success), "reason": info["reason"], "steps": info["steps"],
        "reward": total_r, "battery_left": env.battery, "energy_used": energy, "collisions": int(collisions),
        "optimal_cost": opt, "energy_ratio": (energy / opt) if success else float("nan"),
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    lo, hi = wilson_ci(ok, n)
    succ = [r for r in rows if r["success"]]
    reasons = {k: sum(r["reason"] == k for r in rows) for k in ("goal", "battery_depleted", "timeout")}
    return {
        "episodes": n, "success_rate": ok / n, "success_ci95": [lo, hi], "reasons": reasons,
        "mean_battery_left": float(np.mean([r["battery_left"] for r in rows])),
        "mean_battery_left_success": float(np.mean([r["battery_left"] for r in succ])) if succ else float("nan"),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "collisions_per_episode": float(np.mean([r["collisions"] for r in rows])),
        "mean_energy_ratio_success": float(np.nanmean([r["energy_ratio"] for r in succ])) if succ else float("nan"),
        "min_battery_seen": int(min(r["battery_left"] for r in rows)),
    }


def evaluate(env: RoverEnv, ev: Evaluator, policy, seeds, logger=None, log_margin=False) -> tuple[list[dict], dict]:
    rows = [run_episode(env, ev, policy, s, logger, i, log_margin) for i, s in enumerate(seeds)]
    return rows, summarise(rows)
