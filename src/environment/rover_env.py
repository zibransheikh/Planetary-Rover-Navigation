"""Policy-facing battery-aware rover environment (audited version of the original 10x10 env).

Design rules enforced here (Task 1):
  * Ground truth (`_World`) is private. Agents only receive `obs` vectors and a small, non-privileged `info`.
  * The ONLY map information in an onboard observation is the 3x3 local terrain patch around the rover.
  * Full-map quantities (optimal cost, Dijkstra cost-to-goal, budget margin) live in `evaluation.evaluator`.
  * Every episode ends with an explicit reason: goal | battery_depleted | timeout. Success == reason "goal".
  * Battery is never negative. Randomness is only drawn from per-episode seeded generators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from src.seeding import episode_rng

ONBOARD_FEATURES = (
    "x_norm", "y_norm", "goal_dx", "goal_dy", "battery_ratio", "steps_ratio",
) + tuple(f"patch_{i}{j}" for i in "mzp" for j in "mzp")  # 3x3 patch, rows (-1,0,+1) x cols (-1,0,+1)
LEGACY_FEATURES = (
    "x_norm", "y_norm", "goal_dx", "goal_dy", "battery_ratio", "cost_to_goal_ratio[PRIVILEGED]",
    "budget_margin[PRIVILEGED]", "steps_ratio",
) + ONBOARD_FEATURES[6:]


@dataclass
class _World:
    """Hidden ground truth. Not exposed through the policy-facing API."""
    grid: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    optimal_cost: int
    patch_pad: np.ndarray  # padded terrain->patch-value grid (border = 1.0, i.e. "blocked")
    map_id: str


class RoverEnv:
    n_actions = 4

    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        if self.cfg.obs_mode not in ("onboard", "legacy17"):
            raise ValueError(f"obs_mode must be 'onboard' or 'legacy17', got {self.cfg.obs_mode!r}")
        self.size = self.cfg.size
        self.max_battery = self.cfg.battery
        self.max_steps = self.cfg.max_steps or self.cfg.battery
        self.obs_dim = 17 if self.cfg.obs_mode == "legacy17" else 15
        self.feature_names = LEGACY_FEATURES if self.cfg.obs_mode == "legacy17" else ONBOARD_FEATURES
        self._base_seed = seed
        self._episode_index = 0
        self._world: _World | None = None
        self.position = (0, 0)
        self.battery = self.max_battery
        self.steps = 0
        self._done = True

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed: int | None = None) -> np.ndarray:
        """New episode. `seed=None` -> next map of this env's reproducible stream; int -> that exact map."""
        if seed is None:
            rng = episode_rng(self._base_seed, self._episode_index)
            map_id = f"stream({self._base_seed},{self._episode_index})"
            self._episode_index += 1
        else:
            rng = np.random.default_rng(seed)
            map_id = f"seed({seed})"
        start, goal = (0, 0), (self.size - 1, self.size - 1)
        c = self.cfg
        for _ in range(c.map_retry_limit):
            grid = T.generate_grid(rng, self.size, c.p_normal, c.p_sand, c.p_rock)
            T.clear_endpoints(grid, start, goal)
            cost = T.dijkstra(grid, start)[goal]
            if cost >= T.INF:
                continue  # unreachable goal
            if c.ensure_energy_feasible and cost > self.max_battery:
                continue  # reachable but impossible within battery capacity
            break
        else:
            raise RuntimeError(f"No valid map after {c.map_retry_limit} attempts; check terrain probabilities")
        self._install_world(grid, start, goal, int(cost), map_id)
        return self._observe()

    def _install_world(self, grid, start, goal, optimal_cost, map_id) -> None:
        pad = np.pad(T.PATCH_VALUE[grid], 1, constant_values=1.0)
        self._world = _World(grid.copy(), tuple(start), tuple(goal), int(optimal_cost), pad, map_id)
        self.position = tuple(start)
        self.battery = self.max_battery
        self.steps = 0
        self._done = False

    @property
    def map_id(self) -> str:
        """Seed label of the current map (reproducibility metadata, carries no terrain information)."""
        return self._world.map_id

    # ------------------------------------------------------------------ observation
    def _observe(self) -> np.ndarray:
        w, n = self._world, self.size
        x, y = self.position
        base = [x / (n - 1), y / (n - 1), (w.goal[0] - x) / (n - 1), (w.goal[1] - y) / (n - 1)]
        batt = self.battery / self.max_battery
        steps = self.steps / self.max_battery
        patch = w.patch_pad[x:x + 3, y:y + 3].ravel()  # row-major == original (di outer, dj inner)
        if self.cfg.obs_mode == "legacy17":  # PRIVILEGED features, reproduction/ablation only
            ctg = T.dijkstra(w.grid, (x, y))[w.goal[0], w.goal[1]]
            ctg = self.max_battery if ctg >= T.INF else int(ctg)
            vec = base + [batt, ctg / self.max_battery, (self.battery - ctg) / self.max_battery, steps]
        else:
            vec = base + [batt, steps]
        return np.concatenate([np.asarray(vec, dtype=float), patch.astype(float)])

    # ------------------------------------------------------------------ dynamics
    def step(self, action: int):
        """Returns (obs, reward, done, info). `info['terminated']` / `info['truncated']` separate the two kinds of done."""
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        w, n = self._world, self.size
        self.steps += 1
        x, y = self.position
        dx, dy = T.DELTAS[int(action)]
        nx, ny = x + dx, y + dy
        info = {"event": "move", "reason": None, "terminated": False, "truncated": False, "energy_used": 0}

        out_of_bounds = not (0 <= nx < n and 0 <= ny < n)
        if out_of_bounds or w.grid[nx, ny] == T.OBSTACLE:
            # collision (obstacle OR map boundary, treated identically): no motion, 1 battery unit
            info["event"] = "boundary_collision" if out_of_bounds else "obstacle_collision"
            used = min(1, self.battery)
            self.battery -= used
            info["energy_used"] = used
            if self.battery <= 0:
                return self._finish(-10.0, "battery_depleted", info)
            return self._continue(-2.0, info)

        terrain = int(w.grid[nx, ny])
        cost = T.ENERGY[terrain]
        if cost > self.battery:
            # not enough energy to complete the move: rover stalls where it is, battery is spent
            info["event"] = "stalled"
            info["energy_used"] = self.battery
            self.battery = 0
            return self._finish(-20.0, "battery_depleted", info)

        self.position = (nx, ny)
        self.battery -= cost
        info["energy_used"] = cost

        if (nx, ny) == w.goal:
            efficiency = min(self.battery / w.optimal_cost, 1.0)
            return self._finish(100.0 + 50.0 * efficiency, "goal", info)
        if self.battery == 0:
            return self._finish(-20.0, "battery_depleted", info)

        old_d = abs(x - w.goal[0]) + abs(y - w.goal[1])
        new_d = abs(nx - w.goal[0]) + abs(ny - w.goal[1])
        reward = (old_d - new_d) * 0.5 - 0.05
        if terrain == T.SAND:
            reward -= 0.05
        elif terrain == T.ROCK:
            reward -= 0.1
        if self.cfg.reward_privileged_shaping:  # full-map Dijkstra in the REWARD only (never observed)
            ctg = T.dijkstra(w.grid, (nx, ny))[w.goal[0], w.goal[1]]
            margin = (self.battery - ctg) / self.max_battery
            if margin < 0:
                reward += 0.5 * margin
            elif margin < 0.1:
                reward -= 0.2
        return self._continue(reward, info)

    def _continue(self, reward, info):
        if self.steps >= self.max_steps:
            info["truncated"], info["reason"] = True, "timeout"
            self._done = True
        return self._observe(), reward, self._done, self._stamp(info)

    def _finish(self, reward, reason, info):
        info["terminated"], info["reason"] = True, reason
        self._done = True
        return self._observe(), reward, True, self._stamp(info)

    def _stamp(self, info):
        info["steps"], info["battery"] = self.steps, self.battery  # both are onboard-observable quantities
        return info
