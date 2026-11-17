"""Privileged evaluator: the ONLY module (besides tests) allowed to read the hidden world.

Agents, policies, monitors and fallback planners must never import this module (enforced by
tests/test_no_leakage.py). It computes ground-truth quantities for metrics, replay and validation.
"""
from __future__ import annotations

import numpy as np

from src.config import EnvConfig
from src.environment import procgen as G
from src.environment import terrain as T
from src.environment.advanced_env import AdvancedRoverEnv
from src.environment.rover_env import RoverEnv


class Evaluator:
    def __init__(self, env: RoverEnv):
        self._env = env

    # -- ground-truth access --------------------------------------------------------------
    @property
    def grid(self) -> np.ndarray:
        return self._env._world.grid.copy()

    @property
    def start(self):
        return self._env._world.start

    @property
    def goal(self):
        return self._env._world.goal

    @property
    def optimal_cost(self) -> int:
        """Minimum battery needed from the start cell to the goal on this map."""
        return self._env._world.optimal_cost

    def cost_to_goal(self) -> int:
        e = self._env
        return int(T.dijkstra(e._world.grid, e.position)[e._world.goal])

    def budget_margin(self) -> float:
        """(battery - min cost to goal) / capacity. Negative => goal is no longer reachable. Evaluator-side only."""
        return (self._env.battery - self.cost_to_goal()) / self._env.max_battery

    def optimal_path(self):
        e = self._env
        return T.dijkstra_path(e._world.grid, e.position, e._world.goal)

    # -- map injection (tests, replays, hand-built scenarios) ------------------------------
    def reset_with_map(self, grid: np.ndarray, start=(0, 0), goal=None, map_id: str = "injected") -> np.ndarray:
        e = self._env
        grid = np.asarray(grid, dtype=np.int64).copy()
        goal = goal or (e.size - 1, e.size - 1)
        dist = T.dijkstra(grid, start)[goal]
        e._install_world(grid, start, goal, int(min(dist, T.INF)), map_id)
        return e._observe()


class AdvancedEvaluator(Evaluator):
    """Evaluator for the Task-2 env: same interface, but path costs use the TRUE energy table."""

    def __init__(self, env: AdvancedRoverEnv):
        self._env = env

    def cost_to_goal(self) -> int:
        e = self._env
        return int(G.dijkstra_lut(e._world.grid, e.position, e._lut)[e._world.goal])

    def optimal_path(self):
        e = self._env
        return G.dijkstra_path_lut(e._world.grid, e.position, e._world.goal, e._lut)

    def reset_with_map(self, grid: np.ndarray, start=(0, 0), goal=None, map_id: str = "injected") -> np.ndarray:
        e = self._env
        grid = np.asarray(grid, dtype=np.int64).copy()
        goal = goal or (e.size - 1, e.size - 1)
        dist = G.dijkstra_lut(grid, start, e._lut)[goal]
        e._install_world(grid, start, goal, int(min(dist, T.INF)), map_id)
        return e._observe()

    @property
    def true_slip(self):
        return self._env._slip

    @property
    def true_energy_table(self):
        return tuple(int(v) for v in self._env._lut[:3])


def make_env(cfg: EnvConfig | None = None, seed: int = 0):
    """Factory: hand `env` to agents, keep `evaluator` for metrics. Never pass the evaluator to an agent.
    cfg.kind == "advanced" -> Task-2 32x32 partially observable simulator; otherwise the Task-1 env."""
    if cfg is not None and getattr(cfg, "kind", "basic") == "advanced":
        env = AdvancedRoverEnv(cfg, seed)
        return env, AdvancedEvaluator(env)
    env = RoverEnv(cfg, seed)
    return env, Evaluator(env)
