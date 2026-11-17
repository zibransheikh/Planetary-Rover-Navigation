"""Reactive onboard heuristic for the Task-2 simulator (smoke tests, visualisation, sanity reference).

Uses ONLY the policy-facing observation vector. Not the Task-3 battery-aware A* baseline.
"""
from __future__ import annotations

import numpy as np

from src.environment.advanced_env import FEATURES, R, W

# action -> (d_row, d_col); same convention as the env (0 up, 1 down, 2 left, 3 right). Defined here on purpose:
# policy modules must not import src.environment.terrain (it contains the full-map Dijkstra).
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_WIN0 = FEATURES.index(f"win_r{-R:+d}_c{-R:+d}")
_VIS0 = FEATURES.index(f"visits_r{-R:+d}_c{-R:+d}")


class ReactiveExplorer:
    name = "reactive_explorer"

    def __init__(self, seed: int = 0, terrain_w: float = 0.5, visit_w: float = 0.6):
        self.rng = np.random.default_rng(seed)
        self.terrain_w, self.visit_w = terrain_w, visit_w

    def act(self, obs) -> int:
        gdx, gdy = obs[2], obs[3]
        win = np.asarray(obs[_WIN0:_WIN0 + W * W]).reshape(W, W)
        vis = np.asarray(obs[_VIS0:_VIS0 + W * W]).reshape(W, W)
        best, best_s = 0, -np.inf
        for a, (dr, dc) in enumerate(DELTAS):
            w = win[R + dr, R + dc]
            if w >= 0.999:  # obstacle detected (or map edge)
                continue
            s = 2.0 * (dr * gdx + dc * gdy) - self.terrain_w * w - self.visit_w * vis[R + dr, R + dc]
            s += 1e-3 * self.rng.random()
            if s > best_s:
                best, best_s = a, s
        return best if best_s > -np.inf else int(self.rng.integers(4))
