"""Torch-free reference policies. They see ONLY the onboard observation vector (no env/evaluator access)."""
from __future__ import annotations

import numpy as np

# action -> index inside the flattened 3x3 patch (row-major, centre = 4)
_PATCH_IDX = {0: 1, 1: 7, 2: 3, 3: 5}
BLOCKED = 0.99


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(4))


class GreedyLocalPolicy:
    """Step toward the goal along the cheapest visible, unblocked cell; random unblocked move if stuck."""
    name = "greedy_local"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        off = {15: 6, 17: 8}[len(obs)]
        gdx, gdy = obs[2], obs[3]  # goal direction (+ => increasing x / y)
        patch = obs[off:off + 9]
        toward = ([1] if gdx > 0 else [0] if gdx < 0 else []) + ([3] if gdy > 0 else [2] if gdy < 0 else [])
        open_actions = [a for a in range(4) if patch[_PATCH_IDX[a]] < BLOCKED]
        good = [a for a in toward if a in open_actions]
        if good:
            return min(good, key=lambda a: (patch[_PATCH_IDX[a]], a))
        if open_actions:
            return int(self.rng.choice(open_actions))
        return int(self.rng.integers(4))
