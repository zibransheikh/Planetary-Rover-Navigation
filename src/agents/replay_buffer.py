"""Fixed-size numpy ring buffer. O(batch) sampling (the original deque + random.sample is O(n) per index)."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)  # 1.0 only for true termination (not time-limit truncation)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminated) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(terminated)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch: int):
        idx = self.rng.choice(self.size, batch, replace=False)  # same semantics as the original random.sample
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]
