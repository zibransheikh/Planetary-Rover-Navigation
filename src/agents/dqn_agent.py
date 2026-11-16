"""Double DQN (same algorithm/hyper-parameters as the original). Sees observations only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.replay_buffer import ReplayBuffer
from src.config import DQNConfig


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None, seed: int = 0):
        self.cfg = cfg or DQNConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.epsilon = self.cfg.eps_start
        self.rng = np.random.default_rng(seed)
        self.model = mlp(obs_dim, self.cfg.hidden, n_actions)
        self.target = mlp(obs_dim, self.cfg.hidden, n_actions)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-4)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed)
        self.train_step = 0

    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        return self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    def act(self, obs, greedy: bool = False) -> int:
        if not greedy and self.rng.random() <= self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def remember(self, s, a, r, s2, terminated) -> None:
        self.buffer.add(s, a, r, s2, terminated)

    def learn(self) -> None:
        bs = self.cfg.batch_size
        if len(self.buffer) < bs:
            return
        s, a, r, s2, d = (torch.from_numpy(x) for x in self.buffer.sample(bs))
        with torch.no_grad():  # Double DQN target
            best = self.model(s2).argmax(1, keepdim=True)
            next_q = self.target(s2).gather(1, best).squeeze(1)
            target_q = r + self.cfg.gamma * next_q * (1.0 - d)
        current_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

    def decay_epsilon(self) -> None:
        if self.epsilon > self.cfg.eps_min:
            self.epsilon *= self.cfg.eps_decay

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ck["model"]); self.target.load_state_dict(ck["model"]); self.epsilon = ck["epsilon"]
