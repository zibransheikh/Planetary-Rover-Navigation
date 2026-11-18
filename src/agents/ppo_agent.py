"""PPO actor-critic (same algorithm/hyper-parameters as the original; value-return handling fixed)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.config import PPOConfig


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...]):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(d, n_actions)
        self.critic = nn.Linear(d, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.shared(x)
        return self.actor(z), self.critic(z)


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig | None = None, seed: int = 0):
        self.cfg = cfg or PPOConfig()
        self.model = ActorCritic(obs_dim, n_actions, self.cfg.hidden)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self._reset_buffer()

    def _reset_buffer(self):
        self.obs, self.actions, self.logps, self.rewards, self.values, self.dones = [], [], [], [], [], []

    @torch.no_grad()
    def sample(self, obs) -> tuple[int, float, float]:
        """Training action: returns (action, log_prob, value) as plain python numbers."""
        logits, value = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def act(self, obs, greedy: bool = True) -> int:
        """Evaluation action (greedy by default) so PPO satisfies the same `act(obs)->int` protocol as DQN."""
        logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(logits.argmax(-1).item()) if greedy else int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        """Action preferences (policy logits). Same name as DQN's so `RuleShield` can pick the best SAFE action."""
        logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return logits.squeeze(0).numpy()

    @torch.no_grad()
    def policy_stats(self, obs) -> tuple[float, float]:
        """(entropy in nats, max action probability) of the policy at `obs`; used by scripts/diagnose_ppo.py."""
        logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        dist = Categorical(logits=logits)
        return float(dist.entropy().item()), float(dist.probs.max().item())

    @torch.no_grad()
    def value(self, obs) -> float:
        return float(self.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[1].item())

    def store(self, obs, action, logp, reward, value, done) -> None:
        self.obs.append(obs); self.actions.append(action); self.logps.append(logp)
        self.rewards.append(reward); self.values.append(value); self.dones.append(float(done))

    def _gae(self, last_value: float):
        c, vals = self.cfg, self.values + [last_value]
        gae, returns = 0.0, [0.0] * len(self.rewards)
        for t in reversed(range(len(self.rewards))):
            nonterminal = 1.0 - self.dones[t]
            delta = c.reward_scale * self.rewards[t] + c.gamma * vals[t + 1] * nonterminal - vals[t]
            gae = delta + c.gamma * c.lam * nonterminal * gae
            returns[t] = gae + vals[t]
        returns = np.asarray(returns, dtype=np.float32)
        adv = returns - np.asarray(self.values, dtype=np.float32)
        return returns, (adv - adv.mean()) / (adv.std() + 1e-8)

    def update(self, last_value: float) -> None:
        """`last_value` must be a python float (0.0 if the rollout ended on a terminal state)."""
        if not self.obs:
            return
        c = self.cfg
        returns, adv = self._gae(last_value)
        obs = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32)
        act = torch.as_tensor(self.actions, dtype=torch.int64)
        old_lp = torch.as_tensor(self.logps, dtype=torch.float32)
        ret, adv = torch.from_numpy(returns), torch.from_numpy(adv.astype(np.float32))
        n = len(self.obs)
        for _ in range(c.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, c.minibatch):
                idx = torch.as_tensor(perm[i:i + c.minibatch])
                logits, v = self.model(obs[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(act[idx]) - old_lp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv[idx]
                loss = (-torch.min(s1, s2).mean()
                        + c.value_coef * (ret[idx] - v.squeeze(-1)).pow(2).mean()
                        - c.entropy_coef * dist.entropy().mean())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                self.optimizer.step()
        self._reset_buffer()

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu")["model"])
