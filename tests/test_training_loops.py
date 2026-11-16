"""Loop mechanics verified with torch-free stand-in agents (the real agents need torch)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import Config
from src.policies.baseline_policies import GreedyLocalPolicy
from src.training.loops import train_dqn, train_ppo


class FakeDQN:
    epsilon = 1.0

    def __init__(self):
        self.p, self.remembered, self.learned, self.decays, self.terminal_flags = GreedyLocalPolicy(0), 0, 0, 0, set()

    def act(self, obs, greedy=False): return self.p.act(obs)
    def remember(self, s, a, r, s2, terminated): self.remembered += 1; self.terminal_flags.add(bool(terminated))
    def learn(self): self.learned += 1
    def decay_epsilon(self): self.decays += 1


class FakePPO:
    def __init__(self):
        self.p, self.buf, self.updates, self.last_values = GreedyLocalPolicy(0), 0, 0, []

    def sample(self, obs): return self.p.act(obs), -1.0, 0.0
    def act(self, obs, greedy=True): return self.p.act(obs)
    def value(self, obs): return 0.0
    def store(self, *a): self.buf += 1
    def update(self, last_value):
        assert isinstance(last_value, float)
        self.updates += 1; self.last_values.append(last_value); self.buf = 0


def small_cfg(tmp):
    cfg = Config()
    cfg.run.episodes, cfg.run.eval_every, cfg.run.eval_episodes, cfg.run.out_dir = 40, 10, 5, tmp
    cfg.dqn.warmup_steps, cfg.ppo.rollout_steps = 50, 64
    return cfg


class TestLoops(unittest.TestCase):
    def check_outputs(self, out: Path, episodes: int):
        rows = list(csv.DictReader((out / "episodes.csv").read_text().splitlines()))
        self.assertEqual(len(rows), episodes)
        for r in rows:
            self.assertGreaterEqual(int(r["battery"]), 0)
            self.assertEqual(r["success"] == "1", r["reason"] == "goal")
            self.assertGreaterEqual(int(r["steps"]), 1)
        self.assertEqual(len((out / "eval_greedy_val.csv").read_text().strip().splitlines()) - 1, episodes // 10)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["episodes"], episodes)
        self.assertEqual(s["greedy_val_200"]["episodes"], 200)
        self.assertGreaterEqual(s["train_time_min_battery"], 0)
        return rows, s

    def test_dqn_loop(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, agent = small_cfg(d), FakeDQN()
            train_dqn(agent, cfg, d)
            rows, s = self.check_outputs(Path(d), 40)
            self.assertEqual(agent.decays, 40)                      # epsilon decays once per episode
            self.assertEqual(agent.learned, s["env_steps"])         # one learn() per env step
            self.assertEqual(agent.remembered, 50 + s["env_steps"])  # warm-up + training transitions
            self.assertEqual(agent.terminal_flags, {True, False})

    def test_ppo_loop(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, agent = small_cfg(d), FakePPO()
            train_ppo(agent, cfg, d)
            rows, s = self.check_outputs(Path(d), 40)
            self.assertGreaterEqual(agent.updates, s["env_steps"] // 64)
            self.assertEqual(agent.buf, 0)                           # final partial rollout was flushed

    def test_loops_are_reproducible(self):
        outs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                train_dqn(FakeDQN(), small_cfg(d), d)
                outs.append((Path(d) / "episodes.csv").read_text())
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
