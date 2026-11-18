"""PPO wiring for Task 3: action preferences, shield integration, harness construction, best-on-validation checkpoint,
and the run_task3.py --algo ppo path (train -> eval)."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from src.config import Config, EnvConfig, PPOConfig, load_config
from src.evaluation.evaluator import make_env

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def cfg32(**kw):
    base = dict(kind="advanced", size=32, battery=250, p_normal=.60, p_sand=.15, p_rock=.10)
    base.update(kw)
    return EnvConfig(**base)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPPOWiring(unittest.TestCase):
    def _agent(self):
        from src.agents.ppo_agent import PPOAgent
        env, ev = make_env(cfg32(), 0)
        return PPOAgent(env.obs_dim, env.n_actions, PPOConfig(hidden=(16,)), seed=0), env, ev

    def test_q_values_are_the_policy_logits(self):
        import torch
        agent, env, _ = self._agent()
        obs = env.reset(seed=1)
        q = agent.q_values(obs)
        self.assertEqual(q.shape, (4,))
        with torch.no_grad():
            ref = agent.model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
        np.testing.assert_allclose(q, ref)
        self.assertEqual(int(np.argmax(q)), agent.act(obs, greedy=True))

    def test_shield_around_ppo_runs_and_only_overrides_with_its_best_safe_logit(self):
        from src.evaluation.harness import GreedyAgentPolicy, evaluate_method
        from src.policies.rule_shield import RuleShield
        agent, env, ev = self._agent()
        sh = RuleShield(GreedyAgentPolicy(agent, "ppo"))
        _, s = evaluate_method(sh, ROOT / "configs/sim32_id.yaml", range(20000, 20003), "ppo_shield", "id")
        self.assertEqual(s["episodes"], 3)
        self.assertGreater(sh.stats["decisions"], 0)

    def test_build_policy_roundtrip(self):
        from src.evaluation.harness import build_policy
        agent, env, _ = self._agent()
        cfg = replace(load_config(ROOT / "configs/sim32_id.yaml"), ppo=PPOConfig(hidden=(16,)))
        with tempfile.TemporaryDirectory() as d:
            agent.save(Path(d) / "c.pt")
            plain = build_policy("ppo", cfg, str(Path(d) / "c.pt"))
            shielded = build_policy("ppo_shield", cfg, str(Path(d) / "c.pt"))
        obs = env.reset(seed=2)
        self.assertEqual(plain.act(obs), agent.act(obs, greedy=True))
        self.assertTrue(hasattr(shielded, "attach") and hasattr(shielded.base, "q_values"))

    def test_train_ppo_saves_best_val_and_final(self):
        from src.agents.ppo_agent import PPOAgent
        from src.training.loops import train_ppo
        cfg = Config(env=cfg32(), ppo=PPOConfig(hidden=(16,), rollout_steps=128),
                     run=replace(Config().run, episodes=4, eval_every=2, eval_episodes=2))
        env, _ = make_env(cfg.env, 0)
        agent = PPOAgent(env.obs_dim, env.n_actions, cfg.ppo, seed=0)
        with tempfile.TemporaryDirectory() as d:
            s = train_ppo(agent, cfg, d)
            self.assertTrue((Path(d) / "checkpoints/best_val.pt").exists())
            self.assertTrue((Path(d) / "checkpoints/final.pt").exists())
        self.assertIn("sampled_val_200", s)   # PPO also reports the sampled policy (Task-1 diagnostic)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRunnerPPOPath(unittest.TestCase):
    def test_train_then_eval_through_run_task3(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = yaml.safe_load((ROOT / "configs/sim32_id.yaml").read_text())
            cfg["ppo"] = {"hidden": [16], "rollout_steps": 128, "reward_scale": 0.02}
            cfg["run"].update(episodes=3, eval_every=3, eval_episodes=2)
            cfg_path = Path(d) / "tiny_ppo.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/run_task3.py"), *a, "--out", str(Path(d) / "res")],  # noqa: E731
                                            capture_output=True, text=True, cwd=ROOT)
            r = run("train", "--algo", "ppo", "--config", str(cfg_path), "--seeds", "0", "--jobs", "1")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((Path(d) / "res/train/ppo_s0/checkpoints/best_val.pt").exists(), r.stdout)
            r = run("eval", "--config", str(cfg_path), "--seeds", "0", "--jobs", "2", "--split", "val", "--n-maps", "2",
                    "--methods", "astar,ppo,ppo_shield")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            files = sorted(p.name for p in (Path(d) / "res/eval/val/id").glob("*.json"))
            self.assertEqual(files, ["astar.json", "ppo_s0.json", "ppo_shield_s0.json"])
            r = run("report", "--split", "val", "--config", str(cfg_path))
            self.assertIn("ppo_shield", r.stdout)
            self.assertIn("ppo ", r.stdout.replace("|", " "))
            r = run("curves", "--algo", "ppo", "--seeds", "0")
            self.assertIn("PPO: greedy validation success", r.stdout)


if __name__ == "__main__":
    unittest.main()
