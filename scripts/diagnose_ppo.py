"""Diagnose greedy-vs-sampled PPO.
   python scripts/diagnose_ppo.py --config configs/baseline_10x10_onboard.yaml --ckpt results/ref_onboard_ppo_s0/checkpoints/final.pt
Runs the same validation maps in both modes and prints success, end reasons, collisions/episode and the policy's
mean entropy / max-probability along the visited states."""
from _common import ROOT  # noqa: F401

import argparse

import numpy as np
import torch

from src.agents.ppo_agent import PPOAgent
from src.config import load_config
from src.evaluation.evaluator import make_env
from src.evaluation.runner import evaluate
from src.seeding import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

cfg = load_config(ROOT / args.config)
seed_everything(cfg.run.seed)
env, ev = make_env(cfg.env, seed=cfg.run.seed)
agent = PPOAgent(env.obs_dim, env.n_actions, cfg.ppo, seed=cfg.run.seed)
agent.load(args.ckpt)
seeds = range(cfg.run.val_seed_offset, cfg.run.val_seed_offset + args.n)


class Probe:
    def __init__(self, greedy):
        self.greedy, self.ent, self.pmax = greedy, [], []

    def act(self, obs):
        e, p = agent.policy_stats(obs)
        self.ent.append(e); self.pmax.append(p)
        return agent.act(obs, greedy=self.greedy)


print(f"{'mode':8} {'SR':>5} {'goal':>5} {'batt':>5} {'tmo':>4} {'coll/ep':>8} {'steps':>6} {'entropy':>8} {'pmax':>6}  (max entropy = {np.log(env.n_actions):.2f})")
for name, greedy in (("greedy", True), ("sampled", False)):
    pr = Probe(greedy)
    _, s = evaluate(env, ev, pr, seeds)
    r = s["reasons"]
    print(f"{name:8} {s['success_rate']:5.2f} {r['goal']:5d} {r['battery_depleted']:5d} {r['timeout']:4d} "
          f"{s['collisions_per_episode']:8.2f} {s['mean_steps']:6.1f} {np.mean(pr.ent):8.2f} {np.mean(pr.pmax):6.2f}")
