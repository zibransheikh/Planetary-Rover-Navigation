"""Torch-free, fully reproducible 10x10 reference run (random + onboard-greedy policies) on the audited env.

Evaluates on fixed validation (10_000+) and test (20_000+) map seeds, writes metrics + a decision log, and
proves reproducibility by running everything twice and comparing a SHA-256 of all trajectories.
    python scripts/run_reference.py            (~10 s)
"""
from _common import ROOT, parse  # noqa: F401

import hashlib
import json

from src.config import load_config
from src.evaluation.evaluator import make_env
from src.evaluation.runner import evaluate
from src.policies.baseline_policies import GreedyLocalPolicy, RandomPolicy
from src.utils.run_meta import write_meta
from src.utils.step_logger import StepLogger

args, ov = parse("Reference run")
cfg = load_config(ROOT / args.config, ov)
out = ROOT / "results" / "reference_10x10"
N = 200


def one_pass(log_path=None):
    env, ev = make_env(cfg.env, seed=cfg.run.seed)
    res, digest = {}, hashlib.sha256()
    for split, off in (("val", cfg.run.val_seed_offset), ("test", cfg.run.test_seed_offset)):
        for P in (RandomPolicy, GreedyLocalPolicy):
            logger = StepLogger(log_path) if (log_path and split == "test" and P is GreedyLocalPolicy) else None
            rows, summ = evaluate(env, ev, P(cfg.run.seed), range(off, off + N), logger=logger)
            if logger:
                logger.close()
            res[f"{P.name}/{split}"] = summ
            digest.update(repr([(r["map"], r["steps"], round(r["reward"], 6), r["battery_left"], r["reason"]) for r in rows]).encode())
    return res, digest.hexdigest()


res1, h1 = one_pass(out / "greedy_local_test_decisions.jsonl")
res2, h2 = one_pass()
assert h1 == h2, "reference run is NOT reproducible"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"obs_mode": cfg.env.obs_mode, "episodes_per_split": N, "trajectory_sha256": h1,
                                              "reproduced_identically_on_second_run": h1 == h2, "results": res1}, indent=2, default=float))
write_meta(out, cfg.to_dict())
print(f"trajectory sha256: {h1[:16]}...  (identical on second run: {h1 == h2})")
for k, s in res1.items():
    lo, hi = s["success_ci95"]
    print(f"{k:22s} success {s['success_rate']:.3f} [{lo:.3f},{hi:.3f}]  battery_left(succ) {s['mean_battery_left_success']:.1f}  "
          f"collisions/ep {s['collisions_per_episode']:.2f}  reasons {s['reasons']}")
