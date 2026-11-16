import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/baseline_10x10_onboard.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--out", default=None, help="output dir (default: run.out_dir from config + _<algo>_s<seed>)")
    a = p.parse_args()
    ov = {k: v for k, v in {"run.seed": a.seed, "run.episodes": a.episodes, "run.eval_every": a.eval_every,
                            "run.eval_episodes": a.eval_episodes}.items() if v is not None}
    return a, ov
