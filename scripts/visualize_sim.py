"""Record, save and draw episodes of the Task-2 simulator.
   python scripts/visualize_sim.py --config configs/sim32_id.yaml --seed 20000 --out results/viz/id_20000
   python scripts/visualize_sim.py --replay results/viz/id_20000/episode.json --gif   # redraw a saved record
Writes episode.json (record), final.png (3-panel figure) and, with --gif, replay.gif."""
from _common import ROOT  # noqa: F401

import argparse
from pathlib import Path

from src.config import load_config
from src.evaluation.evaluator import make_env
from src.evaluation.replay import load_record, plot_episode, record_episode, save_record, save_replay_gif, trajectory_hash
from src.policies.sim32_heuristic import ReactiveExplorer

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sim32_id.yaml")
ap.add_argument("--seed", type=int, default=20000, help="map/episode seed (test maps are 20000+)")
ap.add_argument("--out", default="results/viz/episode")
ap.add_argument("--replay", help="path of a saved episode.json to redraw instead of simulating")
ap.add_argument("--gif", action="store_true")
a = ap.parse_args()

if a.replay:
    rec, out = load_record(a.replay), Path(a.replay).parent
else:
    cfg = load_config(ROOT / a.config)
    env, ev = make_env(cfg.env, cfg.run.seed)
    rec, out = record_episode(env, ev, ReactiveExplorer(seed=a.seed), a.seed), ROOT / a.out
    save_record(rec, out / "episode.json")
plot_episode(rec, out / "final.png")
if a.gif:
    save_replay_gif(rec, out / "replay.gif")
print(f"{rec['map_id']}: outcome={rec['reason']} steps={len(rec['steps'])} hash={trajectory_hash(rec)} -> {out}")
