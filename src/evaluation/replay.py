"""Episode recording / replay / map visualisation for the Task-2 simulator (evaluator side: may read ground truth).

record_episode() runs a policy and stores everything needed to redraw the episode WITHOUT re-simulating:
true grid, trajectory, events, the 5x5 readings the rover received, and explored-map snapshots.
Re-running the same (config, seed, policy) reproduces the record exactly (`trajectory_hash`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.environment.advanced_env import R, W

TERRAIN_COLORS = ["#e8dcc0", "#d9a441", "#7a6a5a", "#1b1b1b"]  # normal, sand, rock, obstacle
TERRAIN_NAMES = ["normal", "sand", "rock", "obstacle"]


def record_episode(env, ev, policy, seed: int, snap_every: int = 1) -> dict:
    obs = env.reset(seed=seed)
    rec = {"map_id": env.map_id, "seed": seed, "size": env.size, "grid": ev.grid.tolist(), "start": list(ev.start),
           "goal": list(ev.goal), "optimal_cost": ev.optimal_cost, "max_battery": env.max_battery,
           "steps": [], "snapshots": {}}

    def snap(t):
        """Belief map, one char per cell: U unknown | N normal | S sand | R rock | X believed obstacle."""
        m = env.explored_map()
        cls = np.clip(np.rint(np.nan_to_num(m.terrain_mean, nan=0.0) / 0.3).astype(int), 0, 2)
        ch = np.where(~m.known, "U", np.where(m.obstacle_frac > 0.5, "X", np.array(["N", "S", "R"])[cls]))
        rec["snapshots"][t] = ["".join(row) for row in ch]

    win0 = obs[9:9 + W * W].reshape(W, W)
    rec["initial"] = {"pos": list(env.position), "window": win0.round(3).tolist(), "battery": env.battery}
    snap(0)
    done, info = False, {}
    while not done:
        a = int(policy.act(obs))
        before = env.position
        obs, r, done, info = env.step(a)
        t = info["steps"]
        rec["steps"].append({"t": t, "from": list(before), "action": a, "to": list(env.position), "reward": round(float(r), 4),
                             "battery": info["battery"], "event": info["event"], "slipped": info["slipped"],
                             "energy_used": info["energy_used"], "window": obs[9:9 + W * W].reshape(W, W).round(3).tolist()})
        if t % snap_every == 0 or done:
            snap(t)
    rec["reason"], rec["success"] = info["reason"], int(info["reason"] == "goal")
    return rec


def trajectory_hash(rec: dict) -> str:
    key = json.dumps([[s["from"], s["action"], s["event"], s["battery"]] for s in rec["steps"]] + [rec["map_id"]])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def save_record(rec: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec))


def load_record(path) -> dict:
    rec = json.loads(Path(path).read_text())
    rec["snapshots"] = {int(k): v for k, v in rec["snapshots"].items()}
    return rec


def _snapshot_at(rec: dict, t: int) -> dict:
    ks = [k for k in rec["snapshots"] if k <= t]
    return rec["snapshots"][max(ks)]


def _pos_at(rec: dict, t: int):
    return tuple(rec["initial"]["pos"]) if t == 0 else tuple(rec["steps"][t - 1]["to"])


def plot_episode(rec: dict, out_path=None, t: int | None = None, fig=None):
    """3 panels at step t (default: last): ground truth + path | explored map (belief) | the 5x5 readings."""
    import matplotlib
    if fig is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    n_steps = len(rec["steps"])
    t = n_steps if t is None else max(0, min(t, n_steps))
    grid = np.array(rec["grid"])
    path = [tuple(rec["initial"]["pos"])] + [tuple(s["to"]) for s in rec["steps"][:t]]
    pos = _pos_at(rec, t)
    snap = _snapshot_at(rec, t)
    fig = fig or plt.figure(figsize=(13.5, 4.6), dpi=130)
    fig.clear()
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    cmap = ListedColormap(TERRAIN_COLORS)

    def frame(a, title):
        a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        a.add_patch(Rectangle((pos[1] - R - .5, pos[0] - R - .5), W, W, fill=False, ec="#d62728", lw=1.3))

    ax[0].imshow(grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax[0].plot([p[1] for p in path], [p[0] for p in path], "-", c="#1f77b4", lw=1.4)
    ax[0].plot(*rec["start"][::-1], "s", c="#2ca02c", ms=7); ax[0].plot(*rec["goal"][::-1], "*", c="#d62728", ms=11)
    ax[0].plot(pos[1], pos[0], "o", c="#d62728", ms=5)
    frame(ax[0], "Ground truth (evaluator only) + trajectory")

    ch = np.array([list(r) for r in snap])
    known = ch != "U"
    img = np.full(grid.shape + (3,), 0.78)  # unknown = grey
    for letter, rgb in (("N", (0.91, 0.86, 0.75)), ("S", (0.85, 0.64, 0.25)), ("R", (0.48, 0.42, 0.35)), ("X", (0.11, 0.11, 0.11))):
        img[ch == letter] = rgb
    ax[1].imshow(img, interpolation="nearest")
    ax[1].plot([p[1] for p in path], [p[0] for p in path], "-", c="#1f77b4", lw=1.0, alpha=.7)
    ax[1].plot(pos[1], pos[0], "o", c="#d62728", ms=5)
    frame(ax[1], f"Explored map (belief), {known.mean():.0%} known")

    win = np.array(rec["initial"]["window"] if t == 0 else rec["steps"][t - 1]["window"])
    ax[2].imshow(win, cmap="YlOrBr", vmin=0, vmax=1, interpolation="nearest")
    for i in range(W):
        for j in range(W):
            ax[2].text(j, i, f"{win[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if win[i, j] < .7 else "w")
    ax[2].plot(R, R, "o", mfc="none", mec="#d62728", mew=1.5, ms=26)
    ax[2].set_title("5x5 sensor readings (1.00 = obstacle seen)", fontsize=10); ax[2].set_xticks([]); ax[2].set_yticks([])
    batt = rec["initial"]["battery"] if t == 0 else rec["steps"][t - 1]["battery"]
    fig.suptitle(f"{rec['map_id']}  step {t}/{n_steps}  battery {batt}/{rec['max_battery']}  "
                 f"optimal cost {rec['optimal_cost']}  outcome: {rec['reason'] if t == n_steps else 'running'}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig


def save_replay_gif(rec: dict, out_path, stride: int = 3, fps: int = 8) -> None:
    """Frame-by-frame replay straight from the record (no re-simulation)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    frames = []
    fig = plt.figure(figsize=(13.5, 4.6), dpi=70)
    n = len(rec["steps"])
    for t in sorted(set(list(range(0, n + 1, stride)) + [n])):
        plot_episode(rec, None, t, fig)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=int(1000 / fps), loop=0)
