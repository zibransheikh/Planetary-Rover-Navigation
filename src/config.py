"""Typed configuration loaded from YAML. One file fully describes a run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    size: int = 10
    battery: int = 100
    # terrain probabilities (obstacle probability is the remainder)
    p_normal: float = 0.55
    p_sand: float = 0.15
    p_rock: float = 0.15
    # "onboard": 15-d, policy-facing, no full-map information
    # "legacy17": original 17-d vector incl. Dijkstra cost-to-goal/budget-margin (reproduction/ablation ONLY)
    obs_mode: str = "onboard"
    # per-step reward term that uses full-map Dijkstra (training-signal only, never observed)
    reward_privileged_shaping: bool = False
    max_steps: int | None = None  # default: battery (cannot trigger, every action costs >= 1)
    map_retry_limit: int = 1000
    ensure_energy_feasible: bool = True


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.992  # per episode (as in original)
    buffer_size: int = 50_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    target_update_freq: int = 500
    grad_clip: float = 1.0
    hidden: tuple[int, ...] = (256, 256, 128)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    minibatch: int = 64
    rollout_steps: int = 512
    grad_clip: float = 0.5
    hidden: tuple[int, ...] = (256, 256)


@dataclass
class RunConfig:
    seed: int = 0
    episodes: int = 1500
    eval_every: int = 100        # greedy evaluation on fixed validation maps
    eval_episodes: int = 50
    val_seed_offset: int = 10_000  # validation maps: seeds 10_000...
    test_seed_offset: int = 20_000  # test maps: seeds 20_000... (never used for selection)
    torch_threads: int = 1       # tiny MLPs are fastest on 1 thread; avoids oversubscription
    log_margin: bool = False     # evaluator-side budget margin each step (costs one Dijkstra/step)
    out_dir: str = "results/run"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fill(cls, data: dict | None):
    data = dict(data or {})
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    if "hidden" in data:
        data["hidden"] = tuple(data["hidden"])
    return cls(**data)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"env", "dqn", "ppo", "run"}
    if unknown:
        raise KeyError(f"Unknown top-level config sections: {sorted(unknown)}")
    cfg = Config(
        env=_fill(EnvConfig, raw.get("env")),
        dqn=_fill(DQNConfig, raw.get("dqn")),
        ppo=_fill(PPOConfig, raw.get("ppo")),
        run=_fill(RunConfig, raw.get("run")),
    )
    for dotted, value in (overrides or {}).items():  # e.g. {"run.seed": 3}
        section, key = dotted.split(".")
        setattr(getattr(cfg, section), key, value)
    return cfg
