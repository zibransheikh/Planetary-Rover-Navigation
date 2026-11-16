import importlib.util
from pathlib import Path

import numpy as np

from src.config import EnvConfig
from src.environment import terrain as T
from src.evaluation.evaluator import make_env

ROOT = Path(__file__).resolve().parents[1]


def flat_grid(n=10, fill=T.NORMAL):
    return np.full((n, n), fill, dtype=np.int64)


def fresh(cfg=None, seed=0):
    return make_env(cfg or EnvConfig(), seed)


def load_legacy_env_module():
    """Import the ORIGINAL, untouched rover_env.py from legacy/ (needs only numpy)."""
    spec = importlib.util.spec_from_file_location("legacy_rover_env", ROOT / "legacy" / "rover_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
