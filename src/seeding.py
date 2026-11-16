"""Single entry point for seeding so that every run is reproducible."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch. Call once at program start."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # small MLPs on CPU are deterministic anyway
    except ImportError:  # core pipeline (env/tests/audit) must work without torch
        pass


def episode_rng(base_seed: int, episode_index: int) -> np.random.Generator:
    """Independent, reproducible RNG stream for one episode / map."""
    return np.random.default_rng(np.random.SeedSequence([base_seed, episode_index]))
