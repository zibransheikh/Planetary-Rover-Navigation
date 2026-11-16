"""Reproducibility metadata written next to every result."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def run_meta(config_dict: dict) -> dict:
    meta = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "config": config_dict,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        meta["torch"] = None
    return meta


def write_meta(out_dir: str | Path, config_dict: dict) -> None:
    import json

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "run_meta.json").write_text(json.dumps(run_meta(config_dict), indent=2, default=str))
