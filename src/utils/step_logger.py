"""JSONL decision log (one object per env step). Schema is a strict subset of the Section 14.3 log."""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


class StepLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w")

    def log(self, **record) -> None:
        record["schema"] = SCHEMA_VERSION
        self._f.write(json.dumps(record, separators=(",", ":"), default=float) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
