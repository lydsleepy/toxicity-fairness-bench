"""
Result caching to avoid redundant API calls.

Results are stored as Parquet files. On re-run, cached results are
loaded instead of re-calling the API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class ResultCache:
    def __init__(self, cache_dir: str | Path = "results/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, dataset: str, model: str, sample: int | None) -> str:
        payload = json.dumps(
            {"dataset": dataset, "model": model, "sample": sample},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path(key), index=False)

    def load(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(key))

    def clear(self, key: str | None = None) -> None:
        if key:
            p = self.path(key)
            if p.exists():
                p.unlink()
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
