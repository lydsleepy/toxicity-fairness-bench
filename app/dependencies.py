from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_DATASETS = {
    "primary": _RESULTS_DIR / "raw_results.parquet",
    "cohere_500": _RESULTS_DIR / "cohere_500" / "raw_results.parquet",
}

@lru_cache(maxsize=len(_DATASETS))
def load_df(dataset: str = "primary") -> pd.DataFrame:
    return pd.read_parquet(_DATASETS[dataset])


def df_available(dataset: str = "primary") -> bool:
    path = _DATASETS.get(dataset)
    return path is not None and path.exists()
