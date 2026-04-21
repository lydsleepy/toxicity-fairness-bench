from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_PARQUET = Path(__file__).parent.parent / "results" / "raw_results.parquet"


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    return pd.read_parquet(_PARQUET)


def df_available() -> bool:
    return _PARQUET.exists()
