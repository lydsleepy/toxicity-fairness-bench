"""
Dataset loaders for toxicity benchmarking.

Both loaders return a standardized DataFrame with columns:
  id, text, actual_label, protected_attribute, attribute_value
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def load_jigsaw(
    csv_path: str | Path,
    sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the Jigsaw toxic comment dataset.

    Download from:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    Place train.csv in data/raw/jigsaw/
    """
    df = pd.read_csv(csv_path)

    df["actual_label"] = (df["toxic"] >= 0.5).map(
        {True: "toxic", False: "non-toxic"}
    )
    df = df.rename(columns={"comment_text": "text"})
    df["id"] = df["id"].astype(str)
    df["protected_attribute"] = "none"
    df["attribute_value"] = "unknown"

    gender_map = {"male": "Male", "female": "Female"}
    for col, val in gender_map.items():
        if col in df.columns:
            mask = df[col] >= 0.5
            df.loc[mask, "protected_attribute"] = "Gender"
            df.loc[mask, "attribute_value"] = val

    result = df[
        ["id", "text", "actual_label", "protected_attribute", "attribute_value"]
    ]

    if sample:
        result = result.sample(n=min(sample, len(result)), random_state=seed)

    return result.reset_index(drop=True)


def load_hatexplain(
    sample: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load HateXplain from HuggingFace hub (downloads automatically).

    Majority vote across annotators determines the final label:
      - "hatespeech" or "offensive" → "toxic"
      - "normal" → "non-toxic"
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    ds = load_dataset(
        "hatexplain",
        split="train",
        trust_remote_code=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    df = ds.to_pandas()

    # ClassLabel features may be decoded as ints (0=hatespeech, 1=normal,
    # 2=offensive) or strings depending on the datasets library version.
    # numpy 2.0 removed np.int64's inheritance from Python int, so
    # isinstance(v, int) is False for numpy scalars — use str check instead.
    _int_to_label: dict[int, str] = {0: "hatespeech", 1: "normal", 2: "offensive"}

    rows = []
    for _, row in df.iterrows():
        raw_votes = row["annotators"]["label"]
        label_votes: list[str] = [
            v if isinstance(v, str) else _int_to_label[int(v)]
            for v in raw_votes
        ]
        majority = max(set(label_votes), key=label_votes.count)
        actual = "non-toxic" if majority == "normal" else "toxic"

        targets: list[list[str]] = row["annotators"].get("target", [])
        flat_targets = [t for sublist in targets for t in sublist if t]
        attribute_value = flat_targets[0] if flat_targets else "unknown"
        protected_attribute = _infer_protected_attribute(attribute_value)

        text = " ".join(row["post_tokens"])
        row_id = hashlib.md5(text.encode()).hexdigest()[:8]

        rows.append({
            "id": row_id,
            "text": text,
            "actual_label": actual,
            "protected_attribute": protected_attribute,
            "attribute_value": attribute_value,
        })

    result = pd.DataFrame(rows)

    if sample:
        result = result.sample(n=min(sample, len(result)), random_state=seed)

    return result.reset_index(drop=True)


def _infer_protected_attribute(target: str) -> str:
    target_lower = target.lower()
    gender_terms   = {"women", "men", "lgbtq", "transgender", "gay", "lesbian"}
    race_terms     = {"african", "black", "white", "asian", "hispanic",
                      "arab", "jewish", "jewish people"}
    religion_terms = {"muslim", "christian", "hindu", "buddhist", "islam"}
    age_terms      = {"elder", "youth", "children"}

    if any(t in target_lower for t in gender_terms):
        return "Gender"
    if any(t in target_lower for t in race_terms):
        return "Race/Ethnicity"
    if any(t in target_lower for t in religion_terms):
        return "Religion"
    if any(t in target_lower for t in age_terms):
        return "Age"
    return "Other"


def load_dataset_by_name(
    name: str,
    sample: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Unified entry point. name: 'jigsaw' or 'hatexplain'."""
    if name == "jigsaw":
        return load_jigsaw(sample=sample, **kwargs)
    if name == "hatexplain":
        return load_hatexplain(sample=sample, **kwargs)
    raise ValueError(f"Unknown dataset: {name!r}. Choose 'jigsaw' or 'hatexplain'.")
