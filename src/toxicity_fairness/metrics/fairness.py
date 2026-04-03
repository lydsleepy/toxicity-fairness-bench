"""
Fairness metrics for evaluating toxicity classifier bias.

All metrics operate on a pandas DataFrame with at minimum:
  - actual_label     : "toxic" | "non-toxic"
  - predicted_label  : "toxic" | "non-toxic"
  - attribute_value  : str (e.g., "Male", "Female")

References:
  Hardt et al., "Equality of Opportunity in Supervised Learning" (2016)
  Chouldechova, "Fair prediction with disparate impact" (2017)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _binary(series: pd.Series) -> np.ndarray:
    """Convert 'toxic'/'non-toxic' string labels to 1/0 int array."""
    return (series == "toxic").astype(int).to_numpy()


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for stat_fn(values)."""
    rng = np.random.default_rng(rng_seed)
    boot_stats = [
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def group_stats(df: pd.DataFrame, group_col: str = "attribute_value") -> pd.DataFrame:
    """
    Compute per-group accuracy, TPR, FPR, FNR, and positive prediction rate
    with 95% bootstrap confidence intervals.

    Returns a DataFrame indexed by group value.
    """
    rows = []
    for group, gdf in df.groupby(group_col):
        y_true = _binary(gdf["actual_label"])
        y_pred = _binary(gdf["predicted_label"])
        n = len(y_true)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        acc    = (tp + tn) / n if n else float("nan")
        tpr    = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr    = fp / (fp + tn) if (fp + tn) else float("nan")
        fnr    = fn / (fn + tp) if (fn + tp) else float("nan")
        pprate = (tp + fp) / n if n else float("nan")

        correct = (y_pred == y_true).astype(float)
        acc_ci = _bootstrap_ci(correct, np.mean)

        rows.append({
            "group":     group,
            "n":         n,
            "accuracy":  acc,
            "acc_ci_lo": acc_ci[0],
            "acc_ci_hi": acc_ci[1],
            "tpr":       tpr,
            "fpr":       fpr,
            "fnr":       fnr,
            "pprate":    pprate,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    return pd.DataFrame(rows).set_index("group")


def equalized_odds_gap(stats_df: pd.DataFrame) -> dict[str, float]:
    """
    Equalized odds: model should have equal TPR and FPR across groups.
    Returns max pairwise gap for each rate.
    """
    tpr_gap = float(stats_df["tpr"].max() - stats_df["tpr"].min())
    fpr_gap = float(stats_df["fpr"].max() - stats_df["fpr"].min())
    return {"tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "max_gap": max(tpr_gap, fpr_gap)}


def demographic_parity_gap(stats_df: pd.DataFrame) -> float:
    """Demographic parity: equal positive prediction rate across groups."""
    return float(stats_df["pprate"].max() - stats_df["pprate"].min())


def accuracy_gap(stats_df: pd.DataFrame) -> float:
    """Max accuracy difference across groups."""
    return float(stats_df["accuracy"].max() - stats_df["accuracy"].min())


def fairness_report(
    df: pd.DataFrame,
    model_col: str = "model",
    group_col: str = "attribute_value",
) -> pd.DataFrame:
    """
    Generate a summary fairness report across all (model, attribute) pairs.

    Returns a DataFrame with one row per model.
    """
    rows = []
    for model, mdf in df.groupby(model_col):
        stats = group_stats(mdf, group_col)
        eo = equalized_odds_gap(stats)
        rows.append({
            "model":            model,
            "overall_accuracy": float(
                (mdf["actual_label"] == mdf["predicted_label"]).mean()
            ),
            "accuracy_gap":     accuracy_gap(stats),
            "dp_gap":           demographic_parity_gap(stats),
            "tpr_gap":          eo["tpr_gap"],
            "fpr_gap":          eo["fpr_gap"],
        })
    return pd.DataFrame(rows).set_index("model")
