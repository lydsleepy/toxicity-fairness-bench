from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.dependencies import df_available, load_df
from toxicity_fairness.metrics.fairness import fairness_report, group_stats

router = APIRouter()

_LABEL_MAP = {
    "perspective": "Perspective",
    "claude/claude-haiku-4-5-20251001": "Claude Haiku",
    "gemini": "Gemini",
}


def _display(model_key: str) -> str:
    return _LABEL_MAP.get(
        model_key,
        model_key.split("/")[-1].replace("-", " ").title(),
    )


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/inf with None for JSON serialisation."""
    if isinstance(obj, float) and (
        obj != obj or obj == float("inf") or obj == float("-inf")
    ):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


@router.get("/filters")
def get_filters() -> dict:
    if not df_available():
        return {"models": [], "protected_attributes": [], "error": None}
    try:
        df = load_df()
        models = sorted(df["model"].unique().tolist())
        attrs = sorted(df["protected_attribute"].dropna().unique().tolist())
        return {"models": models, "protected_attributes": attrs, "error": None}
    except Exception as exc:
        return {"models": [], "protected_attributes": [], "error": str(exc)}


@router.get("/metrics")
def get_metrics(  # noqa: B008
    models: list[str] = Query(default=[]),  # noqa: B008
    attribute: str = Query(default=""),  # noqa: B008
) -> dict[str, Any]:
    if not df_available():
        raise HTTPException(
            status_code=503,
            detail="Benchmark results not found. Run the benchmark first.",
        )

    df = load_df()

    if attribute:
        df = df[df["protected_attribute"] == attribute]
    if models:
        df = df[df["model"].isin(models)]

    if df.empty:
        raise HTTPException(
            status_code=404, detail="No data matches the selected filters."
        )

    # Accuracy tiles
    tiles = []
    for model_key, mdf in df.groupby("model"):
        acc = float((mdf["actual_label"] == mdf["predicted_label"]).mean())
        tiles.append({
            "model": _display(model_key),
            "model_key": model_key,
            "accuracy": acc,
        })

    # Accuracy by group
    by_group: list[dict[str, Any]] = []
    for model_key, mdf in df.groupby("model"):
        stats = group_stats(mdf.reset_index(drop=True)).reset_index()
        for _, row in stats.iterrows():
            by_group.append({
                "model": _display(model_key),
                "group": row["group"],
                "accuracy": row["accuracy"],
                "n": int(row["n"]),
            })

    # Fairness report
    report_df = fairness_report(df.reset_index(drop=True)).reset_index()
    report_rows = []
    for _, row in report_df.iterrows():
        report_rows.append({
            "model": row["model"],
            "overall_accuracy": row["overall_accuracy"],
            "accuracy_gap": row["accuracy_gap"],
            "dp_gap": row["dp_gap"],
            "tpr_gap": row["tpr_gap"],
            "fpr_gap": row["fpr_gap"],
        })

    # Scatter: FPR vs FNR per group
    scatter: list[dict[str, Any]] = []
    for model_key, mdf in df.groupby("model"):
        stats = group_stats(mdf.reset_index(drop=True)).reset_index()
        for _, row in stats.iterrows():
            scatter.append({
                "model": _display(model_key),
                "group": row["group"],
                "fpr": row["fpr"],
                "fnr": row["fnr"],
                "n": int(row["n"]),
            })

    return _clean({
        "accuracy_tiles": tiles,
        "accuracy_by_group": by_group,
        "fairness_report": report_rows,
        "scatter_points": scatter,
    })
