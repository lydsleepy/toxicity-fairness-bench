#!/usr/bin/env python3
"""
CLI entry point for running the toxicity fairness benchmark.

Examples:
    python scripts/run_benchmark.py --dataset hatexplain --sample 100 --models perspective
    python scripts/run_benchmark.py --dataset hatexplain --sample 500 \\
        --models perspective gemini claude --output results/
    python scripts/run_benchmark.py --dataset hatexplain --sample 500 \\
        --models gemini --use-cache
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toxicity_fairness.analyzers.base import AnalysisResult
from toxicity_fairness.data.loaders import load_dataset_by_name
from toxicity_fairness.metrics.fairness import fairness_report, group_stats
from toxicity_fairness.utils.cache import ResultCache


ANALYZER_MAP = {
    "perspective": ("toxicity_fairness.analyzers.perspective", "PerspectiveAnalyzer"),
    "gemini":      ("toxicity_fairness.analyzers.gemini",      "GeminiAnalyzer"),
    "claude":      ("toxicity_fairness.analyzers.claude",      "ClaudeAnalyzer"),
}


def load_analyzer(name: str):
    module_path, class_name = ANALYZER_MAP[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def results_to_df(
    results: list[AnalysisResult],
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    # Results are returned in the same order as source_df (analyze_batch
    # preserves input order). Use positional assignment to avoid row
    # duplication when duplicate texts exist in the dataset.
    out = source_df.copy()
    out["predicted_label"] = [r.label for r in results]
    out["score"] = [r.score for r in results]
    out["model"] = [r.model for r in results]
    out["error"] = [r.error for r in results]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toxicity fairness benchmark")
    parser.add_argument("--dataset",   default="hatexplain",
                        choices=["hatexplain", "jigsaw"])
    parser.add_argument("--sample",    type=int, default=1000)
    parser.add_argument("--models",    nargs="+", default=["perspective"],
                        choices=list(ANALYZER_MAP))
    parser.add_argument("--output",    default="results/")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(output_dir / "cache")

    print(f"Loading dataset: {args.dataset} (sample={args.sample})")
    source_df = load_dataset_by_name(args.dataset, sample=args.sample, seed=args.seed)
    print(f"  {len(source_df)} rows loaded")

    all_results: list[pd.DataFrame] = []

    for model_name in args.models:
        cache_key = cache.make_key(args.dataset, model_name, args.sample)

        if args.use_cache and cache.exists(cache_key):
            print(f"\n[{model_name}] Loading from cache...")
            merged = cache.load(cache_key)
        else:
            print(f"\n[{model_name}] Running API calls...")
            analyzer = load_analyzer(model_name)
            results = analyzer.analyze_batch(source_df["text"].tolist())
            merged = results_to_df(results, source_df)
            cache.save(cache_key, merged)

        all_results.append(merged)

    combined = pd.concat(all_results, ignore_index=True)

    raw_path = output_dir / "raw_results.parquet"
    combined.to_parquet(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    print("\n── Fairness Report ──────────────────────────────────────")
    report = fairness_report(combined)
    print(report.to_string())

    report_path = output_dir / "fairness_report.csv"
    report.to_csv(report_path)
    print(f"\nReport saved to {report_path}")

    print("\n── Per-group stats ──────────────────────────────────────")
    for model, mdf in combined.groupby("model"):
        print(f"\n  {model}:")
        stats = group_stats(mdf)
        print(stats[["n", "accuracy", "tpr", "fpr"]].to_string())


if __name__ == "__main__":
    main()
