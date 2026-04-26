"""
Unit tests for toxicity_fairness.

Run: pytest tests/ -v
Run with coverage: pytest tests/ --cov=src --cov-report=term-missing
"""

import pandas as pd
import pytest

from toxicity_fairness.analyzers.base import AnalysisResult, BaseAnalyzer
from toxicity_fairness.analyzers.gemini import GeminiAnalyzer
from toxicity_fairness.metrics.fairness import (
    MIN_CLASS_N,
    accuracy_gap,
    demographic_parity_gap,
    equalized_odds_gap,
    fairness_report,
    group_stats,
    skewed_groups,
)
from toxicity_fairness.utils.cache import ResultCache


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Minimal results DataFrame for metric testing."""
    return pd.DataFrame([
        {"text": "A", "actual_label": "toxic",     "predicted_label": "toxic",
         "model": "m1", "attribute_value": "Male"},
        {"text": "B", "actual_label": "toxic",     "predicted_label": "non-toxic",
         "model": "m1", "attribute_value": "Male"},
        {"text": "C", "actual_label": "non-toxic", "predicted_label": "non-toxic",
         "model": "m1", "attribute_value": "Male"},
        {"text": "D", "actual_label": "non-toxic", "predicted_label": "toxic",
         "model": "m1", "attribute_value": "Male"},
        {"text": "E", "actual_label": "toxic",     "predicted_label": "toxic",
         "model": "m1", "attribute_value": "Female"},
        {"text": "F", "actual_label": "toxic",     "predicted_label": "toxic",
         "model": "m1", "attribute_value": "Female"},
        {"text": "G", "actual_label": "non-toxic", "predicted_label": "non-toxic",
         "model": "m1", "attribute_value": "Female"},
        {"text": "H", "actual_label": "non-toxic", "predicted_label": "non-toxic",
         "model": "m1", "attribute_value": "Female"},
    ])


class TestAnalysisResult:
    def test_from_error_returns_non_toxic_label(self):
        r = AnalysisResult.from_error("hello", "test-model", "timeout")
        assert r.label == "non-toxic"
        assert r.score is None
        assert r.error == "timeout"

    def test_normal_construction(self):
        r = AnalysisResult(text="hello", score=0.8, label="toxic", model="m")
        assert r.label == "toxic"
        assert r.error is None


class TestBaseAnalyzer:
    def test_score_to_label_threshold(self):
        class DummyAnalyzer(BaseAnalyzer):
            def analyze_one(self, text): ...
        a = DummyAnalyzer("dummy")
        assert a._score_to_label(0.6) == "toxic"
        assert a._score_to_label(0.5) == "toxic"
        assert a._score_to_label(0.49) == "non-toxic"
        assert a._score_to_label(0.0) == "non-toxic"


class TestGeminiParser:
    @pytest.mark.parametrize("raw,expected", [
        ("0.8",                        0.8),
        ("0.0",                        0.0),
        ("1.0",                        1.0),
        ("1",                          1.0),
        ("0",                          0.0),
        ("The score is 0.7 out of 1.", 0.7),
        ("not a number",               None),
        ("",                           None),
    ])
    def test_parse_score(self, raw, expected):
        assert GeminiAnalyzer._parse_score(raw) == expected


class TestGroupStats:
    def test_returns_expected_groups(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert set(stats.index) == {"Male", "Female"}

    def test_accuracy_values(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert stats.loc["Male", "accuracy"] == pytest.approx(0.5)
        assert stats.loc["Female", "accuracy"] == pytest.approx(1.0)

    def test_fpr_male(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert stats.loc["Male", "fpr"] == pytest.approx(0.5)

    def test_n_counts(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert stats.loc["Male", "n"] == 4
        assert stats.loc["Female", "n"] == 4

    def test_n_pos_n_neg_columns_exist(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert "n_pos" in stats.columns
        assert "n_neg" in stats.columns

    def test_n_pos_n_neg_values(self, sample_results_df):
        stats = group_stats(sample_results_df)
        # Male: rows A (toxic), B (toxic), C (non-toxic), D (non-toxic)
        assert stats.loc["Male", "n_pos"] == 2
        assert stats.loc["Male", "n_neg"] == 2
        # Female: rows E (toxic), F (toxic), G (non-toxic), H (non-toxic)
        assert stats.loc["Female", "n_pos"] == 2
        assert stats.loc["Female", "n_neg"] == 2


class TestSkewedGroups:
    def test_flags_all_positive_group(self):
        df = pd.DataFrame(
            [{"actual_label": "toxic", "predicted_label": "toxic",
              "attribute_value": "GroupA"}] * 40
            + [{"actual_label": "toxic", "predicted_label": "toxic",
                "attribute_value": "GroupB"}] * 20
            + [{"actual_label": "non-toxic", "predicted_label": "non-toxic",
                "attribute_value": "GroupB"}] * 20
        )
        stats = group_stats(df)
        bad = skewed_groups(stats, min_class_n=1)
        assert "GroupA" in bad
        assert "GroupB" not in bad

    def test_threshold_respected(self, sample_results_df):
        stats = group_stats(sample_results_df)
        # Both groups have n_pos=n_neg=2, which is below default MIN_CLASS_N=30
        assert len(skewed_groups(stats)) == 2
        # With threshold=2, both groups have exactly 2 of each class → not skewed
        assert len(skewed_groups(stats, min_class_n=2)) == 0

    def test_min_class_n_constant(self):
        assert MIN_CLASS_N == 30


class TestGapMetrics:
    def test_accuracy_gap(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert accuracy_gap(stats) == pytest.approx(0.5)

    def test_demographic_parity_gap_nonnegative(self, sample_results_df):
        stats = group_stats(sample_results_df)
        assert demographic_parity_gap(stats) >= 0.0

    def test_equalized_odds_gap_keys(self, sample_results_df):
        stats = group_stats(sample_results_df)
        eo = equalized_odds_gap(stats)
        assert {"tpr_gap", "fpr_gap", "max_gap"} == set(eo.keys())


class TestFairnessReport:
    def test_one_row_per_model(self, sample_results_df):
        report = fairness_report(sample_results_df)
        assert len(report) == 1
        assert report.index[0] == "m1"

    def test_accuracy_in_range(self, sample_results_df):
        report = fairness_report(sample_results_df)
        assert 0.0 <= report.loc["m1", "overall_accuracy"] <= 1.0

    def test_multi_model(self, sample_results_df):
        df2 = sample_results_df.copy()
        df2["model"] = "m2"
        combined = pd.concat([sample_results_df, df2], ignore_index=True)
        report = fairness_report(combined)
        assert set(report.index) == {"m1", "m2"}


class TestResultCache:
    def test_save_and_load(self, tmp_path, sample_results_df):
        cache = ResultCache(tmp_path)
        key = cache.make_key("hatexplain", "gemini", 100)
        cache.save(key, sample_results_df)
        loaded = cache.load(key)
        assert list(loaded.columns) == list(sample_results_df.columns)
        assert len(loaded) == len(sample_results_df)

    def test_exists_false_before_save(self, tmp_path):
        cache = ResultCache(tmp_path)
        assert not cache.exists(cache.make_key("jigsaw", "claude", 50))

    def test_key_is_deterministic(self, tmp_path):
        cache = ResultCache(tmp_path)
        k1 = cache.make_key("hatexplain", "gemini", 100)
        k2 = cache.make_key("hatexplain", "gemini", 100)
        assert k1 == k2

    def test_different_params_different_keys(self, tmp_path):
        cache = ResultCache(tmp_path)
        assert (
            cache.make_key("hatexplain", "gemini", 100)
            != cache.make_key("hatexplain", "claude", 100)
        )

    def test_clear_removes_file(self, tmp_path, sample_results_df):
        cache = ResultCache(tmp_path)
        key = cache.make_key("hatexplain", "gemini", 100)
        cache.save(key, sample_results_df)
        cache.clear(key)
        assert not cache.exists(key)
