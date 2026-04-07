"""Base class for all toxicity analyzer implementations."""

# changes annotation behavior - become strings rather than evaluated types
# so Python 3.9 can run syntax
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """Structured result from a toxicity analysis call."""

    text: str
    score: float | None          # 0.0–1.0; None if call failed
    label: str                   # "toxic" | "non-toxic"
    model: str
    error: str | None = None
    raw_response: dict = field(default_factory=dict)

    @classmethod
    def from_error(cls, text: str, model: str, error: str) -> "AnalysisResult":
        return cls(text=text, score=None, label="non-toxic", model=model, error=error)


class BaseAnalyzer(ABC):
    """
    Abstract base class for toxicity analyzers.

    All concrete analyzers must implement `analyze_one`. The public
    `analyze_batch` method handles rate limiting and progress reporting
    uniformly across all implementations.
    """

    THRESHOLD: float = 0.5  # score above this → toxic

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def analyze_one(self, text: str) -> AnalysisResult:
        """
        Call the underlying API for a single text string.

        Implementations should raise exceptions on unrecoverable errors
        (e.g., invalid API key) and return AnalysisResult.from_error()
        for recoverable per-request failures.
        """
        ...

    def analyze_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[AnalysisResult]:
        """
        Analyze a list of texts, handling progress display.

        Args:
            texts: List of text strings to analyze.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            List of AnalysisResult objects in the same order as input.
        """
        from tqdm import tqdm

        results: list[AnalysisResult] = []
        iterator = tqdm(texts, desc=self.model_name) if show_progress else texts

        for text in iterator:
            result = self.analyze_one(str(text))
            results.append(result)

        return results

    def _score_to_label(self, score: float) -> str:
        return "toxic" if score >= self.THRESHOLD else "non-toxic"
