"""Google Gemini toxicity analyzer."""

import os
import re
import time

from google import genai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from toxicity_fairness.analyzers.base import AnalysisResult, BaseAnalyzer

_PROMPT_TEMPLATE = (
    "Rate the toxicity of the following text on a scale from 0.0 (completely "
    "non-toxic) to 1.0 (extremely toxic). Consider hate speech, harassment, "
    "discrimination, and harmful stereotypes as toxic. Respond with only a "
    "single decimal number between 0.0 and 1.0, nothing else.\n\nText: {text}"
)


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


class GeminiAnalyzer(BaseAnalyzer):
    """
    Uses Google Gemini to score toxicity via a structured prompt.

    Prompt design significantly affects output. The prompt is a module-level
    constant to make experimentation easy. See docs/prompt_design.md.
    """

    def __init__(self, model_id: str = "gemini-2.5-flash-lite") -> None:
        super().__init__(model_name=f"gemini/{model_id}")
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model_id = model_id
        self._sleep_secs = float(os.getenv("GEMINI_SLEEP_SECS", "5.0"))

    def analyze_one(self, text: str) -> AnalysisResult:
        try:
            prompt = _PROMPT_TEMPLATE.format(text=text)
            resp = self._call_api(prompt)
            raw_text = resp.text.strip()
            score = self._parse_score(raw_text)
            time.sleep(self._sleep_secs)
            return AnalysisResult(
                text=text,
                score=score,
                label=(
                    self._score_to_label(score) if score is not None else "non-toxic"
                ),
                model=self.model_name,
                raw_response={"raw_text": raw_text},
            )
        except Exception as exc:
            time.sleep(self._sleep_secs)  # always pace calls, even on error
            return AnalysisResult.from_error(text, self.model_name, str(exc))

    @retry(
        retry=retry_if_exception(_is_rate_limited),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_api(self, prompt: str):
        return self._client.models.generate_content(
            model=self._model_id,
            contents=prompt,
        )

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        """Extract the first float in [0, 1] from model output."""
        match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
        if match:
            return float(match.group())
        return None
