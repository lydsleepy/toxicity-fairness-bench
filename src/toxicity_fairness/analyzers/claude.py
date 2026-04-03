"""Anthropic Claude toxicity analyzer."""

import os
import re
import time

import anthropic

from toxicity_fairness.analyzers.base import AnalysisResult, BaseAnalyzer

_SYSTEM_PROMPT = (
    "You are a toxicity classifier. When given a text, respond with only a "
    "single decimal number between 0.0 and 1.0 representing toxicity: "
    "0.0 means completely non-toxic, 1.0 means extremely toxic. "
    "Consider hate speech, harassment, harmful stereotypes, and discrimination "
    "as toxic. Output only the number — no explanation, no punctuation."
)


class ClaudeAnalyzer(BaseAnalyzer):
    """
    Uses Anthropic Claude to score toxicity via a structured system prompt.

    A system prompt separates task definition from text input — cleaner
    design than embedding instructions in the user turn.
    """

    def __init__(self, model_id: str = "claude-haiku-4-5-20251001") -> None:
        super().__init__(model_name=f"claude/{model_id}")
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model_id = model_id
        self._sleep_secs = float(os.getenv("CLAUDE_SLEEP_SECS", "0.5"))

    def analyze_one(self, text: str) -> AnalysisResult:
        try:
            message = self._client.messages.create(
                model=self._model_id,
                max_tokens=16,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            raw_text = message.content[0].text.strip()
            score = self._parse_score(raw_text)
            time.sleep(self._sleep_secs)
            return AnalysisResult(
                text=text,
                score=score,
                label=(
                    self._score_to_label(score) if score is not None else "non-toxic"
                ),
                model=self.model_name,
                raw_response={
                    "raw_text": raw_text,
                    "stop_reason": message.stop_reason,
                },
            )
        except Exception as exc:
            return AnalysisResult.from_error(text, self.model_name, str(exc))

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
        return float(match.group()) if match else None
