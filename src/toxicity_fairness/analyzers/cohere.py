"""Cohere toxicity analyzer"""

import os
import re
import time

import cohere
from cohere.errors import TooManyRequestsError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from toxicity_fairness.analyzers.base import AnalysisResult, BaseAnalyzer

_SYSTEM_PROMPT = (
    "You are a toxicity classifier. When you're given a text, respond with only a "
    "single decimal number between 0.0 and 1.0 representing toxicity: "
    "0.0 means completely non-toxic, 1.0 means extremely toxic. "
    "Consider hate speech, harassment, harmful stereotypes, and discrimination "
    "as toxic. Output only the number; no explanation, no punctuation."
)

def _redact_key(msg: str, api_key: str) -> str:
    """strip the raw api key from error strings, in case the sdk echoes it back"""
    return msg.replace(api_key, "REDACTED") if api_key else msg

class CohereAnalyzer(BaseAnalyzer):
    """uses cohere's command model to score toxicity via a structured system prompt
    mirrors ClaudeAnalyzer's system/user turn split rather than GeminiAnalyzer's
    single-turn prompt. cohere's ClientV2.chat() takes a `messages` list with the
    same {"role": "system"/"user", "content": ...} shape anthropic's sdk uses, so
    separating task definition from text input is equally natural here."""

    def __init__(self, model_id: str = "command-r7b-12-2024") -> None:
        super().__init__(model_name=f"cohere/{model_id}")
        self._api_key = os.environ["COHERE_API_KEY"]
        self._client = cohere.ClientV2(api_key=self._api_key)
        self._model_id = model_id
        self._sleep_secs = float(os.getenv("COHERE_SLEEP_SECS", "3.5"))

    def analyze_one(self, text: str) -> AnalysisResult:
        try:
            response = self._call_api(text)
            raw_text = response.message.content[0].text.strip()
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
                    "finish_reason": response.finish_reason,
                },
            )
        except Exception as exc:
            time.sleep(self._sleep_secs) # always pace calls even on error
            return AnalysisResult.from_error(
                text, self.model_name, _redact_key(str(exc), self._api_key)
            )
    
    @retry(
        retry=retry_if_exception_type(TooManyRequestsError),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    
    def _call_api(self, text:str):
        return self._client.chat(
            model=self._model_id,
            messages=[
                cohere.SystemChatMessageV2(content=_SYSTEM_PROMPT),
                cohere.UserChatMessageV2(content=text),
            ],
            max_tokens=16,
        )

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", raw)
        return float(match.group()) if match else None