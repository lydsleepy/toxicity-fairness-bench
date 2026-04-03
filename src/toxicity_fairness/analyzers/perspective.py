"""Google Perspective API toxicity analyzer."""

import os
import time

from googleapiclient import discovery

from toxicity_fairness.analyzers.base import AnalysisResult, BaseAnalyzer


class PerspectiveAnalyzer(BaseAnalyzer):
    """
    Wraps Google's Perspective API for toxicity scoring.

    Rate limit: 1 QPS on free tier. We enforce a 1.1s sleep between
    calls to stay safely under the limit. Set PERSPECTIVE_SLEEP_SECS=0
    in .env with a paid key to remove the delay.
    """

    ATTRIBUTE = "TOXICITY"
    DISCOVERY_URL = (
        "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
    )

    def __init__(self) -> None:
        super().__init__(model_name="perspective")
        api_key = os.environ["PERSPECTIVE_API_KEY"]
        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl=self.DISCOVERY_URL,
            static_discovery=False,
        )
        self._sleep_secs = float(os.getenv("PERSPECTIVE_SLEEP_SECS", "1.1"))

    def analyze_one(self, text: str) -> AnalysisResult:
        try:
            body = {
                "comment": {"text": text},
                "requestedAttributes": {self.ATTRIBUTE: {}},
            }
            resp = self._client.comments().analyze(body=body).execute()
            score: float = (
                resp["attributeScores"][self.ATTRIBUTE]["summaryScore"]["value"]
            )
            time.sleep(self._sleep_secs)
            return AnalysisResult(
                text=text,
                score=score,
                label=self._score_to_label(score),
                model=self.model_name,
                raw_response=resp,
            )
        except Exception as exc:
            return AnalysisResult.from_error(text, self.model_name, str(exc))
