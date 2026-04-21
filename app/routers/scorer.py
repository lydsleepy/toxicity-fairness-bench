from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=3)

_ANALYZER_MAP = {
    "perspective": (
        "toxicity_fairness.analyzers.perspective",
        "PerspectiveAnalyzer",
        "Perspective",
    ),
    "claude": (
        "toxicity_fairness.analyzers.claude",
        "ClaudeAnalyzer",
        "Claude Haiku",
    ),
    "gemini": (
        "toxicity_fairness.analyzers.gemini",
        "GeminiAnalyzer",
        "Gemini",
    ),
}


def _run_analyzer(key: str, text: str) -> dict[str, Any]:
    module_path, class_name, display_name = _ANALYZER_MAP[key]
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        analyzer = cls()
        result = analyzer.analyze_one(text)
        score = result.score
        return {
            "model": display_name,
            "model_key": key,
            "score": score,
            "score_pct": f"{score:.0%}" if score is not None else None,
            "label": result.label,
            "error": result.error,
        }
    except Exception as exc:
        return {
            "model": display_name,
            "model_key": key,
            "score": None,
            "score_pct": None,
            "label": None,
            "error": str(exc),
        }


class ScoreRequest(BaseModel):
    text: str


@router.post("/score")
async def score(req: ScoreRequest) -> dict[str, Any]:
    text = req.text.strip()
    if not text:
        return {"results": []}

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(_executor, _run_analyzer, key, text)
        for key in _ANALYZER_MAP
    ]
    results = await asyncio.gather(*tasks)
    return {"results": list(results)}
