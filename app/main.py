from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import data, scorer

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

app = FastAPI(title="Toxicity Fairness Benchmark", docs_url=None, redoc_url=None)
app.include_router(data.router, prefix="/api")
app.include_router(scorer.router, prefix="/api")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_templates = Jinja2Templates(directory=str(BASE / "templates"))

_PARQUET = ROOT / "results" / "raw_results.parquet"


@app.on_event("startup")
async def _startup_diagnostics() -> None:
    log.warning("=== STARTUP DIAGNOSTICS ===")
    log.warning("cwd:            %s", os.getcwd())
    log.warning("__file__:       %s", __file__)
    log.warning("ROOT:           %s", ROOT)
    log.warning("parquet path:   %s", _PARQUET)
    log.warning("parquet exists: %s", _PARQUET.exists())
    results_dir = ROOT / "results"
    if results_dir.exists():
        log.warning("results/ contents: %s", list(results_dir.iterdir()))
    else:
        log.warning("results/ directory NOT FOUND")
    _keep = {".toml", ".txt", ".py", ".parquet"}
    root_entries = sorted(
        p.name for p in ROOT.iterdir() if p.is_dir() or p.suffix in _keep
    )
    log.warning("ROOT entries:   %s", root_entries)
    log.warning("=== END DIAGNOSTICS ===")


@app.get("/api/debug")
async def debug() -> dict:
    results_dir = ROOT / "results"
    return {
        "cwd": os.getcwd(),
        "file": __file__,
        "root": str(ROOT),
        "parquet_path": str(_PARQUET),
        "parquet_exists": _PARQUET.exists(),
        "results_dir_exists": results_dir.exists(),
        "results_contents": (
            [str(p) for p in results_dir.iterdir()] if results_dir.exists() else []
        ),
        "root_dirs": sorted(p.name for p in ROOT.iterdir() if p.is_dir()),
    }


@app.get("/")
async def index(request: Request):
    return _templates.TemplateResponse(request, "index.html")
