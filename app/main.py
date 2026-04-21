from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import data, scorer

BASE = Path(__file__).parent
ROOT = BASE.parent

app = FastAPI(title="Toxicity Fairness Benchmark", docs_url=None, redoc_url=None)
app.include_router(data.router, prefix="/api")
app.include_router(scorer.router, prefix="/api")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_templates = Jinja2Templates(directory=str(BASE / "templates"))


@app.get("/")
async def index(request: Request):
    return _templates.TemplateResponse(request, "index.html")
