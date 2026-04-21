# CLAUDE.md — toxicity-fairness-bench

This file tells Claude Code everything it needs to know to work effectively in
this repository. Read it fully before taking any action.

---

## Project overview

`toxicity-fairness-bench` is a Python package that benchmarks commercial
toxicity detection APIs for demographic fairness. It was refactored from a
class assignment (`AI-Bias`, a 40-row notebook) into a production-quality
portfolio project with a proper package structure, CI, and a FastAPI web app
deployable on Railway.

**What it does:**
- Loads the HateXplain dataset (20k posts, auto-downloaded from HuggingFace)
- Runs texts through toxicity APIs and records binary predictions
- Computes fairness metrics per protected attribute (Gender, Race/Ethnicity,
  Religion, Age) — accuracy, FPR, FNR, equalized odds, demographic parity
- Reports results via CLI and an interactive Streamlit dashboard

**Published benchmark:** Perspective API vs. Claude Haiku, 1,000-sample draw
from HateXplain, seed=42. Gemini is implemented but excluded — see below.

**Key result (as of last benchmark run):**

| Model           | Overall Accuracy | Gender Gap | Race Gap | Religion Gap |
|-----------------|-----------------|------------|----------|--------------|
| Perspective API | 61%             | 16 pp      | 57 pp    | 44 pp        |
| Claude Haiku    | 66%             | 9 pp       | 26 pp    | 21 pp        |

Gap = max accuracy difference between any two subgroups (95% bootstrap CI).

---

## Repository layout

```
toxicity-fairness-bench/
├── src/toxicity_fairness/       # Installable Python package
│   ├── analyzers/
│   │   ├── base.py              # Abstract BaseAnalyzer + AnalysisResult dataclass
│   │   ├── perspective.py       # Google Perspective API (purpose-built classifier)
│   │   ├── gemini.py            # Google Gemini (prompted LLM, rate-limited)
│   │   └── claude.py            # Anthropic Claude (prompted LLM)
│   ├── metrics/
│   │   └── fairness.py          # group_stats, fairness_report, gap metrics
│   ├── data/
│   │   └── loaders.py           # load_hatexplain(), load_jigsaw(), load_dataset_by_name()
│   └── utils/
│       └── cache.py             # Parquet cache keyed by (dataset, model, sample)
├── app/                         # FastAPI web application
│   ├── main.py                  # App factory, route wiring, static + template mounts
│   ├── dependencies.py          # load_df() — lru_cache parquet singleton
│   ├── routers/
│   │   ├── data.py              # GET /api/filters, GET /api/metrics
│   │   └── scorer.py            # POST /api/score (live scorer)
│   └── templates/
│       └── index.html           # Single-page HTML shell
├── static/
│   ├── css/main.css             # Design token system + all component styles
│   └── js/app.js                # State, fetch, Plotly charts, scorer, tabs
├── scripts/
│   ├── run_benchmark.py         # CLI entry point
│   └── dashboard.py             # Legacy Streamlit app (kept for reference)
├── tests/
│   └── test_all.py              # 26 unit tests (no API keys required)
├── notebooks/
│   ├── analysis.ipynb           # Full benchmark analysis with charts
│   └── bias_analysis.ipynb      # Original class assignment (preserved, do not modify)
├── docs/
│   ├── deploy.md                # Streamlit Cloud deployment guide (legacy)
│   ├── datasets.md              # Dataset setup + cost estimates
│   ├── gemini-rate-limits.md    # Why Gemini was excluded from the benchmark
│   ├── prompt_design.md         # Prompt wording rationale + experiment ideas
│   └── execution_plan.md        # Original implementation plan (historical)
├── data.csv                     # Original 40-row hand-labeled dataset (do not modify)
├── Procfile                     # Railway: uvicorn app.main:app --host 0.0.0.0 --port $PORT
├── railway.toml                 # Railway build + deploy config
├── requirements.txt             # Production deps (mirrors pyproject.toml)
├── pyproject.toml               # Package config, ruff, mypy, pytest settings
├── .env.example                 # Template for API keys
└── .github/workflows/ci.yml     # GitHub Actions: lint + test on Python 3.11 and 3.12
```

---

## Environment setup

**Python:** 3.11 (installed at `/opt/homebrew/bin/python3.11`). The venv is at
`.venv/` — always activate before running anything:

```bash
source .venv/bin/activate
```

Or use the full path `.venv/bin/python` / `.venv/bin/uvicorn` / etc.

**Install (editable + dev extras):**

```bash
pip install -e ".[dev]"
```

**API keys:** Copy `.env.example` to `.env` and fill in. Keys are loaded by
`python-dotenv` at runtime. Never commit `.env`.

```
PERSPECTIVE_API_KEY=...
GEMINI_API_KEY=...        # optional — Gemini excluded from benchmark run
ANTHROPIC_API_KEY=...
```

Optional rate-limit tuning (defaults shown):
```
PERSPECTIVE_SLEEP_SECS=1.1
GEMINI_SLEEP_SECS=5.0
CLAUDE_SLEEP_SECS=0.5
```

**Dataset dependency quirk:** `datasets` must be `>=2.18,<3.0` — v3+ dropped
support for dataset scripts and breaks HateXplain. Pass `trust_remote_code=True`
(already done in `loaders.py`). The constraint is pinned in both
`pyproject.toml` and `requirements.txt`.

---

## Running things

### Tests (no API keys needed)

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

All 26 tests pass without any API keys. Tests cover `AnalysisResult`,
`BaseAnalyzer`, `GeminiAnalyzer._parse_score`, all fairness metrics, and
`ResultCache`.

### Lint

```bash
ruff check src/ tests/ scripts/ app/
ruff check --fix src/ tests/ scripts/ app/   # auto-fix what can be fixed
```

`E402` is suppressed for `scripts/*.py` via `per-file-ignores` in
`pyproject.toml` because both scripts do a `sys.path.insert` before local
imports — this is intentional (scripts aren't installed as a module).

### Benchmark CLI

```bash
# Standard published run (Perspective + Claude, 1,000 samples)
python scripts/run_benchmark.py --sample 1000 --models perspective claude

# Resume from cache (skip already-computed models)
python scripts/run_benchmark.py --sample 1000 --models perspective claude \
    --use-cache --output results/

# Small smoke test
python scripts/run_benchmark.py --sample 50 --models perspective claude

# Gemini small-sample run (see rate-limit notes before running large samples)
python scripts/run_benchmark.py --sample 100 --models gemini
```

Output written to `results/` (gitignored):
- `results/raw_results.parquet` — all predictions, read by the dashboard
- `results/fairness_report.csv` — summary metrics per model
- `results/cache/<hash>.parquet` — per-(model, dataset, sample) cache

### Web app (FastAPI)

```bash
uvicorn app.main:app --reload
# or: python -m uvicorn app.main:app --reload
```

Runs on http://localhost:8000. The app reads `results/raw_results.parquet`
(committed to the repo). Deploy to Railway: connect the GitHub repo, set API
key env vars, and Railway auto-detects the `Procfile` / `railway.toml`.

**API endpoints:**
- `GET /api/filters` — available models + protected attributes
- `GET /api/metrics?models=X&attribute=Y` — all chart data in one payload
- `POST /api/score` — live scorer (body: `{"text": "..."}`)

The legacy `scripts/dashboard.py` (Streamlit) is preserved but not the
primary interface.

---

## Package architecture

### Analyzers

All analyzers extend `BaseAnalyzer` (abstract). The interface is:

```python
analyzer.analyze_one(text: str) -> AnalysisResult
analyzer.analyze_batch(texts: list[str]) -> list[AnalysisResult]
```

`AnalysisResult` fields: `text`, `score` (0.0–1.0 or None on failure),
`label` ("toxic"/"non-toxic"), `model`, `error`, `raw_response`.

`analyze_batch` adds a tqdm progress bar; subclasses only need to implement
`analyze_one`. Rate limiting (sleep) is handled inside `analyze_one` of each
concrete class, not in the base.

**Adding a new analyzer:** subclass `BaseAnalyzer`, implement `analyze_one`,
add it to `ANALYZER_MAP` in `scripts/run_benchmark.py`. See `gemini.py` for
the retry+sleep pattern.

### Fairness metrics (`src/toxicity_fairness/metrics/fairness.py`)

Input: DataFrame with columns `actual_label`, `predicted_label`,
`attribute_value` (subgroup, e.g. "Male"), optionally `model`.

Key functions:
- `group_stats(df)` → DataFrame indexed by group with accuracy, TPR, FPR, FNR,
  PPR, and 95% bootstrap CI on accuracy (1000 resamples, seed=42)
- `accuracy_gap(stats)` → max pairwise accuracy difference
- `demographic_parity_gap(stats)` → max positive prediction rate difference
- `equalized_odds_gap(stats)` → dict with `tpr_gap`, `fpr_gap`, `max_gap`
- `fairness_report(df)` → one-row-per-model summary DataFrame

**Dataset caveat:** HateXplain is heavily skewed toward toxic content. Many
subgroups have no non-toxic examples, so FPR is `NaN` for those groups. The
notebook discusses this limitation in detail.

### Data loaders (`src/toxicity_fairness/data/loaders.py`)

- `load_hatexplain(sample, seed)` — downloads from HuggingFace, majority-votes
  annotator labels, infers protected attribute from target community tags
- `load_jigsaw(csv_path, sample, seed)` — loads Kaggle Jigsaw CSV (manual
  download required; not used in published benchmark)
- `load_dataset_by_name(name, sample)` — unified dispatcher

Standardized output schema: `id`, `text`, `actual_label`,
`protected_attribute`, `attribute_value`.

### Cache (`src/toxicity_fairness/utils/cache.py`)

`ResultCache` stores per-(dataset, model, sample) results as Parquet files.
Cache key = SHA-256 of `{dataset, model, sample}` JSON (first 16 hex chars).
Use `--use-cache` in the CLI to skip already-run models. Cache lives in
`results/cache/` (gitignored). Useful when re-running a partial benchmark.

---

## Gemini exclusion — what you need to know

Gemini is **fully implemented and tested** but excluded from the published
benchmark. Do not remove or deprecate the `GeminiAnalyzer` class.

**Why excluded:** Free-tier `gemini-2.5-flash-lite` has a burst rate limit of
~10–15 RPM that resets slowly. At 5s/call, 1,000 calls takes 9+ hours due to
constant 429 backoffs. See `docs/gemini-rate-limits.md` for the full timeline.

**Current default model:** `gemini-2.5-flash-lite` (switched from
`gemini-2.0-flash-lite` after that model's daily quota was exhausted).

**Retry logic:** `tenacity` with exponential backoff (10–120s, 4 attempts) on
`429 RESOURCE_EXHAUSTED`. Failed calls also sleep to prevent request storms.

**To run Gemini safely:** use `--sample 100` (completes in ~4 minutes before
rate limiting kicks in). For larger samples, either use a paid key or
`GEMINI_SLEEP_SECS=30` for an overnight run.

**Important:** `docs/deploy.md` and any benchmark run instructions should
specify `--models perspective claude` — **never include `gemini` in the
default run command** shown to end users.

---

## CI

`.github/workflows/ci.yml` runs on every push to `main`/`dev` and all PRs.

Matrix: Python 3.11 and 3.12.

Steps:
1. `pip install -e ".[dev]"`
2. `ruff check src/ tests/ scripts/ app/`
3. `pytest tests/ --cov=src --cov-report=xml`
4. Upload to Codecov (non-fatal if `CODECOV_TOKEN` secret is absent)

**To add `CODECOV_TOKEN`:** go to codecov.io, connect the `lydsleepy/toxicity-fairness-bench`
repo (free for public repos), copy the token, add it as a GitHub Actions
secret named `CODECOV_TOKEN`. This enables the coverage badge.

---

## Known issues and constraints

| Issue | Status | Notes |
|-------|--------|-------|
| Gemini excluded from benchmark | By design | Free-tier rate limits; see `docs/gemini-rate-limits.md` |
| `datasets` must be `<3.0` | Pinned | HateXplain uses old dataset script format |
| `trust_remote_code=True` required | Applied in `loaders.py` | HuggingFace requirement for HateXplain |
| `results/` gitignored | By design | Generate locally before deploying dashboard |
| FPR undefined for some groups | Known | HateXplain skewed toward toxic; discussed in notebook |
| Streamlit live scorer includes Gemini | Known | Dashboard live scorer attempts Gemini even if key missing; errors gracefully |

---

## What still needs to be done (as of 2026-04-06)

1. ~~**Deploy the dashboard**~~ — **Done.** Previously on Streamlit Cloud; now
   migrated to FastAPI. Deploy the new app to Railway by connecting the GitHub
   repo and setting API key env vars (`PERSPECTIVE_API_KEY`, `ANTHROPIC_API_KEY`).
2. **Add Codecov token** — add `CODECOV_TOKEN` to GitHub Secrets for the
   coverage badge. CI currently has `fail_ci_if_error: false` so it doesn't
   block, but the badge is missing.
3. **Run full benchmark and commit analysis notebook** — current benchmark
   results are from a 1,000-sample run. Re-run at larger scale if API budget
   allows; regenerate `notebooks/analysis.ipynb` with full charts.
4. **Add LICENSE file content** — `pyproject.toml` references `LICENSE` but
   the file may be empty. Fill in MIT license text.
5. **README polish** — update README to reflect FastAPI + Railway deployment,
   add CI badge, coverage badge, update live app URL.

---

## Repo history context

This repo was originally called `AI-Bias` (a UT Austin data science class
assignment). It was refactored into `toxicity-fairness-bench` by Claude Code
following a spec in `../CLAUDE_CODE_INSTRUCTIONS.md` (one directory up, outside
this repo — treat as historical context only, not authoritative). The original
40-row dataset and notebook are preserved at `data.csv` and
`notebooks/bias_analysis.ipynb`. Do not modify those files.

---

## Instructions for updating this file

Update this file as changes are made to the repo. 
This file should always contain the most up-to-date information about the repo.