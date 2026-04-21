# toxicity-fairness-bench

A fairness evaluation framework for commercial toxicity detection APIs,
benchmarked across gender, race, and religion using real-world datasets.

Compares **Google Perspective API** and **Anthropic Claude** across multiple
protected attributes, reporting standard fairness metrics (equalized odds,
demographic parity, FPR parity) alongside accuracy. Google Gemini is
supported by the framework but excluded from the published benchmark run
due to free-tier rate limits — see [docs/gemini-rate-limits.md](docs/gemini-rate-limits.md).

---

## Key findings

| Model | Overall Accuracy | Gender Gap | Race Gap | Religion Gap |
|---|---|---|---|---|
| Perspective API | 61% | 16 pp | 57 pp | 44 pp |
| Claude Haiku | 66% | 9 pp | 26 pp | 21 pp |

*"Gap" = max accuracy difference between any two subgroups within that
attribute (95% bootstrap CI). Smaller = fairer. Dataset: HateXplain,
1,000-sample draw, seed 42. Claude achieves both higher accuracy and smaller
fairness gaps across all three attributes.*

See [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) for full
confusion matrices, equalized odds plots, and per-subgroup breakdowns.

---

## Why this matters

Commercial content moderation APIs are widely deployed, yet their fairness
properties across demographic groups are poorly understood. This project
provides:

- **Reproducible benchmarks** on established datasets (HateXplain, 20k samples)
- **Per-attribute analysis** — Gender, Race/Ethnicity, Religion independently
- **Multiple fairness criteria** — because optimizing for one can hurt another
- **Real-world API comparison** — purpose-built classifier vs. prompted LLM

---

## Quickstart

```bash
# 1. Clone and install (requires Python 3.11+)
git clone https://github.com/lydsleepy/toxicity-fairness-bench.git
cd toxicity-fairness-bench
pip install -e ".[dev]"

# 2. Set up API keys
cp .env.example .env
# Edit .env with your keys (see "API Keys" section below)

# 3. Run the benchmark
python scripts/run_benchmark.py --sample 1000 --models perspective claude

# 4. Launch the web app
uvicorn app.main:app --reload
# Open http://localhost:8000
```

---

## Project structure

```
toxicity-fairness-bench/
├── app/                        # FastAPI web application
│   ├── main.py                 # App factory, routes
│   ├── dependencies.py         # Parquet data loader (cached singleton)
│   ├── routers/
│   │   ├── data.py             # GET /api/filters, GET /api/metrics
│   │   └── scorer.py           # POST /api/score (live API calls)
│   └── templates/
│       └── index.html          # Single-page HTML shell
├── static/
│   ├── css/main.css            # Design token system + components
│   └── js/app.js               # Plotly charts, filters, live scorer
├── src/toxicity_fairness/      # Installable Python package
│   ├── analyzers/              # One module per API
│   │   ├── base.py             # Abstract base + AnalysisResult dataclass
│   │   ├── perspective.py
│   │   ├── gemini.py           # Rate-limited; see docs/gemini-rate-limits.md
│   │   └── claude.py
│   ├── metrics/
│   │   └── fairness.py         # group_stats, fairness_report, gap metrics
│   ├── data/
│   │   └── loaders.py          # load_hatexplain(), load_jigsaw()
│   └── utils/
│       └── cache.py            # Parquet cache keyed by (dataset, model, sample)
├── scripts/
│   ├── run_benchmark.py        # CLI: runs APIs, saves results/raw_results.parquet
│   └── dashboard.py            # Legacy Streamlit app (preserved, not primary)
├── tests/                      # 26 unit tests — no API keys required
├── notebooks/
│   ├── analysis.ipynb          # Full benchmark analysis with charts
│   └── bias_analysis.ipynb     # Original class assignment (preserved)
├── results/
│   └── raw_results.parquet     # Pre-computed benchmark results (committed)
├── Procfile                    # Railway deployment entry point
├── railway.toml                # Railway build + deploy config
└── docs/                       # Setup guides and design notes
```

---

## API keys

| API | URL | Free tier |
|---|---|---|
| Google Perspective | [perspectiveapi.com](https://perspectiveapi.com) | Yes (1 QPS) |
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | Yes — see [rate limit notes](docs/gemini-rate-limits.md) |

Copy `.env.example` to `.env` and fill in your keys. Keys are never
committed — `.env` is in `.gitignore`.

---

## Datasets

| Dataset | Size | License | Notes |
|---|---|---|---|
| HateXplain | 20k | CC BY 4.0 | Auto-downloaded via HuggingFace |

HateXplain downloads automatically on first benchmark run
(`trust_remote_code=True` required; pin `datasets<3.0` — see [docs/datasets.md](docs/datasets.md)).

The original 40-row hand-labeled dataset from the class assignment is
preserved at `data.csv` for reference.

---

## Fairness metrics

For each (model, protected attribute) pair:

- **Accuracy** — correct / total per subgroup
- **False Positive Rate (FPR)** — non-toxic text flagged as toxic
- **False Negative Rate (FNR)** — toxic text missed
- **Equalized Odds** — difference in TPR and FPR across groups
- **Demographic Parity** — difference in positive prediction rates
- All accuracy estimates include 95% bootstrap confidence intervals

Note: HateXplain is heavily skewed toward toxic content, so FPR is
undefined for subgroups with no non-toxic examples. See the notebook for
full discussion.

---

## Deploy to Railway

1. Fork or push this repo to GitHub
2. Create a new project on [railway.app](https://railway.app) and connect the repo
3. Add environment variables: `PERSPECTIVE_API_KEY`, `ANTHROPIC_API_KEY`
4. Railway detects the `Procfile` automatically and deploys

The `results/raw_results.parquet` file is committed, so the dashboard
loads immediately without needing to re-run the benchmark.

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # with coverage
```

All 26 tests pass without API keys. CI runs automatically on every push
via GitHub Actions (Python 3.11 and 3.12).

---

## Tech stack

Python 3.11 · FastAPI · Uvicorn · Plotly.js · pandas · scikit-learn ·
anthropic · google-genai · google-api-python-client · tenacity · pytest · Railway

---

## Original class assignment

`notebooks/bias_analysis.ipynb` and `data.csv` are the original deliverables
from an introductory data science course at UT Austin. Preserved as-is.

---

## License

MIT
