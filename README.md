# toxicity-fairness-bench

A fairness evaluation framework for commercial toxicity detection APIs,
benchmarked across gender, race, and religion using real-world datasets.

Data Visualization Website Deployed at: https://toxicity-fairness-bench.up.railway.app/

Compares **Google Perspective API** and **Anthropic Claude** across multiple
protected attributes, reporting standard fairness metrics (equalized odds,
demographic parity, FPR parity) alongside accuracy. Google Gemini and
Cohere's Command R7B are also supported by the framework. Gemini is
excluded from the published benchmark due to free-tier rate limits, see
[docs/gemini-rate-limits.md](docs/gemini-rate-limits.md). Cohere is
benchmarked separately at a smaller sample size (n=500 vs. n=1,000) because
of trial-key quota limits, and it is deliberately excluded from the Live
Scorer tab because Cohere trial keys are not licensed for production or
public-facing use, see [docs/cohere-rate-limits.md](docs/cohere-rate-limits.md).
The dashboard's "Dataset" selector switches between the two views.

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

### Cohere comparison (separate n=500 run)

Cohere's Command R7B was added after the original benchmark was published.
Trial API keys cap chat calls at 1,000 per month, so Cohere was benchmarked
on the first 500 rows of the same seed=42 HateXplain draw (500 calls, zero
errors) instead of the full 1,000. See
[docs/cohere-rate-limits.md](docs/cohere-rate-limits.md) for the rate-limit
math and the confidence-interval cost of the smaller sample.

At n=500, demographic subgroup counts fall below the 5-example-per-class
threshold for Gender, Race/Ethnicity, and Religion, for every model
compared at that sample size, not just Cohere. Gap metrics for those three
attributes are unavailable in this view. Only overall accuracy is
meaningfully comparable across all four models at n=500. Use the
dashboard's "Dataset" selector to switch between "Published Benchmark
(n=1,000)" (Perspective and Claude, full gap metrics) and "Cohere
Comparison (n=500)" (all four models, gap metrics limited to the "Other"
category).

---

## Why this matters

Commercial content moderation APIs are widely deployed, yet their fairness
properties across demographic groups are poorly understood. This project
provides:

- **Reproducible benchmarks** on an established dataset (HateXplain, 20k samples)
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
# Edit .env with your keys (see "API keys" section below)

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
│   ├── main.py
│   ├── dependencies.py         # Parquet data loader (cached singleton)
│   ├── routers/
│   │   ├── data.py             # GET /api/filters, GET /api/metrics
│   │   └── scorer.py           # POST /api/score (live API calls)
│   └── templates/index.html
├── static/
│   ├── css/main.css
│   └── js/app.js
├── src/toxicity_fairness/      # Installable Python package
│   ├── analyzers/              # One module per API
│   │   ├── base.py
│   │   ├── perspective.py
│   │   ├── gemini.py
│   │   ├── claude.py
│   │   └── cohere.py
│   ├── metrics/fairness.py     # group_stats, fairness_report, gap metrics
│   ├── data/loaders.py         # load_hatexplain(), load_jigsaw()
│   └── utils/cache.py          # Parquet cache keyed by (dataset, model, sample)
├── scripts/
│   ├── run_benchmark.py        # CLI: runs APIs, saves results/raw_results.parquet
│   └── dashboard.py            # Legacy Streamlit app (preserved)
├── tests/                      # 26 unit tests — no API keys required
├── notebooks/
│   ├── analysis.ipynb          # Full benchmark analysis with charts
│   └── bias_analysis.ipynb     # Original class assignment (preserved)
└── results/
    ├── raw_results.parquet         # Published benchmark (n=1,000): Perspective + Claude
    └── cohere_500/
        └── raw_results.parquet     # Cohere comparison (n=500): all four models
```

---

## API keys

| API | URL | Free tier |
|---|---|---|
| Google Perspective | [perspectiveapi.com](https://perspectiveapi.com) | Yes (1 QPS) |
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | Yes, see [rate limit notes](docs/gemini-rate-limits.md) |
| Cohere | [dashboard.cohere.com](https://dashboard.cohere.com) | Trial key, see [rate limit notes](docs/cohere-rate-limits.md) |

Copy `.env.example` to `.env` and fill in your keys. Keys are never
committed, `.env` is in `.gitignore`.

`COHERE_API_KEY` is required to run `scripts/run_benchmark.py --models
cohere` locally. It is not required on the Railway deployment. Cohere is
intentionally excluded from the Live Scorer's API surface (see above), so
the deployed app never calls Cohere's API and never needs this key set as
a Railway environment variable.

---

## Datasets

| Dataset | Size | License |
|---|---|---|
| HateXplain | 20k | CC BY 4.0 |

HateXplain downloads automatically on first benchmark run via HuggingFace
(`trust_remote_code=True` required; pin `datasets<3.0` — see [docs/datasets.md](docs/datasets.md)).

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
undefined for subgroups with no non-toxic examples.

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

All 26 tests pass without API keys. CI runs on every push via GitHub Actions
(Python 3.11 and 3.12).

---

## Tech stack

Python 3.11 · FastAPI · Uvicorn · Plotly.js · pandas · scikit-learn ·
anthropic · google-genai · google-api-python-client · cohere · tenacity ·
pytest · GitHub Actions

---

## Original class assignment

`notebooks/bias_analysis.ipynb` and `data.csv` are the original deliverables
from an introductory data science course at UT Austin. Preserved as-is.

---

## License

MIT
