# toxicity-fairness-bench

A fairness evaluation framework for commercial toxicity detection APIs,
benchmarked across gender, race, and religion using real-world datasets.

Compares **Google Perspective API** and **Anthropic Claude** across multiple
protected attributes, reporting standard fairness metrics (equalized odds,
demographic parity, FPR parity) alongside accuracy. Google Gemini is
supported by the framework but excluded from the published benchmark run
due to free-tier rate limits — see [docs/gemini-rate-limits.md](docs/gemini-rate-limits.md).

> **Run locally:** `streamlit run scripts/dashboard.py`
> **Deploy to Streamlit Cloud:** see [docs/deploy.md](docs/deploy.md) for 5-minute instructions.

---

## Key findings

| Model | Overall Accuracy | Gender Gap | Race Gap | Religion Gap |
|---|---|---|---|---|
| Perspective API | 61% | 16 pp | 57 pp | 44 pp |
| Claude Haiku | 66% | 9 pp | 26 pp | 21 pp |

*"Gap" = max accuracy difference between any two subgroups within that
attribute (95% bootstrap CI). Smaller = fairer. Dataset: HateXplain,
1,000-sample draw. Claude achieves both higher accuracy and smaller
fairness gaps across all three attributes.*

*Google Gemini is supported but excluded from this run — see
[docs/gemini-rate-limits.md](docs/gemini-rate-limits.md).*

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

# 3. Run the benchmark on a sample
python scripts/run_benchmark.py --sample 1000 --models perspective claude

# 4. Launch the dashboard
streamlit run scripts/dashboard.py
```

To include Gemini, first read [docs/gemini-rate-limits.md](docs/gemini-rate-limits.md)
for guidance on free-tier quotas, then add `gemini` to `--models`.

---

## Project structure

```
toxicity-fairness-bench/
├── src/toxicity_fairness/
│   ├── analyzers/          # One module per API
│   │   ├── base.py         # Abstract base class
│   │   ├── perspective.py
│   │   ├── gemini.py       # Gemini 2.5 Flash Lite (see rate-limit notes)
│   │   └── claude.py
│   ├── metrics/            # Fairness metric implementations
│   │   └── fairness.py
│   ├── data/               # Dataset loaders
│   │   └── loaders.py
│   └── utils/
│       └── cache.py        # Parquet cache to avoid redundant API calls
├── tests/                  # pytest unit tests (no API keys needed)
├── notebooks/
│   ├── bias_analysis.ipynb # Original class assignment (preserved)
│   └── analysis.ipynb      # Full benchmark analysis notebook
├── scripts/
│   ├── run_benchmark.py    # CLI entry point
│   └── dashboard.py        # Streamlit app
├── docs/                   # Setup guides and design notes
└── data.csv                # Original 40-row hand-labeled dataset (preserved)
```

---

## API keys

| API | URL | Free tier |
|---|---|---|
| Google Perspective | [perspectiveapi.com](https://perspectiveapi.com) | Yes (1 QPS) |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | Yes — see [rate limit notes](docs/gemini-rate-limits.md) |
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |

Copy `.env.example` to `.env` and fill in your keys. Keys are never
committed — `.env` is in `.gitignore`.

---

## Datasets

| Dataset | Size | License |
|---|---|---|
| HateXplain | 20k | CC BY 4.0 |

HateXplain downloads automatically via HuggingFace (`trust_remote_code=True`
required; pin `datasets<3.0` — see [docs/datasets.md](docs/datasets.md)).

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
- All metrics include 95% bootstrap confidence intervals

Note: HateXplain is a hate-speech dataset; most subgroup samples are
labeled toxic, so FPR is undefined for groups with no non-toxic examples.
See the notebook for full discussion of dataset limitations.

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # with coverage
```

CI runs automatically on every push via GitHub Actions.

---

## Tech stack

Python 3.11 · pandas · scikit-learn · anthropic · google-genai ·
google-api-python-client · streamlit · plotly · tenacity · pytest · GitHub Actions

---

## Original class assignment

The `notebooks/bias_analysis.ipynb` notebook and `data.csv` file are the
original deliverables from an introductory data science course at UT Austin.
They are preserved as-is for reference and to show the project's origins.

---

## License

MIT
