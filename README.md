# toxicity-fairness-bench

A fairness evaluation framework for commercial toxicity detection APIs,
benchmarked across gender, race, and age using real-world datasets.

Compares **Google Perspective API**, **Google Gemini**, and **Anthropic Claude**
across multiple protected attributes, reporting standard fairness metrics
(equalized odds, demographic parity, FPR parity) alongside accuracy.

> **Live demo →** [your-app.streamlit.app](https://your-app.streamlit.app)
> *(deploy instructions below)*

---

## Key findings

| Model | Overall Accuracy | Gender Gap | Race Gap | Age Gap |
|---|---|---|---|---|
| Perspective API | 62% | 10 pp | 18 pp | 12 pp |
| Gemini 2.0 Flash | 87% | 5 pp | 9 pp | 7 pp |
| Claude Sonnet | TBD | TBD | TBD | TBD |

*"Gap" = max accuracy difference between any two subgroups within that
attribute. Smaller is fairer. Fill in Claude column after running the
benchmark.*

See [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) for full
confusion matrices, calibration curves, and statistical significance tests.

---

## Why this matters

Commercial content moderation APIs are widely deployed, yet their fairness
properties across demographic groups are poorly understood. This project
provides:

- **Reproducible benchmarks** on established datasets (HateXplain)
- **Intersectional analysis** — not just per-attribute, but across
  attribute combinations
- **Multiple fairness criteria** — because optimizing for one can hurt
  another
- **Actionable prompt interventions** — testing whether rephrasing API
  prompts reduces bias

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/lydsleepy/AI-Bias.git
cd AI-Bias
pip install -e ".[dev]"

# 2. Set up API keys
cp .env.example .env
# Edit .env with your keys (see "API Keys" section below)

# 3. Run the benchmark on a small sample
python scripts/run_benchmark.py --sample 200 --models perspective gemini

# 4. Launch the dashboard
streamlit run scripts/dashboard.py
```

---

## Project structure

```
AI-Bias/
├── src/toxicity_fairness/
│   ├── analyzers/          # One module per API
│   │   ├── base.py         # Abstract base class
│   │   ├── perspective.py
│   │   ├── gemini.py
│   │   └── claude.py
│   ├── metrics/            # Fairness metric implementations
│   │   └── fairness.py
│   ├── data/               # Dataset loaders
│   │   └── loaders.py
│   └── utils/
│       └── cache.py
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
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | Yes |
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |

Copy `.env.example` to `.env` and fill in your keys. Keys are never
committed — `.env` is in `.gitignore`.

---

## Datasets

| Dataset | Size | License |
|---|---|---|
| HateXplain | 20k | CC BY 4.0 |

HateXplain downloads automatically via HuggingFace. See
[`docs/datasets.md`](docs/datasets.md) for full setup instructions.

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

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # with coverage
```

CI runs automatically on every push via GitHub Actions.

---

## Tech stack

Python 3.11 · pandas · scikit-learn · anthropic · google-generativeai ·
googleapiclient · streamlit · plotly · pytest · GitHub Actions

---

## Original class assignment

The `notebooks/bias_analysis.ipynb` notebook and `data.csv` file are the
original deliverables from an introductory data science course at UT Austin.
They are preserved as-is for reference and to show the project's origins.

---

## License

MIT
