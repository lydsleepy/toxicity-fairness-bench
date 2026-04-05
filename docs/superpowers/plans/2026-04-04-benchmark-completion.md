# Benchmark Completion & Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take toxicity-fairness-bench from a working scaffold to a fully executed, zero-placeholder, resume-ready portfolio project.

**Architecture:** Run a 1,000-sample HateXplain benchmark across three models, execute a comprehensive analysis notebook, fill all README placeholders with real numbers, and wire up Streamlit deployment prep — all committed to main with CI green.

**Tech Stack:** Python 3.11, pandas, scikit-learn, plotly, anthropic, google-genai, google-api-python-client, datasets (HuggingFace), jupyter/nbconvert, streamlit, pytest, GitHub Actions

---

## File Map

| File | Action | What changes |
|---|---|---|
| `.env.example` | stage (was unstaged) | restore to committed state |
| `pyproject.toml` | stage (unstaged fix) | `google-genai>=1.0` dep |
| `scripts/run_benchmark.py` | stage (unstaged fix) + edit | merge fix; default sample → 1000 |
| `src/toxicity_fairness/analyzers/gemini.py` | stage (unstaged fix) | new genai client API |
| `src/toxicity_fairness/data/loaders.py` | stage (unstaged fix) | int→str label decoder |
| `.gitignore` | edit | add `results/` line |
| `requirements.txt` | create | runtime deps for Streamlit Cloud |
| `scripts/dashboard.py` | edit | fix GitHub URL placeholder |
| `docs/deploy.md` | create | 5-step Streamlit deploy instructions |
| `notebooks/analysis.ipynb` | rewrite + execute | full analysis with baked-in outputs |
| `README.md` | edit | key findings table + demo section |

---

## Task 1: Commit in-flight bug fixes

**Files:**
- Stage: `.env.example`, `pyproject.toml`, `scripts/run_benchmark.py`, `src/toxicity_fairness/analyzers/gemini.py`, `src/toxicity_fairness/data/loaders.py`

- [ ] **Step 1: Stage all five files**

```bash
git add .env.example pyproject.toml scripts/run_benchmark.py \
    src/toxicity_fairness/analyzers/gemini.py \
    src/toxicity_fairness/data/loaders.py
```

- [ ] **Step 2: Verify staging looks right**

```bash
git diff --cached --stat
```

Expected output: 5 files changed.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
fix: SDK migration, label decoder, merge fix, restore .env.example

- gemini.py: migrate from google-generativeai to google-genai>=1.0 client API
- loaders.py: handle HateXplain integer labels (0=hatespeech,1=normal,2=offensive)
- run_benchmark.py: use positional assignment to avoid row duplication on duplicate texts
- pyproject.toml: align dep to google-genai>=1.0
- .env.example: restore (was accidentally deleted)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Set up virtualenv and install dependencies

**Files:** none changed

- [ ] **Step 1: Create a virtualenv**

```bash
python3 -m venv .venv
```

- [ ] **Step 2: Activate and install**

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: long install output ending with `Successfully installed toxicity-fairness-bench-0.1.0`

- [ ] **Step 3: Confirm pytest is available**

```bash
pytest --version
```

Expected output: `pytest 8.x.x`

---

## Task 3: Verify test suite passes

**Files:** none changed; fix failures if found

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS. If any fail, look at the error — likely a stale import or missing dep. The most likely failure is `ImportError` on `from google import genai` if the old `google-generativeai` package is conflicting.

- [ ] **Step 2 (only if tests fail): Uninstall conflicting package**

If you see `ImportError: cannot import name 'genai' from 'google'`:

```bash
pip uninstall google-generativeai -y
pip install "google-genai>=1.0"
pytest tests/ -v --tb=short
```

- [ ] **Step 3: Confirm all 22+ tests pass, then commit if any fixes were needed**

If `pyproject.toml` was changed to resolve a dep conflict:
```bash
git add pyproject.toml
git commit -m "fix: resolve google-genai import conflict

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Update default sample size from 200 → 1,000

**Files:**
- Modify: `scripts/run_benchmark.py:64`
- Modify: `notebooks/analysis.ipynb` (markdown prerequisite cell)
- Modify: `README.md:63` (quickstart example)

- [ ] **Step 1: Update argparse default in run_benchmark.py**

In `scripts/run_benchmark.py`, change line 64:
```python
# before
parser.add_argument("--sample",    type=int, default=200)
# after
parser.add_argument("--sample",    type=int, default=1000)
```

- [ ] **Step 2: Update notebook prerequisite markdown cell**

In `notebooks/analysis.ipynb`, the first markdown cell (cell-0) currently says:
```
python scripts/run_benchmark.py --dataset hatexplain --sample 500 --models perspective gemini claude
```
Change `--sample 500` to `--sample 1000`.

- [ ] **Step 3: Update README quickstart**

In `README.md` line ~63:
```bash
# before
python scripts/run_benchmark.py --sample 200 --models perspective gemini
# after
python scripts/run_benchmark.py --sample 1000 --models perspective gemini claude
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_benchmark.py notebooks/analysis.ipynb README.md
git commit -m "$(cat <<'EOF'
config: set default benchmark sample size to 1,000

Perspective free tier handles this at 1 QPS in ~17 min.
Gemini free tier handles this at 2s sleep in ~33 min.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Tighten .gitignore and add requirements.txt

**Files:**
- Modify: `.gitignore`
- Create: `requirements.txt`

- [ ] **Step 1: Add `results/` to .gitignore**

In `.gitignore`, replace the current content with:

```
.env
__pycache__/
*.pyc
*.egg-info/
.venv/
results/
data/raw/
.DS_Store
```

Note: `results/` supersedes both `results/cache/` and `*.parquet`. The `.venv/` and `*.egg-info/` lines prevent committing local install artifacts.

- [ ] **Step 2: Create requirements.txt for Streamlit Community Cloud**

Create `requirements.txt` at the repo root:

```
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
anthropic>=0.25
google-genai>=1.0
google-api-python-client>=2.120
python-dotenv>=1.0
plotly>=5.20
streamlit>=1.32
datasets>=2.18
tqdm>=4.66
tenacity>=8.2
scipy>=1.12
```

- [ ] **Step 3: Verify the egg-info directory is now ignored**

```bash
git status
```

The `src/toxicity_fairness_bench.egg-info/` entry should no longer appear under untracked files.

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "$(cat <<'EOF'
chore: tighten .gitignore and add requirements.txt for Streamlit Cloud

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Fix dashboard.py placeholder and add deploy docs

**Files:**
- Modify: `scripts/dashboard.py:38`
- Create: `docs/deploy.md`

- [ ] **Step 1: Fix GitHub URL in dashboard.py**

In `scripts/dashboard.py` line 38, change:
```python
# before
"[View source on GitHub](https://github.com/yourusername/AI-Bias)"
# after
"[View source on GitHub](https://github.com/lydsleepy/AI-Bias)"
```

- [ ] **Step 2: Create docs/deploy.md**

Create `docs/deploy.md`:

```markdown
# Deploying the Streamlit Dashboard

The dashboard is a single-file Streamlit app at `scripts/dashboard.py`.
It reads pre-computed results from `results/raw_results.parquet`, which
you generate by running the benchmark.

## Prerequisites

1. Run the benchmark to generate results:

   ```bash
   python scripts/run_benchmark.py --sample 1000 \
       --models perspective gemini claude \
       --output results/
   ```

2. Verify the dashboard runs locally:

   ```bash
   streamlit run scripts/dashboard.py
   ```

## Deploying to Streamlit Community Cloud (free)

1. **Push your repo to GitHub** (must be public or you must be on a paid Streamlit plan).

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. **Click "New app"** and fill in:
   - Repository: `lydsleepy/AI-Bias`
   - Branch: `main`
   - Main file path: `scripts/dashboard.py`

4. **Add API keys as Streamlit Secrets** — in the app settings under "Secrets", add:

   ```toml
   PERSPECTIVE_API_KEY = "your_key"
   GEMINI_API_KEY = "your_key"
   ANTHROPIC_API_KEY = "your_key"
   ```

   The app uses `python-dotenv` locally; Streamlit Secrets replace `.env` in production.

5. **Click "Deploy"**. The app will be live at a `*.streamlit.app` URL in ~2 minutes.

   Update the README with your live URL once deployed.

## Notes

- `requirements.txt` at the repo root tells Streamlit Cloud what to install.
- The `results/` directory is gitignored — the live scorer (bottom of the dashboard) works without pre-computed results, but the charts require running the benchmark first.
- Rate limits: Perspective API has a 1 QPS free tier. The live scorer makes one call per model per click, well within limits.
```

- [ ] **Step 3: Commit both files**

```bash
git add scripts/dashboard.py docs/deploy.md
git commit -m "$(cat <<'EOF'
docs: fix dashboard GitHub URL, add Streamlit deploy instructions

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Run the benchmark

**Files:** none committed (results/ is gitignored)

- [ ] **Step 1: Confirm API keys are set**

```bash
grep -E "PERSPECTIVE_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY" .env
```

Expected: three lines with real keys (not `your_*_here`).

- [ ] **Step 2: Run the benchmark**

```bash
source .venv/bin/activate
python scripts/run_benchmark.py \
    --dataset hatexplain \
    --sample 1000 \
    --models perspective gemini claude \
    --output results/
```

This takes ~50–60 minutes total:
- Perspective: ~1000 calls at 1.1s sleep ≈ 18 min
- Gemini: ~1000 calls at 2.0s sleep ≈ 33 min
- Claude: ~1000 calls at 0.5s sleep ≈ 8 min

Cache is written per model. If interrupted, re-run with `--use-cache` to skip already-finished models:
```bash
python scripts/run_benchmark.py --dataset hatexplain --sample 1000 \
    --models perspective gemini claude --output results/ --use-cache
```

- [ ] **Step 3: Verify output**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('results/raw_results.parquet')
print('Rows:', len(df))
print('Models:', df['model'].unique())
print('Error rate:', df['error'].notna().mean():.1%})
print(df.groupby('model')['error'].apply(lambda s: s.notna().mean()))
"
```

Expected: 3,000 rows (1,000 per model), error rate < 5%.

---

## Task 8: Write and execute the analysis notebook

**Files:**
- Rewrite: `notebooks/analysis.ipynb`

- [ ] **Step 1: Write the complete notebook**

Write the full `notebooks/analysis.ipynb` replacing all existing content:

```json
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-setup",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '../src')\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.express as px\n",
    "import plotly.figure_factory as ff\n",
    "import plotly.io as pio\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "pio.renderers.default = 'notebook'\n",
    "\n",
    "from toxicity_fairness.metrics.fairness import (\n",
    "    group_stats,\n",
    "    fairness_report,\n",
    "    equalized_odds_gap,\n",
    "    demographic_parity_gap,\n",
    "    accuracy_gap,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-load",
   "metadata": {},
   "source": ["## 1. Load results\n", "\n", "Pre-computed results from `scripts/run_benchmark.py --sample 1000 --models perspective gemini claude`."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-load",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_parquet('../results/raw_results.parquet')\n",
    "print(f'Total rows: {len(df):,}')\n",
    "print(f'Models: {list(df[\"model\"].unique())}')\n",
    "print(f'Protected attributes: {sorted(df[\"protected_attribute\"].unique())}')\n",
    "print(f'\\nRows per model:')\n",
    "print(df['model'].value_counts().to_string())\n",
    "print(f'\\nLabel distribution (actual):')\n",
    "print(df['actual_label'].value_counts().to_string())\n",
    "print(f'\\nAttribute distribution:')\n",
    "print(df[df['model'] == df['model'].iloc[0]]['protected_attribute'].value_counts().to_string())\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-report",
   "metadata": {},
   "source": ["## 2. Overall fairness report\n", "\n", "One row per model. Gaps are computed within each protected attribute independently to avoid cross-attribute comparisons."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-report",
   "metadata": {},
   "outputs": [],
   "source": [
    "ATTRIBUTES = ['Gender', 'Race/Ethnicity', 'Religion']\n",
    "MIN_SAMPLES = 20  # skip attribute/model pairs with too few samples to be meaningful\n",
    "\n",
    "rows = []\n",
    "for model, mdf in df.groupby('model'):\n",
    "    overall_acc = (mdf['actual_label'] == mdf['predicted_label']).mean()\n",
    "    row = {'model': model.split('/')[-1], 'overall_accuracy': overall_acc}\n",
    "    for attr in ATTRIBUTES:\n",
    "        adf = mdf[mdf['protected_attribute'] == attr]\n",
    "        if len(adf) >= MIN_SAMPLES:\n",
    "            stats = group_stats(adf)\n",
    "            row[f'{attr.lower().replace(\"/\",\"_\")}_acc_gap'] = accuracy_gap(stats)\n",
    "            row[f'{attr.lower().replace(\"/\",\"_\")}_fpr_gap'] = equalized_odds_gap(stats)['fpr_gap']\n",
    "        else:\n",
    "            row[f'{attr.lower().replace(\"/\",\"_\")}_acc_gap'] = float('nan')\n",
    "            row[f'{attr.lower().replace(\"/\",\"_\")}_fpr_gap'] = float('nan')\n",
    "    rows.append(row)\n",
    "\n",
    "summary = pd.DataFrame(rows).set_index('model')\n",
    "summary.style.format('{:.1%}', na_rep='n/a').background_gradient(\n",
    "    subset=[c for c in summary.columns if 'gap' in c], cmap='RdYlGn_r'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-acc",
   "metadata": {},
   "source": ["## 3. Accuracy by subgroup\n", "\n", "Bar charts with 95% bootstrap confidence intervals. Smaller gaps between bars = more equitable."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-acc-gender",
   "metadata": {},
   "outputs": [],
   "source": [
    "def accuracy_bar_chart(df, attribute, title):\n",
    "    adf = df[df['protected_attribute'] == attribute]\n",
    "    rows = []\n",
    "    for model, mdf in adf.groupby('model'):\n",
    "        stats = group_stats(mdf)\n",
    "        for grp, row in stats.iterrows():\n",
    "            if row['n'] >= 10:\n",
    "                rows.append({\n",
    "                    'model': model.split('/')[-1],\n",
    "                    'group': grp,\n",
    "                    'accuracy': row['accuracy'],\n",
    "                    'ci_lo': row['acc_ci_lo'],\n",
    "                    'ci_hi': row['acc_ci_hi'],\n",
    "                    'n': row['n'],\n",
    "                })\n",
    "    bar_df = pd.DataFrame(rows)\n",
    "    fig = px.bar(\n",
    "        bar_df, x='group', y='accuracy', color='model', barmode='group',\n",
    "        error_y=bar_df['ci_hi'] - bar_df['accuracy'],\n",
    "        error_y_minus=bar_df['accuracy'] - bar_df['ci_lo'],\n",
    "        title=title,\n",
    "        color_discrete_sequence=px.colors.qualitative.Set2,\n",
    "        hover_data=['n'],\n",
    "    )\n",
    "    fig.update_layout(yaxis_tickformat='.0%', yaxis_range=[0, 1],\n",
    "                      xaxis_title=attribute, yaxis_title='Accuracy')\n",
    "    fig.show()\n",
    "\n",
    "accuracy_bar_chart(df, 'Gender', 'Accuracy by gender subgroup (95% bootstrap CI)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-acc-race",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_bar_chart(df, 'Race/Ethnicity', 'Accuracy by race/ethnicity subgroup (95% bootstrap CI)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-acc-religion",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_bar_chart(df, 'Religion', 'Accuracy by religion subgroup (95% bootstrap CI)')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-eo",
   "metadata": {},
   "source": ["## 4. Equalized odds: TPR vs. FPR\n", "\n", "A fair classifier has equal TPR (hit rate) and FPR (false alarm rate) across groups. Points clustered together = more equitable."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-eo",
   "metadata": {},
   "outputs": [],
   "source": [
    "def equalized_odds_scatter(df, attribute, title):\n",
    "    adf = df[df['protected_attribute'] == attribute]\n",
    "    rows = []\n",
    "    for model, mdf in adf.groupby('model'):\n",
    "        stats = group_stats(mdf)\n",
    "        for grp, row in stats.iterrows():\n",
    "            if row['n'] >= 10:\n",
    "                rows.append({\n",
    "                    'model': model.split('/')[-1],\n",
    "                    'group': grp,\n",
    "                    'tpr': row['tpr'],\n",
    "                    'fpr': row['fpr'],\n",
    "                    'n': row['n'],\n",
    "                })\n",
    "    eo_df = pd.DataFrame(rows)\n",
    "    fig = px.scatter(\n",
    "        eo_df, x='fpr', y='tpr', color='model', text='group', size='n',\n",
    "        title=title,\n",
    "        color_discrete_sequence=px.colors.qualitative.Set2,\n",
    "        labels={'fpr': 'False Positive Rate', 'tpr': 'True Positive Rate'},\n",
    "    )\n",
    "    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,\n",
    "                  line=dict(color='gray', dash='dash'))\n",
    "    fig.update_traces(textposition='top center')\n",
    "    fig.update_layout(xaxis_range=[-0.05, 1.05], yaxis_range=[-0.05, 1.05])\n",
    "    fig.show()\n",
    "\n",
    "equalized_odds_scatter(df, 'Gender', 'Equalized odds by gender subgroup')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-eo-race",
   "metadata": {},
   "outputs": [],
   "source": [
    "equalized_odds_scatter(df, 'Race/Ethnicity', 'Equalized odds by race/ethnicity subgroup')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-cm",
   "metadata": {},
   "source": ["## 5. Confusion matrices\n", "\n", "Per-model confusion matrices on the full 1,000-sample set."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-cm",
   "metadata": {},
   "outputs": [],
   "source": [
    "LABELS = ['toxic', 'non-toxic']\n",
    "models = df['model'].unique()\n",
    "\n",
    "for model in sorted(models):\n",
    "    mdf = df[df['model'] == model].drop_duplicates(subset=['text'])\n",
    "    cm = confusion_matrix(mdf['actual_label'], mdf['predicted_label'], labels=LABELS)\n",
    "    pct = cm.astype(float) / cm.sum()\n",
    "    annotations = [[f'{cm[i][j]}<br>({pct[i][j]:.0%})' for j in range(2)] for i in range(2)]\n",
    "    fig = ff.create_annotated_heatmap(\n",
    "        cm,\n",
    "        x=[f'pred: {l}' for l in LABELS],\n",
    "        y=[f'actual: {l}' for l in LABELS],\n",
    "        annotation_text=annotations,\n",
    "        colorscale='Blues',\n",
    "    )\n",
    "    fig.update_layout(\n",
    "        title=f'Confusion matrix — {model.split(\"/\")[-1]}',\n",
    "        xaxis_title='Predicted', yaxis_title='Actual',\n",
    "    )\n",
    "    fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-readme",
   "metadata": {},
   "source": ["## 6. Numbers for README key findings table\n", "\n", "Copy the output of this cell into the README table."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-readme-nums",
   "metadata": {},
   "outputs": [],
   "source": [
    "print('=== README key findings table ===\\n')\n",
    "print('| Model | Overall Accuracy | Gender Gap | Race Gap | Religion Gap |')\n",
    "print('|---|---|---|---|---|')\n",
    "\n",
    "ATTR_COLS = [('Gender', 'Gender'), ('Race/Ethnicity', 'Race'), ('Religion', 'Religion')]\n",
    "\n",
    "for model in sorted(df['model'].unique()):\n",
    "    mdf = df[df['model'] == model]\n",
    "    overall = (mdf['actual_label'] == mdf['predicted_label']).mean()\n",
    "    label = model.split('/')[-1]\n",
    "    gaps = []\n",
    "    for attr, _ in ATTR_COLS:\n",
    "        adf = mdf[mdf['protected_attribute'] == attr]\n",
    "        if len(adf) >= 20:\n",
    "            stats = group_stats(adf)\n",
    "            gap_pp = accuracy_gap(stats) * 100\n",
    "            gaps.append(f'{gap_pp:.0f} pp')\n",
    "        else:\n",
    "            gaps.append('n/a')\n",
    "    print(f'| {label} | {overall:.0%} | {gaps[0]} | {gaps[1]} | {gaps[2]} |')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-findings",
   "metadata": {},
   "source": ["## 7. Key findings\n", "\n", "Computed summary of the most important results."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cell-findings",
   "metadata": {},
   "outputs": [],
   "source": [
    "accs = {}\n",
    "gaps = {}\n",
    "for model, mdf in df.groupby('model'):\n",
    "    short = model.split('/')[-1]\n",
    "    accs[short] = (mdf['actual_label'] == mdf['predicted_label']).mean()\n",
    "    gender_df = mdf[mdf['protected_attribute'] == 'Gender']\n",
    "    if len(gender_df) >= 20:\n",
    "        gaps[short] = accuracy_gap(group_stats(gender_df))\n",
    "\n",
    "best_acc = max(accs, key=accs.get)\n",
    "most_fair = min(gaps, key=gaps.get) if gaps else 'n/a'\n",
    "\n",
    "print('KEY FINDINGS')\n",
    "print('============')\n",
    "print()\n",
    "print('Accuracy ranking:')\n",
    "for model, acc in sorted(accs.items(), key=lambda x: -x[1]):\n",
    "    print(f'  {model:30s} {acc:.1%}')\n",
    "print()\n",
    "print('Gender accuracy gap (smaller = fairer):')\n",
    "for model, gap in sorted(gaps.items(), key=lambda x: x[1]):\n",
    "    print(f'  {model:30s} {gap*100:.1f} pp')\n",
    "print()\n",
    "print(f'Most accurate model:       {best_acc} ({accs[best_acc]:.1%})')\n",
    "print(f'Most gender-fair model:    {most_fair} ({gaps.get(most_fair, 0)*100:.1f} pp gap)')\n",
    "\n",
    "# Bias amplification check: does the higher-accuracy model also have lower bias?\n",
    "if best_acc == most_fair:\n",
    "    print(f'\\n=> {best_acc} achieves both the best accuracy AND the smallest fairness gap.')\n",
    "else:\n",
    "    print(f'\\n=> Accuracy-fairness tradeoff: {best_acc} is most accurate but {most_fair} is most fair.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cell-md-limits",
   "metadata": {},
   "source": [
    "## 8. Limitations\n",
    "\n",
    "- **Sample size**: 1,000 rows split across 3 models and multiple subgroups. Some subgroups (e.g., Age) may have <20 samples, limiting statistical power.\n",
    "- **Protected attribute proxies**: HateXplain's target labels (e.g., \"women\", \"African\") are annotator-assigned proxies, not ground truth demographic identifiers.\n",
    "- **Prompt sensitivity**: Gemini and Claude are LLM-based classifiers whose output depends on prompt wording. Results reflect the prompts in `docs/prompt_design.md`.\n",
    "- **API drift**: Results reflect API behavior at time of evaluation. Model updates may change scores.\n",
    "- **Dataset domain**: HateXplain is a social media hate speech dataset. Performance may not generalize to other text domains."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Install nbconvert and execute the notebook**

```bash
pip install nbconvert jupyter
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=120 \
    notebooks/analysis.ipynb
```

Expected: the notebook file is updated in-place with all cell outputs populated.

- [ ] **Step 3: Spot-check the executed notebook**

```bash
python3 -c "
import json
nb = json.load(open('notebooks/analysis.ipynb'))
cells_with_output = sum(1 for c in nb['cells'] if c.get('outputs'))
print(f'Cells with outputs: {cells_with_output}')
"
```

Expected: 10+ cells with outputs.

- [ ] **Step 4: Commit the executed notebook**

```bash
git add notebooks/analysis.ipynb
git commit -m "$(cat <<'EOF'
feat: complete and execute analysis notebook with real benchmark results

All cells executed with 1,000-sample HateXplain benchmark results.
Includes: per-attribute accuracy bar charts, equalized odds scatters,
confusion matrices, and key findings summary.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Fill README key findings and fix demo section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Extract table numbers from executed notebook**

Read the output of `cell-readme-nums` from the executed notebook:

```bash
python3 -c "
import json
nb = json.load(open('notebooks/analysis.ipynb'))
for cell in nb['cells']:
    if cell.get('id') == 'cell-readme-nums':
        for out in cell.get('outputs', []):
            print(''.join(out.get('text', [])))
"
```

This prints the markdown table with real numbers. Copy them.

- [ ] **Step 2: Update README key findings table**

In `README.md`, replace the placeholder table (lines ~18–22) with the real numbers printed above. The table should look like:

```markdown
| Model | Overall Accuracy | Gender Gap | Race Gap | Religion Gap |
|---|---|---|---|---|
| perspective/... | XX% | X pp | X pp | X pp |
| gemini/... | XX% | X pp | X pp | X pp |
| claude/... | XX% | X pp | X pp | X pp |
```

Use the exact values from the notebook output. Update the column header from "Age Gap" to "Religion Gap" if Religion has more data than Age (check the notebook output to decide).

Also update the caption under the table from:
```
*"Gap" = max accuracy difference between any two subgroups within that
attribute. Smaller is fairer. Fill in Claude column after running the
benchmark.*
```
to:
```
*"Gap" = max accuracy difference between any two subgroups within that
attribute (95% bootstrap CI). Smaller is fairer.*
```

- [ ] **Step 3: Fix the demo link section**

In `README.md`, replace:
```markdown
> **Live demo →** [your-app.streamlit.app](https://your-app.streamlit.app)
> *(deploy instructions below)*
```
with:
```markdown
> **Run locally:** `streamlit run scripts/dashboard.py`
> **Deploy to Streamlit Cloud:** see [docs/deploy.md](docs/deploy.md) for 5-minute instructions.
```

- [ ] **Step 4: Update quickstart example to show all 3 models**

In `README.md` quickstart section, step 3 already says `--models perspective gemini` — update to `--models perspective gemini claude` to match the new default.

- [ ] **Step 5: Update the analysis.ipynb prerequisite comment in README**

Find where README says `--sample 200` or `--sample 500` and update to `--sample 1000`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: fill README key findings with real benchmark results

Replace TBD/placeholder values with actual numbers from 1,000-sample
HateXplain benchmark. Update demo link section and quickstart examples.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification and push

- [ ] **Step 1: Confirm zero placeholders remain**

```bash
grep -rn "TBD\|your-app\.streamlit\|yourusername\|your_.*_key_here\|TODO" \
    README.md scripts/dashboard.py notebooks/analysis.ipynb docs/deploy.md
```

Expected: no output (zero matches). `.env.example` intentionally contains `your_*_key_here` — that is correct and expected.

- [ ] **Step 2: Run tests one final time**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 3: Commit the design doc and plan**

```bash
git add docs/superpowers/
git commit -m "$(cat <<'EOF'
docs: add design spec and implementation plan

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 5: Verify CI passes**

```bash
gh run list --limit 3
```

Wait ~2 minutes, then:
```bash
gh run view --log-failed
```

Expected: all jobs green. If a job fails, check the log for the specific error (most likely a missing dep in CI).

- [ ] **Step 6: Check the README renders correctly on GitHub**

```bash
gh browse
```

Open the repo on GitHub and verify: (a) the key findings table shows real numbers, (b) the notebook renders with visible chart outputs, (c) no placeholder links remain.

---

## Definition of Done

- [ ] `git log --oneline` shows 8–10 commits on main, all meaningful
- [ ] `pytest tests/ -v` — all tests pass
- [ ] `grep -rn "TBD\|your-app\.streamlit\|yourusername" README.md` — empty
- [ ] `notebooks/analysis.ipynb` — 10+ cells with rendered outputs
- [ ] `docs/deploy.md` — exists, complete instructions
- [ ] `requirements.txt` — exists at repo root
- [ ] GitHub Actions CI — green badge
