# Design: Benchmark Completion & Portfolio Polish

**Date:** 2026-04-04  
**Project:** toxicity-fairness-bench  
**Scope:** Take the project from scaffold to resume-ready state with no placeholders remaining.

---

## Goals

- All in-flight bug fixes committed and CI green
- Real benchmark results (1,000 samples, 3 models) in the repo
- Analysis notebook fully executed with rendered plots
- README key findings table filled with real numbers
- Streamlit dashboard verified and deployment-ready
- Zero placeholder text anywhere in the repo

---

## Constraints

- Perspective API: free tier at 1 QPS (rate limiter already in code)
- Gemini API: free tier (2s sleep between calls already in code)
- Anthropic: pay-as-you-go, Claude Haiku; minimize cost
- Default benchmark sample size: **1,000 rows** (not 20k)
- Streamlit deployment: user handles browser login; repo prepares everything else

---

## Step-by-Step Plan

### 1. Commit in-flight bug fixes
Files with unstaged changes:
- `pyproject.toml` — dep corrected to `google-genai>=1.0`
- `src/toxicity_fairness/analyzers/gemini.py` — migrated to `google.genai` client API
- `src/toxicity_fairness/data/loaders.py` — HateXplain label decoder (int→str)
- `scripts/run_benchmark.py` — positional merge fix for duplicate texts
- `.env.example` — restored (was accidentally deleted)

Commit these together as a single bug-fix commit.

### 2. Environment setup + test suite
- Create virtualenv, `pip install -e ".[dev]"`
- Run `pytest tests/ -v --cov=src`
- Fix any failures
- Push so GitHub Actions CI runs and badge goes green

### 3. Benchmark run (1,000 samples, all 3 models)
- Run `scripts/run_benchmark.py --sample 1000 --models perspective gemini claude`
- Output: `results/raw_results.parquet`
- The existing `ResultCache` prevents re-running on interruption
- Add `results/` to `.gitignore` (raw results are large; notebook outputs are the artifact)
- Update `README.md` default sample size reference from 20k → 1,000

### 4. Analysis notebook execution
- Fill out all skeleton cells in `notebooks/analysis.ipynb`:
  - Summary fairness report table
  - Accuracy-by-subgroup bar charts with 95% CI error bars (per attribute)
  - Equalized-odds scatter (TPR vs FPR) per model
  - Calibration curves
  - Written narrative (key finding per model, one paragraph each)
- Execute via `jupyter nbconvert --to notebook --execute`
- Commit with outputs rendered

### 5. README key findings
- Run `fairness_report()` on results to extract real numbers
- Fill in the Claude Sonnet row in the key findings table
- Replace placeholder demo URL with a Streamlit deploy callout

### 6. Streamlit prep + deploy instructions
- Run `streamlit run scripts/dashboard.py` locally against real results to verify
- Add `requirements.txt` (Streamlit Community Cloud needs it)
- Write `docs/deploy.md` with 5-step deploy instructions
- Update README demo link section with clear callout pointing to `docs/deploy.md`

---

## Approach: Notebook Execution

Use `nbconvert --execute` (approach A) so outputs are baked into the committed `.ipynb`. This makes plots visible directly on GitHub without any setup — what a recruiter actually sees.

---

## Definition of Done

- [ ] `git status` is clean on main
- [ ] CI badge is green
- [ ] `grep -r "TBD\|your-app\|placeholder" README.md` returns nothing
- [ ] `notebooks/analysis.ipynb` has rendered outputs in every cell
- [ ] `docs/deploy.md` exists with complete Streamlit deploy instructions
- [ ] `requirements.txt` exists at repo root
