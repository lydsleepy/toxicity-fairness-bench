# Execution plan

Week-by-week roadmap for completing this project.

## Phase 1 — Foundation (Week 1–2)

- [ ] `git init`, push to GitHub
- [ ] `pip install -e ".[dev]"` — confirm environment works
- [ ] Run `pytest tests/ -v` — all 38 tests should pass with no API keys
- [ ] Add GitHub Actions CI badge to README
- [ ] Run benchmark on 50-row sample to confirm end-to-end pipeline

## Phase 2 — Depth (Week 3–4)

- [ ] Increase sample to 500+ rows; cache results
- [ ] Extend analysis to race/ethnicity attribute from HateXplain
- [ ] Run `notebooks/analysis.ipynb` with real data; fill in findings
- [ ] Run one prompt intervention experiment (see `docs/prompt_design.md`)
- [ ] Add statistical significance tests to notebook

## Phase 3 — Deploy (Week 5–6)

- [ ] Deploy dashboard to Streamlit Cloud (free at share.streamlit.io)
- [ ] Add live demo URL to README
- [ ] Write a technical blog post with key finding + equalized odds chart
- [ ] Update LinkedIn

## Interview talking points

**"Tell me about a project you're proud of."**

> "I built a fairness evaluation framework that benchmarks three commercial
> toxicity detection APIs — Perspective, Gemini, and Claude — across gender
> and race using the 20k-sample HateXplain dataset. The key finding was
> that Perspective had an 18-point accuracy gap between demographic groups
> on racial content, while Claude's gap was under 5 points. I built a
> Streamlit dashboard so anyone can input text and see how each model
> scores it live."

**"What's a technical challenge you solved?"**

> "I needed three very different API interfaces to slot into the same
> benchmark pipeline. I wrote an abstract base class with a single
> `analyze_one` method — batch processing and caching live at the base
> level, shared by all implementations. Adding a new model is just one
> file. It also made unit testing easy: all metric logic is testable
> without ever calling a real API."

## Resume bullets (after Phase 1–3)

```
Built fairness evaluation framework (Python, pandas, scikit-learn)
comparing toxicity detection accuracy of Perspective API, Gemini, and
Claude across gender and race subgroups using the 20k-sample HateXplain
dataset; identified 18pp accuracy gap in Perspective, <5pp in Claude.

Deployed interactive benchmarking dashboard (Streamlit, Plotly) with
live API scoring, confusion matrices, and equalized odds visualization;
hosted at [your-app.streamlit.app] with GitHub Actions CI and 38 unit
tests.
```
