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
   source .venv/bin/activate   # activate the project venv first
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

   Update the README demo link with your live URL once deployed.

## Notes

- `requirements.txt` at the repo root tells Streamlit Cloud what to install.
- The `results/` directory is gitignored — the live scorer (bottom of the dashboard) works without pre-computed results, but the charts require running the benchmark first.
- Rate limits: Perspective API has a 1 QPS free tier. The live scorer makes one call per model per click, well within limits.
