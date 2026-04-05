"""
Streamlit dashboard for toxicity fairness benchmark results.

Run locally:
    streamlit run scripts/dashboard.py

Deploy free:
    1. Push repo to GitHub
    2. Go to share.streamlit.io → connect repo → set main file to
       scripts/dashboard.py
    3. Add API keys in Streamlit Secrets
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toxicity_fairness.metrics.fairness import fairness_report, group_stats

st.set_page_config(
    page_title="Toxicity Fairness Benchmark",
    page_icon="⚖",
    layout="wide",
)

st.title("⚖ Toxicity Fairness Benchmark")
st.caption(
    "Comparing Perspective API, Gemini, and Claude across gender, race, "
    "and age using real-world toxicity datasets. "
    "[View source on GitHub](https://github.com/lydsleepy/toxicity-fairness-bench)"
)

RESULTS_PATH = Path("results/raw_results.parquet")


@st.cache_data
def load_results() -> pd.DataFrame | None:
    if RESULTS_PATH.exists():
        return pd.read_parquet(RESULTS_PATH)
    return None


df = load_results()

if df is None:
    st.warning(
        "No results found. Run the benchmark first:\n"
        "```\n"
        "python scripts/run_benchmark.py --sample 200 "
        "--models perspective gemini claude\n"
        "```"
    )
    st.stop()

with st.sidebar:
    st.header("Filters")
    available_models = sorted(df["model"].unique())
    selected_models  = st.multiselect("Models", available_models,
                                      default=available_models)
    available_attrs  = sorted(df["protected_attribute"].unique())
    selected_attr    = st.selectbox("Protected attribute", available_attrs)

filtered_df = df[
    (df["model"].isin(selected_models)) &
    (df["protected_attribute"] == selected_attr)
]

st.subheader("Overall accuracy")
cols = st.columns(len(selected_models))
for col, model in zip(cols, selected_models, strict=False):
    mdf = filtered_df[filtered_df["model"] == model]
    if len(mdf):
        acc = (mdf["actual_label"] == mdf["predicted_label"]).mean()
        col.metric(model.split("/")[-1], f"{acc:.1%}")

st.subheader(f"Accuracy by {selected_attr} group")
rows = []
for model in selected_models:
    mdf = filtered_df[filtered_df["model"] == model]
    for grp, gdf in mdf.groupby("attribute_value"):
        acc = (gdf["actual_label"] == gdf["predicted_label"]).mean()
        rows.append({"model": model.split("/")[-1], "group": grp,
                     "accuracy": acc, "n": len(gdf)})

if rows:
    bar_df = pd.DataFrame(rows)
    fig = px.bar(
        bar_df,
        x="group", y="accuracy", color="model", barmode="group",
        text=bar_df["accuracy"].map("{:.0%}".format),
        labels={"accuracy": "Accuracy", "group": selected_attr, "model": "Model"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Fairness metrics")
report = fairness_report(filtered_df)
st.dataframe(
    report.style.format({
        "overall_accuracy": "{:.1%}",
        "accuracy_gap":     "{:.1%}",
        "dp_gap":           "{:.1%}",
        "tpr_gap":          "{:.1%}",
        "fpr_gap":          "{:.1%}",
    }).background_gradient(
        subset=["accuracy_gap", "dp_gap", "tpr_gap"], cmap="RdYlGn_r"
    ),
    use_container_width=True,
)
st.caption(
    "**accuracy_gap** = max accuracy difference across subgroups. "
    "**dp_gap** = demographic parity gap. "
    "**tpr_gap / fpr_gap** = equalized odds gaps. Smaller = fairer."
)

st.subheader("False positive vs. false negative rate by group")
scatter_rows = []
for model in selected_models:
    mdf = filtered_df[filtered_df["model"] == model]
    stats = group_stats(mdf)
    for grp, row in stats.iterrows():
        scatter_rows.append({
            "model": model.split("/")[-1],
            "group": grp,
            "fpr":   row["fpr"],
            "fnr":   row["fnr"],
            "n":     row["n"],
        })

if scatter_rows:
    sc_df = pd.DataFrame(scatter_rows)
    fig2 = px.scatter(
        sc_df, x="fpr", y="fnr", color="model", text="group",
        size="n", size_max=30,
        labels={"fpr": "False Positive Rate", "fnr": "False Negative Rate"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_traces(textposition="top center")
    fig2.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", dash="dash")
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Points near the origin are better. Equal FPR and FNR across "
        "groups = equalized odds."
    )

st.divider()
st.subheader("Live scorer")
st.caption("Score any text with all available models in real time.")

user_text = st.text_area("Enter text to score",
                         placeholder="Type something here...")

if st.button("Score") and user_text.strip():
    live_cols = st.columns(3)
    model_configs = [
        ("Perspective", "toxicity_fairness.analyzers.perspective",
         "PerspectiveAnalyzer"),
        ("Gemini",      "toxicity_fairness.analyzers.gemini",
         "GeminiAnalyzer"),
        ("Claude",      "toxicity_fairness.analyzers.claude",
         "ClaudeAnalyzer"),
    ]
    for col, (label, module_path, class_name) in zip(
        live_cols, model_configs, strict=False
    ):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            analyzer = getattr(mod, class_name)()
            result = analyzer.analyze_one(user_text)
            score_pct = f"{result.score:.0%}" if result.score is not None else "n/a"
            icon = "🔴" if result.label == "toxic" else "🟢"
            col.metric(label, f"{icon} {score_pct}", result.label)
        except Exception as e:
            col.error(f"{label}: {e}")

st.divider()
st.caption(
    "Datasets: [HateXplain](https://huggingface.co/datasets/hatexplain) · "
    "MIT License"
)
