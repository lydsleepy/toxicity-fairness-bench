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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Liquid-glass theme: pastel pinks, greys, blues ────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

:root {
    --glass          : rgba(255, 255, 255, 0.75);
    --glass-border   : rgba(255, 255, 255, 0.55);
    --glass-shadow   : 0 2px 20px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(0,0,0,0.04);
    --radius         : 18px;
    --accent         : #8AAFC4;
    --text-primary   : #2c2c2e;
    --text-secondary : #6e6e73;
    --text-tertiary  : #98989d;
    --font           : 'DM Sans', -apple-system, BlinkMacSystemFont,
                       'Helvetica Neue', sans-serif;
}

/* ── strip only decorative chrome, never hide interactive controls ── */
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── hide sidebar toggle — sidebar is always visible ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsed"] {
    display: none !important;
}

/* ── app background: pastel blobs ── */
.stApp {
    background:
        radial-gradient(ellipse 65% 50% at 10% 18%,
            rgba(138,175,196,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 55% 60% at 90% 82%,
            rgba(212,168,184,0.16) 0%, transparent 65%),
        radial-gradient(ellipse 75% 40% at 50% 108%,
            rgba(196,184,208,0.14) 0%, transparent 60%),
        #f5f5f7;
    font-family           : var(--font) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── global typography ── */
*, *::before, *::after {
    font-family   : var(--font) !important;
    letter-spacing: -0.01em;
}

/* ── main content padding ── */
[data-testid="block-container"] {
    padding   : 2rem 2.75rem 4rem !important;
    max-width : 1400px;
}

/* ── sidebar: solid pastel panel (avoid invisible-via-transparency bug) ── */
section[data-testid="stSidebar"] {
    background   : #f0eef6 !important;
    border-right : 0.5px solid rgba(138,175,196,0.22) !important;
    box-shadow   : 2px 0 16px rgba(0,0,0,0.05) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.75rem !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size      : 0.65rem !important;
    font-weight    : 600 !important;
    text-transform : uppercase !important;
    letter-spacing : 0.1em !important;
    color          : var(--text-tertiary) !important;
    margin-bottom  : 0.85rem !important;
}

/* ── headings ── */
h1 {
    font-size      : 1.85rem !important;
    font-weight    : 600 !important;
    color          : var(--text-primary) !important;
    letter-spacing : -0.045em !important;
    line-height    : 1.1 !important;
    margin-bottom  : 0.1rem !important;
}

h2, h3 {
    font-size      : 0.9rem !important;
    font-weight    : 500 !important;
    color          : var(--text-secondary) !important;
    letter-spacing : -0.02em !important;
    margin-top     : 2rem !important;
    margin-bottom  : 0.65rem !important;
    padding-bottom : 0.45rem !important;
    border-bottom  : 0.5px solid rgba(0,0,0,0.07) !important;
}

/* ── captions ── */
.stCaption p,
[data-testid="stCaptionContainer"] p {
    color          : var(--text-tertiary) !important;
    font-size      : 0.76rem !important;
    font-weight    : 400 !important;
    letter-spacing : -0.005em !important;
}

/* ── links ── */
a { color: var(--accent) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }

/* ── metric tiles ── */
[data-testid="metric-container"] {
    background              : var(--glass) !important;
    backdrop-filter         : blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter : blur(20px) saturate(1.5) !important;
    border                  : 0.5px solid var(--glass-border) !important;
    border-radius           : var(--radius) !important;
    box-shadow              : var(--glass-shadow) !important;
    padding                 : 1.2rem 1.5rem !important;
    transition              : transform 0.18s ease, box-shadow 0.18s ease !important;
}

[data-testid="metric-container"]:hover {
    transform  : translateY(-2px) !important;
    box-shadow : 0 8px 32px rgba(0,0,0,0.08),
                 0 0 0 0.5px rgba(0,0,0,0.04) !important;
}

[data-testid="metric-container"] label {
    font-size      : 0.68rem !important;
    font-weight    : 600 !important;
    text-transform : uppercase !important;
    letter-spacing : 0.08em !important;
    color          : var(--text-tertiary) !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size      : 2rem !important;
    font-weight    : 300 !important;
    color          : var(--text-primary) !important;
    letter-spacing : -0.045em !important;
    line-height    : 1.1 !important;
}

/* ── plotly chart containers ── */
[data-testid="stPlotlyChart"] > div {
    background              : var(--glass) !important;
    backdrop-filter         : blur(20px) !important;
    -webkit-backdrop-filter : blur(20px) !important;
    border                  : 0.5px solid var(--glass-border) !important;
    border-radius           : var(--radius) !important;
    box-shadow              : var(--glass-shadow) !important;
    padding                 : 0.75rem 0.5rem 0.25rem !important;
    overflow                : hidden !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"] {
    background              : var(--glass) !important;
    backdrop-filter         : blur(20px) !important;
    -webkit-backdrop-filter : blur(20px) !important;
    border                  : 0.5px solid var(--glass-border) !important;
    border-radius           : var(--radius) !important;
    box-shadow              : var(--glass-shadow) !important;
    overflow                : hidden !important;
}

/* ── button: muted pill ── */
.stButton > button {
    background    : var(--accent) !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : 980px !important;
    padding       : 0.52rem 1.75rem !important;
    font-size     : 0.875rem !important;
    font-weight   : 500 !important;
    letter-spacing: -0.01em !important;
    cursor        : pointer !important;
    transition    : opacity 0.15s ease, transform 0.15s ease !important;
    box-shadow    : 0 1px 8px rgba(138,175,196,0.40) !important;
}

.stButton > button:hover {
    opacity   : 0.84 !important;
    transform : translateY(-1px) !important;
}

.stButton > button:active {
    transform : scale(0.97) translateY(0) !important;
    opacity   : 1 !important;
}

/* ── text area ── */
.stTextArea textarea {
    background    : rgba(255,255,255,0.92) !important;
    border        : 0.5px solid rgba(0,0,0,0.11) !important;
    border-radius : 12px !important;
    font-size     : 0.9rem !important;
    color         : var(--text-primary) !important;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.03) inset !important;
    transition    : border-color 0.15s ease, box-shadow 0.15s ease !important;
    padding       : 0.75rem 1rem !important;
    resize        : vertical !important;
}

.stTextArea textarea:focus {
    border-color : var(--accent) !important;
    box-shadow   : 0 0 0 3px rgba(138,175,196,0.18) !important;
    outline      : none !important;
}

.stTextArea label {
    font-size      : 0.76rem !important;
    font-weight    : 500 !important;
    color          : var(--text-secondary) !important;
}

/* ── selectbox ── */
.stSelectbox > div > div {
    background    : rgba(255,255,255,0.92) !important;
    border        : 0.5px solid rgba(0,0,0,0.11) !important;
    border-radius : 10px !important;
    font-size     : 0.875rem !important;
}

/* ── checkboxes ── */
.stCheckbox label {
    font-size   : 0.875rem !important;
    color       : var(--text-secondary) !important;
    font-weight : 400 !important;
}

/* ── divider ── */
hr {
    border     : none !important;
    border-top : 0.5px solid rgba(0,0,0,0.08) !important;
    margin     : 1.75rem 0 !important;
}

/* ── alert / warning ── */
[data-testid="stAlert"] {
    background    : rgba(212,168,184,0.12) !important;
    border        : 0.5px solid rgba(212,168,184,0.35) !important;
    border-radius : 12px !important;
    color         : var(--text-primary) !important;
}

/* ── column spacing ── */
[data-testid="column"] {
    padding: 0 0.35rem !important;
}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ── Pastel chart palette: pinks, greys, blues ─────────────────────────────────
_CHART_COLORS = [
    "#8AAFC4",  # dusty blue     (Perspective)
    "#C4899A",  # muted rose     (Claude)
    "#A89BB8",  # soft lavender
    "#7FA8BC",  # powder blue
    "#B89AA8",  # warm mauve
    "#9AADB8",  # blue-grey
]

_FONT = "DM Sans, -apple-system, BlinkMacSystemFont, Helvetica Neue, sans-serif"


def _chart_layout(**overrides) -> dict:
    """Shared Plotly layout — transparent background, minimal axes."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color="#2c2c2e", size=12),
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.04)",
            linecolor="rgba(0,0,0,0.07)",
            tickfont=dict(size=11, color="#98989d"),
            title_font=dict(size=11, color="#6e6e73"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.04)",
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color="#98989d"),
            title_font=dict(size=11, color="#6e6e73"),
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.50)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=0.5,
            font=dict(size=11, color="#6e6e73"),
        ),
    )
    base.update(overrides)
    return base


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1>Toxicity Fairness Benchmark</h1>
    <p style="color:#98989d;font-size:0.86rem;margin-top:0.2rem;
              margin-bottom:0;font-weight:400;letter-spacing:-0.01em;">
      Perspective API&nbsp;&nbsp;·&nbsp;&nbsp;Anthropic Claude&nbsp;&nbsp;·&nbsp;&nbsp;
      HateXplain dataset&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/lydsleepy/toxicity-fairness-bench"
         style="color:#8AAFC4;">GitHub ↗</a>
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
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
        "python scripts/run_benchmark.py --sample 1000 "
        "--models perspective claude\n"
        "```"
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    available_models = sorted(df["model"].unique())
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.1em;color:#98989d;margin-bottom:0.5rem;'>Models</p>",
        unsafe_allow_html=True,
    )
    selected_models = [
        model
        for model in available_models
        if st.checkbox(model.split("/")[-1], value=True, key=f"chk_{model}")
    ]

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    available_attrs = sorted(df["protected_attribute"].unique())
    selected_attr = st.selectbox("Protected attribute", available_attrs)

filtered_df = df[
    (df["model"].isin(selected_models))
    & (df["protected_attribute"] == selected_attr)
]

# ── Overall accuracy tiles ────────────────────────────────────────────────────
st.subheader("Overall accuracy")
if selected_models:
    cols = st.columns(len(selected_models))
    for col, model in zip(cols, selected_models, strict=False):
        mdf = filtered_df[filtered_df["model"] == model]
        if len(mdf):
            acc = (mdf["actual_label"] == mdf["predicted_label"]).mean()
            col.metric(model.split("/")[-1], f"{acc:.1%}")

# ── Accuracy by group ─────────────────────────────────────────────────────────
st.subheader(f"Accuracy by {selected_attr} group")
rows = []
for model in selected_models:
    mdf = filtered_df[filtered_df["model"] == model]
    for grp, gdf in mdf.groupby("attribute_value"):
        acc = (gdf["actual_label"] == gdf["predicted_label"]).mean()
        rows.append(
            {
                "model": model.split("/")[-1],
                "group": grp,
                "accuracy": acc,
                "n": len(gdf),
            }
        )

if rows:
    bar_df = pd.DataFrame(rows)
    fig = px.bar(
        bar_df,
        x="group",
        y="accuracy",
        color="model",
        barmode="group",
        text=bar_df["accuracy"].map("{:.0%}".format),
        labels={"accuracy": "Accuracy", "group": selected_attr, "model": "Model"},
        color_discrete_sequence=_CHART_COLORS,
        height=280,
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=10, color="#98989d"),
        marker_line_width=0,
        opacity=0.85,
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.12],
        bargap=0.24,
        bargroupgap=0.06,
        **_chart_layout(),
    )
    st.plotly_chart(fig, width="stretch")

# ── Fairness metrics table ────────────────────────────────────────────────────
st.subheader("Fairness metrics")
report = fairness_report(filtered_df)
st.dataframe(
    report.style.format(
        {
            "overall_accuracy": "{:.1%}",
            "accuracy_gap": "{:.1%}",
            "dp_gap": "{:.1%}",
            "tpr_gap": "{:.1%}",
            "fpr_gap": "{:.1%}",
        }
    ).background_gradient(
        subset=["accuracy_gap", "dp_gap", "tpr_gap"], cmap="RdPu"
    ),
    width="stretch",
)
st.caption(
    "**accuracy_gap** = max accuracy difference across subgroups.  "
    "**dp_gap** = demographic parity gap.  "
    "**tpr_gap / fpr_gap** = equalized odds gaps.  Smaller is fairer."
)

# ── FPR vs FNR scatter ────────────────────────────────────────────────────────
st.subheader("False positive vs. false negative rate by group")
scatter_rows = []
for model in selected_models:
    mdf = filtered_df[filtered_df["model"] == model]
    stats = group_stats(mdf)
    for grp, row in stats.iterrows():
        scatter_rows.append(
            {
                "model": model.split("/")[-1],
                "group": grp,
                "fpr": row["fpr"],
                "fnr": row["fnr"],
                "n": row["n"],
            }
        )

if scatter_rows:
    sc_df = pd.DataFrame(scatter_rows)
    fig2 = px.scatter(
        sc_df,
        x="fpr",
        y="fnr",
        color="model",
        text="group",
        size="n",
        size_max=26,
        labels={"fpr": "False Positive Rate", "fnr": "False Negative Rate"},
        color_discrete_sequence=_CHART_COLORS,
        height=320,
    )
    fig2.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="#98989d"),
        marker=dict(line=dict(width=0)),
        opacity=0.82,
    )
    fig2.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(0,0,0,0.14)", dash="dot", width=1),
    )
    fig2.update_layout(**_chart_layout())
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "Points near the origin are better. "
        "Equal FPR and FNR across groups = equalized odds."
    )

# ── Live scorer ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Live scorer")
st.caption("Score any text with both models in real time.")

user_text = st.text_area("Text to score", placeholder="Enter any text…", height=100)

if st.button("Score") and user_text.strip():
    live_cols = st.columns(3)
    model_configs = [
        (
            "Perspective",
            "toxicity_fairness.analyzers.perspective",
            "PerspectiveAnalyzer",
        ),
        ("Gemini", "toxicity_fairness.analyzers.gemini", "GeminiAnalyzer"),
        ("Claude", "toxicity_fairness.analyzers.claude", "ClaudeAnalyzer"),
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: [HateXplain](https://huggingface.co/datasets/hatexplain) (CC BY 4.0)"
    "  ·  MIT License"
)
