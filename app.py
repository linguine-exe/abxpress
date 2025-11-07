# app.py - ABxpress
# A clean, beginner friendly A/B test analyzer with auto detection, help text, AI summary, charts, PDF and extras.

import io
import os
import pandas as pd
import streamlit as st

from analysis import validate_and_cast, analyze
from components import timeseries_chart, dist_chart

# --- Data loader helpers (persist file in session state) ---
def _set_data_from_upload(upload_file):
    if upload_file is None:
        return
    st.session_state["data_bytes"] = upload_file.getvalue()
    st.session_state["data_name"] = getattr(upload_file, "name", "uploaded.csv")

def _set_data_from_sample():
    with open("sample_data.csv", "rb") as f:
        st.session_state["data_bytes"] = f.read()
        st.session_state["data_name"] = "sample_data.csv"

def _get_df(nrows=None):
    b = st.session_state.get("data_bytes", None)
    if not b:
        return None
    bio = io.BytesIO(b)
    return pd.read_csv(bio, nrows=nrows)


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="ABxpress", page_icon="📈", layout="wide")
st.title("ABxpress - AI assisted A/B test analyzer")

# ---------------------------
# Intro - what files are expected
# ---------------------------
st.markdown("""
**What this tool does**

ABxpress analyzes A/B test results where you have a control group and a treatment group, and each row represents a user or a session.

**Expected CSV format**
- One row per user or session
- At least these columns:
  - `variant` - which group a row belongs to, for example control or treatment
  - `metric` - the outcome you want to measure, for example `converted` or `revenue`
- Optional columns:
  - `date` - for time series charts
  - `segment` - for slicing, such as platform or country

**Example**

| user_id | variant   | converted | revenue | date       | platform |
|--------:|-----------|-----------|---------|------------|----------|
| 1       | control   | 0         | 0.00    | 2025-01-01 | PS5      |
| 2       | treatment | 1         | 5.99    | 2025-01-01 | PS5      |

Upload a file like this to see effect sizes, confidence intervals, p values, power, and charts.
""")

# ---------------------------
# Guided help and plain English explanations
# ---------------------------
st.divider()
st.header("Quick help")

with st.expander("What should I type in the sidebar boxes?"):
    st.markdown("""
**Variant column**  
A column that labels each row as control or treatment. Common names: `variant`, `group`, `treatment`, `bucket`.

**Metric column**  
The outcome you care about.  
- Binary yes or no: a 0 or 1 column such as `converted`, `clicked`, `purchased`  
- Numeric: a number such as `revenue`, `watch_time`, `amount`

**Metric type**  
- `binary` if the metric only has 0 and 1  
- `numeric` if it is any number

**Date column**  
Optional. A date or timestamp such as `date` or `timestamp`.

**Segment column**  
Optional. A label to slice by, such as `platform`, `country`, `device`, or `cohort`.
""")

with st.expander("What do these results mean? Plain English"):
    st.markdown("""
- **p value** - How surprising the observed difference would be if there is no real effect. Smaller is stronger evidence. A common cutoff is 0.05.  
- **95 percent CI for lift** - A range of plausible differences based on your data. If it includes 0, the result may be inconclusive.  
- **Power** - Probability the test would detect the effect if it is real. Many teams target 0.80 or higher.  
- **Absolute lift** - Treatment minus control.  
- **Relative lift** - Absolute lift divided by control, shown as a percent.
""")

# ---------------------------
# Sidebar - upload and settings (session-state aware)
# ---------------------------
with st.sidebar:
    st.header("Upload and settings")

    # 1) Upload or load sample, persist bytes in session state
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload:
        _set_data_from_upload(upload)

    st.caption("Quick demo")
    if st.button("Load sample data"):
        _set_data_from_sample()

    # 2) Experiment name
    exp_name = st.text_input("Experiment name", value="Hero Image Test")

    # 3) Build controls from a small preview of the data in session
    preview = _get_df(nrows=200)  # None if no data yet

    if preview is not None:
        cols = preview.columns.tolist()

        # Helpers to guess sensible defaults
        import pandas as _pd

        def _guess_variant(cs):
            preferred = ["variant","group","treatment","bucket","arm","ab_group","abgroup","testgroup"]
            lower = {c.lower(): c for c in cs}
            for p in preferred:
                if p in lower:
                    return lower[p]
            # fallback: a low-cardinality column
            for c in cs:
                try:
                    u = preview[c].dropna().astype(str).nunique()
                    if 2 <= u <= 3:
                        return c
                except Exception:
                    pass
            return cs[0] if cs else ""

        def _guess_metric(cs):
            preferred = ["converted","conversion","clicked","click","purchased","purchase",
                         "revenue","amount","value","watch_time","duration","time","score"]
            lower = {c.lower(): c for c in cs}
            for p in preferred:
                if p in lower:
                    return lower[p]
            for c in cs:
                try:
                    if _pd.api.types.is_numeric_dtype(preview[c]):
                        return c
                except Exception:
                    pass
            return cs[0] if cs else ""

        def _guess_date(cs):
            preferred = ["date","day","ds","timestamp","datetime","event_date"]
            lower = {c.lower(): c for c in cs}
            for p in preferred:
                if p in lower:
                    return lower[p]
            return ""

        def _guess_segment(cs):
            preferred = ["platform","country","device","segment","cohort","region","channel"]
            lower = {c.lower(): c for c in cs}
            for p in preferred:
                if p in lower:
                    return lower[p]
            return ""

        default_variant = _guess_variant(cols)
        default_metric  = _guess_metric(cols)
        default_date    = _guess_date(cols)
        default_segment = _guess_segment(cols)

        variant_col = st.selectbox(
            "Variant column",
            options=cols,
            index=(cols.index(default_variant) if default_variant in cols else 0),
        )
        metric_col = st.selectbox(
            "Metric column",
            options=cols,
            index=(cols.index(default_metric) if default_metric in cols else 0),
        )

        metric_type_choice = st.selectbox("Metric type", ["Auto-detect","binary","numeric"], index=0)

        date_options = ["(none)"] + cols
        date_col = st.selectbox(
            "Date column (optional)",
            options=date_options,
            index=(date_options.index(default_date) if default_date in cols else 0),
        )

        seg_options = ["(none)"] + cols
        segment_col = st.selectbox(
            "Segment column (optional)",
            options=seg_options,
            index=(seg_options.index(default_segment) if default_segment in cols else 0),
        )

        # Normalize “(none)” to blank strings so downstream logic is simple
        date_col = "" if date_col == "(none)" else date_col
        segment_col = "" if segment_col == "(none)" else segment_col

    else:
        # Fallback controls before any data is present
        variant_col = st.text_input("Variant column", value="variant")
        metric_col  = st.text_input("Metric column",  value="converted")
        metric_type_choice = st.selectbox("Metric type", ["binary","numeric"], index=0)
        date_col    = st.text_input("Date column (optional)", value="date")
        segment_col = st.text_input("Segment column (optional)", value="platform")

# Build a consistent df reference for the rest of the script
df_preview = _get_df(nrows=200)
df_full = _get_df()  # full dataset when needed


# ---------------------------
# Auto detect suggestions table on the main page
# ---------------------------
def _guess_columns_table(df: pd.DataFrame):
    import numpy as _np
    import pandas as _pd
    # reuse selected names as suggestions
    # and make a best effort check of metric type
    def _is_binary_series(s: pd.Series) -> bool:
        try:
            vals = pd.to_numeric(s.dropna(), errors="coerce").dropna().unique()
            return set(vals) <= {0, 1}
        except Exception:
            return False

    mt_guess = "binary" if metric_col in df and _is_binary_series(df[metric_col]) else "numeric"
    table = _pd.DataFrame({
        "Setting": ["Variant column", "Metric column", "Metric type", "Date column", "Segment column"],
        "Suggested": [variant_col or "(not sure)",
                      metric_col or "(not sure)",
                      mt_guess,
                      date_col or "(optional)",
                      segment_col or "(optional)"]
    })
    return table

st.subheader("Auto detect suggestions")
if df_full is not None:
    try:
        # Show suggestions based on the current selections and data
        df_preview = _get_df(nrows=200)
        df_preview.seek(0)
        st.table(_guess_columns_table(df_preview))
    except Exception:
        st.caption("Upload a valid CSV to see suggestions.")
else:
    st.caption("Upload a CSV to see suggestions here.")

# ---------------------------
# Stop early if no file yet
# ---------------------------
if df_full is None:
    st.info("Upload a CSV to begin, or click Load sample data in the sidebar.")
    st.stop()

# ---------------------------
# Read and validate, with metric type auto option
# ---------------------------
try:
    raw = df_full
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

def _is_binary_series_full(s: pd.Series) -> bool:
    try:
        vals = pd.to_numeric(s.dropna(), errors="coerce").dropna().unique()
        return set(vals) <= {0, 1}
    except Exception:
        return False

if metric_type_choice == "Auto-detect":
    metric_type = "binary" if _is_binary_series_full(raw[metric_col]) else "numeric"
else:
    metric_type = metric_type_choice

# Validate and cast
try:
    df = validate_and_cast(raw, variant_col, metric_col, metric_type)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------------------------
# Core analysis and KPIs
# ---------------------------
res = analyze(df, variant_col, metric_col, metric_type)

st.divider()
st.header("Results")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Control mean", f"{res['mean_c']:.4f}")
k2.metric("Treatment mean", f"{res['mean_t']:.4f}")
k3.metric("Absolute lift", f"{res['abs_lift']:.4f}")
k4.metric("p value", f"{res['p_value']:.3g}")

st.write(
    f"95 percent CI for lift: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]  |  "
    f"Power: {res['power']:.2%}  |  n_c={res['n_c']}  n_t={res['n_t']}"
)

# Charts
c1, c2 = st.columns(2)
fig_ts = timeseries_chart(df, date_col, metric_col, variant_col) if date_col else None
if fig_ts:
    c1.plotly_chart(fig_ts, use_container_width=True, key="plot_ts_main")
dist_fig_main = dist_chart(df, metric_col, variant_col)
c2.plotly_chart(dist_fig_main, use_container_width=True, key="plot_dist_main")

# ---------------------------
# Segment view
# ---------------------------
st.divider()
st.subheader("Segment view (optional)")

if segment_col and segment_col in df.columns:
    try:
        opts = ["All"] + sorted(df[segment_col].dropna().astype(str).unique().tolist())
    except Exception:
        opts = ["All"]
    seg_choice = st.selectbox("Choose a segment value", opts, index=0)

    df_view = df if seg_choice == "All" else df[df[segment_col].astype(str) == seg_choice]

    if df_view[variant_col].nunique() == 2 and len(df_view) >= 2:
        res_seg = analyze(df_view, variant_col, metric_col, metric_type)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Control mean (seg)", f"{res_seg['mean_c']:.4f}")
        s2.metric("Treatment mean (seg)", f"{res_seg['mean_t']:.4f}")
        s3.metric("Abs lift (seg)", f"{res_seg['abs_lift']:.4f}")
        s4.metric("p value (seg)", f"{res_seg['p_value']:.3g}")
        st.write(
            f"95 percent CI: [{res_seg['ci_low']:.4f}, {res_seg['ci_high']:.4f}]  |  "
            f"Power: {res_seg['power']:.2%}  |  n_c={res_seg['n_c']}  n_t={res_seg['n_t']}"
        )

        c3, c4 = st.columns(2)
        fig_ts2 = timeseries_chart(df_view, date_col, metric_col, variant_col) if date_col else None
        # Make a stable, unique key per segment choice
        _seg_key = str(seg_choice).replace(" ", "_")
        if fig_ts2:
            c3.plotly_chart(fig_ts2, use_container_width=True, key=f"plot_ts_seg_{_seg_key}")
        dist_fig_seg = dist_chart(df_view, metric_col, variant_col)
        c4.plotly_chart(dist_fig_seg, use_container_width=True, key=f"plot_dist_seg_{_seg_key}")

    else:
        st.info("Pick a segment that contains both control and treatment.")
else:
    st.caption("Tip: include a column like platform or country and set it as the Segment column in the sidebar.")

# ---------------------------
# Sample size calculator
# ---------------------------
st.divider()
st.subheader("Target sample size for binary metrics")

with st.expander("Open calculator"):
    import statsmodels.stats.api as sms
    p_base = st.number_input(
        "Baseline conversion rate (0 to 1)",
        min_value=0.0, max_value=1.0, value=float(res["mean_c"]) if 0 <= res["mean_c"] <= 1 else 0.10,
        step=0.01, format="%.2f"
    )
    mde_percent = st.number_input(
        "Minimum detectable relative lift (%)",
        min_value=0.1, max_value=100.0, value=5.0, step=0.5, format="%.1f"
    )
    alpha = st.number_input("Significance level alpha", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    power_target = st.number_input("Power", min_value=0.5, max_value=0.99, value=0.80, step=0.05, format="%.2f")

    if st.button("Compute required n per arm"):
        try:
            p_t = p_base * (1 + mde_percent / 100.0)
            effect = sms.proportion_effectsize(p_t, p_base)
            n_per = sms.NormalIndPower().solve_power(effect_size=effect, power=power_target, alpha=alpha, ratio=1)
            st.success(f"Recommended sample size per arm: {int(n_per + 0.5)}")
        except Exception as e:
            st.error(f"Could not compute: {e}")

# ---------------------------
# LinkedIn blurb
# ---------------------------
st.divider()
st.subheader("LinkedIn blurb")

rel_pct = ""
try:
    rel = res.get("rel_lift", None)
    if rel is not None and rel == rel:
        rel_pct = f"{rel*100:.1f}%"
except Exception:
    rel_pct = ""

blurb = f"""Shipped a small analytics tool: ABxpress. It turns A/B test CSVs into decision ready results with effect sizes, confidence intervals, power, and clear charts. It can also draft an executive summary with AI.

Demo run: n_c={res['n_c']}, n_t={res['n_t']}. Observed absolute lift = {res['abs_lift']:.4f}{(' (' + rel_pct + ' relative)') if rel_pct else ''}. 95 percent CI = [{res['ci_low']:.4f}, {res['ci_high']:.4f}]. p value = {res['p_value']:.3g}. Power = {res['power']:.2%}.

Built with Python, Streamlit, pandas, scipy, statsmodels, and plotly. Happy to share the template or adapt it for gaming or streaming experiments.
"""

st.text_area("Copy this:", blurb, height=160)
st.download_button("Download as blurb.txt", blurb.encode("utf-8"), file_name="abxpress_linkedin.txt")
