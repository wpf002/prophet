"""Prophet accuracy dashboard (Phase 6).

A small Streamlit page over the forecast-vs-actual store: rolling MASE-style
error and 95% interval coverage per series, plus the latest drift status.

Run:
    PROPHET_MONITOR_DSN=postgresql://... \\
        uv run --extra dashboard streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from prophet.config import settings
from prophet.monitoring.store import rolling_accuracy

st.set_page_config(page_title="Prophet — Accuracy", layout="wide")
st.title("Prophet — Forecast Accuracy")

dsn = settings.monitor_dsn
if not dsn:
    st.error("PROPHET_MONITOR_DSN is not set — no monitoring database to read.")
    st.stop()

window = st.slider("Rolling window (days)", min_value=7, max_value=90, value=30, step=1)

try:
    rows = rolling_accuracy(dsn, days=window)
except Exception as exc:  # surface connection/query errors in the UI
    st.error(f"Could not read monitoring data: {exc}")
    st.stop()

if not rows:
    st.info(
        f"No forecast/actual overlap in the last {window} days yet. "
        "Serve forecasts (POST /forecast) and run scripts/record_actuals.py to populate."
    )
    st.stop()

n_total = sum(int(r["n"]) for r in rows)
mean_cov = sum(float(r["coverage_95"]) for r in rows) / len(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Series scored", len(rows))
c2.metric("Forecast/actual pairs", n_total)
c3.metric("Mean 95% coverage", f"{mean_cov:.0%}")

st.subheader(f"Per-series accuracy (last {window} days)")
st.dataframe(
    [
        {
            "series": r["series_id"],
            "n": int(r["n"]),
            "MAE": round(float(r["mae"]), 1),
            "95% coverage": f"{float(r['coverage_95']):.0%}",
        }
        for r in rows
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Coverage near the nominal 95% means the intervals are calibrated. "
    "Sustained MAE growth vs. training is the drift signal (see prophet.monitoring.drift)."
)
