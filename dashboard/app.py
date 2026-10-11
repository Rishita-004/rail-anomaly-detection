"""Streamlit dashboard: track health map, score trends, alert feed.

Run:  streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly import ROOT, load_settings
from rail_anomaly.storage import Store

st.set_page_config(page_title="Track Health Monitor", layout="wide")
cfg = load_settings()
store = Store(ROOT / cfg["paths"]["data"] / "pipeline.db")

runs = store.runs()
if not runs:
    st.warning("No runs found. Execute `python scripts/run_pipeline.py config/scenarios/crack_s12.yaml` first.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
sc = store.df("scores", run)
al = store.df("alerts", run)
ev = store.df("events", run)
show_truth = st.sidebar.checkbox("Show ground truth (evaluation mode)", value=False)

st.title("🚆 Track Health Monitor")
st.caption(f"Run **{run}** · {len(sc):,} windows · {len(ev)} events · {len(al)} alerts")

# ---------------- track map ----------------
st.subheader("Track health map")
latest = sc.sort_values("t_min").groupby("segment").tail(1).set_index("segment")
sev_color = {"low": "#2BB673", "medium": "#F2A900", "high": "#E04E39"}
cols = st.columns(len(latest))
for c, (seg, row) in zip(cols, latest.iterrows()):
    color = sev_color[row.severity] if row.in_event else "#2BB673"
    c.markdown(
        f"<div style='background:{color};color:white;padding:14px;border-radius:10px;text-align:center'>"
        f"<b style='font-size:20px'>{seg}</b><br>score {row.score:.2f}<br>"
        f"{'⚠ ' + row.severity.upper() if row.in_event else 'healthy'}</div>",
        unsafe_allow_html=True,
    )

# ---------------- segment trend ----------------
st.subheader("Segment trend")
seg = st.selectbox("Segment", sorted(sc.segment.unique()))
g = sc[sc.segment == seg].sort_values("t_min")
fig = go.Figure()
fig.add_trace(go.Scatter(x=g.t_min, y=g.score, name="ensemble score", line=dict(color="#E04E39", width=1.5)))
fig.add_trace(go.Scatter(x=g.t_min, y=g.theta, name="threshold θ", line=dict(color="#F2A900", dash="dash")))
if "s_lstm" in g and g.s_lstm.notna().any():
    fig.add_trace(go.Scatter(x=g.t_min, y=g.s_iforest, name="iforest", line=dict(color="#5B6B7F", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=g.t_min, y=g.s_lstm, name="lstm-ae", line=dict(color="#1B2430", width=1), opacity=0.5))
if show_truth:
    fig.add_trace(go.Scatter(x=g.t_min, y=g.true_severity, name="true severity", line=dict(color="#2BB673", dash="dot")))
for _, e in ev[ev.segment == seg].iterrows():
    fig.add_vrect(x0=e.t_open, x1=e.t_close if pd.notna(e.t_close) else g.t_min.max(), fillcolor="#E04E39", opacity=0.08, line_width=0)
fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="time (min)", yaxis_title="score", yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

# ---------------- alert feed ----------------
st.subheader("Alert feed")
if al.empty:
    st.info("No alerts in this run.")
else:
    for _, a in al.sort_values("t_min", ascending=False).iterrows():
        p = json.loads(a.payload)
        with st.expander(f"{a.severity.upper()} · {a.segment} · t={a.t_min:.1f} min · {a.fault_type.replace('_', ' ')} ({a.confidence:.2f})"):
            st.write(a.narrative)
            c1, c2 = st.columns(2)
            c1.markdown("**Top evidence**")
            c1.table(pd.DataFrame(p["evidence"]))
            c2.markdown("**Sensor contribution**")
            c2.bar_chart(pd.Series(p["sensor_contrib"]))
            if show_truth and "ground_truth" in p:
                st.caption(f"Ground truth: {p['ground_truth']['label']} @ severity {p['ground_truth']['severity']}")
            st.code(json.dumps(p, indent=2), language="json")
