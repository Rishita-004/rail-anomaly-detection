"""Evaluate a stored run against ground truth.

Metrics: recall on injected faults, false-alarm rate on healthy segment-hours,
detection delay, explanation fidelity (diagnosed fault == injected fault).

Usage: python scripts/evaluate.py crack_s12   (run name = scenario 'name')
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly import ROOT, load_settings
from rail_anomaly.storage import Store


def evaluate(run: str, sev_cut: float = 0.5) -> dict:
    cfg = load_settings()
    st = Store(ROOT / cfg["paths"]["data"] / "pipeline.db")
    sc, ev, al = st.df("scores", run), st.df("events", run), st.df("alerts", run)
    st.close()
    if sc.empty:
        raise SystemExit(f"no run named '{run}' in database")

    out = {"run": run, "windows": len(sc)}
    # --- fault episodes: one per (segment, label != healthy)
    episodes = []
    for (seg, lab), g in sc[sc.label != "healthy"].groupby(["segment", "label"]):
        onset = g.t_min.min()
        t_half = g[g.true_severity >= sev_cut].t_min.min()
        e_seg = ev[ev.segment == seg]
        e_after = e_seg[e_seg.t_open >= onset - 0.5]
        detected_any = not e_after.empty
        # "early" = event opened before severity reached sev_cut (+2 min tolerance for step faults)
        detected = detected_any and (np.isnan(t_half) or e_after.t_open.min() <= t_half + 2.0)
        delay = float(e_after.t_open.min() - onset) if not e_after.empty else None
        a_seg = al[(al.segment == seg) & (al.t_min >= onset)]
        fidelity = float((a_seg.fault_type == lab).mean()) if not a_seg.empty else None
        episodes.append({"segment": seg, "fault": lab, "onset_min": float(onset), "detected": bool(detected_any),
                         "detected_before_sev": bool(detected),
                         "delay_min": delay, "explanation_fidelity": fidelity})
    out["episodes"] = episodes
    if episodes:
        out["recall"] = float(np.mean([e["detected"] for e in episodes]))
        out["early_recall"] = float(np.mean([e["detected_before_sev"] for e in episodes]))
        d = [e["delay_min"] for e in episodes if e["delay_min"] is not None]
        out["mean_delay_min"] = float(np.mean(d)) if d else None
        f = [e["explanation_fidelity"] for e in episodes if e["explanation_fidelity"] is not None]
        out["explanation_fidelity"] = float(np.mean(f)) if f else None
    # --- false alarms: events opened while segment was healthy
    fa = 0
    for _, e in ev.iterrows():
        lab = sc[(sc.segment == e.segment) & (np.isclose(sc.t_min, e.t_open))].label
        if not lab.empty and lab.iloc[0] == "healthy":
            fa += 1
    healthy_hours = (sc.label == "healthy").sum() * cfg["sampling"]["stride_sec"] / 3600
    out["false_alarms"] = int(fa)
    out["false_alarm_rate_per_segment_hour"] = float(fa / healthy_hours) if healthy_hours else None
    out["events_total"] = int(len(ev))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); a = ap.parse_args()
    print(json.dumps(evaluate(a.run), indent=2))
