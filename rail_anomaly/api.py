"""FastAPI alert service: query / acknowledge alerts from the store.

Run:  uvicorn rail_anomaly.api:app --reload
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from . import ROOT, load_settings
from .storage import Store

app = FastAPI(title="Rail Anomaly Alert API", version="0.1")
_cfg = load_settings()


def _store() -> Store:
    return Store(ROOT / _cfg["paths"]["data"] / "pipeline.db")


@app.get("/runs")
def runs():
    s = _store(); r = s.runs(); s.close(); return r


@app.get("/alerts")
def alerts(run: str | None = None, severity: str | None = None):
    s = _store(); df = s.df("alerts", run); s.close()
    if severity:
        df = df[df.severity == severity]
    return [json.loads(p) for p in df.sort_values("t_min", ascending=False).payload]


@app.get("/alerts/{alert_id}")
def alert(alert_id: str):
    s = _store(); df = s.df("alerts"); s.close()
    row = df[df.alert_id == alert_id]
    if row.empty:
        raise HTTPException(404, "alert not found")
    return json.loads(row.iloc[0].payload)


@app.get("/segments/{segment}/scores")
def scores(segment: str, run: str, last_n: int = 500):
    s = _store(); df = s.df("scores", run); s.close()
    g = df[df.segment == segment].sort_values("t_min").tail(last_n)
    return g[["t_min", "score", "theta", "flag", "in_event", "severity"]].to_dict(orient="records")
