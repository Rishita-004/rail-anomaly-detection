"""Persistence: SQLite for scores / events / alerts (dashboard reads it);
Parquet for feature vectors (offline training & evaluation).

Swap for TimescaleDB in production by replacing the connection string.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (run TEXT, segment TEXT, t_min REAL, score REAL, theta REAL,
    flag INTEGER, in_event INTEGER, severity TEXT, label TEXT, true_severity REAL, s_iforest REAL, s_lstm REAL);
CREATE TABLE IF NOT EXISTS events (run TEXT, event_id TEXT, segment TEXT, t_open REAL, t_close REAL);
CREATE TABLE IF NOT EXISTS alerts (run TEXT, alert_id TEXT, event_id TEXT, segment TEXT, t_min REAL,
    score REAL, severity TEXT, fault_type TEXT, confidence REAL, trend TEXT, action TEXT, narrative TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_scores ON scores(run, segment, t_min);
"""


class Store:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self._buf: list[tuple] = []

    def score(self, run: str, r: dict, label: str, true_sev: float, s_if: float, s_lstm: float | None):
        self._buf.append((run, r["segment"], r["t_min"], r["score"], r["theta"], int(r["flag"]), int(r["in_event"]),
                          r["severity"], label, true_sev, s_if, s_lstm))
        if len(self._buf) >= 500:
            self.flush()

    def flush(self):
        if self._buf:
            self.conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", self._buf)
            self._buf.clear()
            self.conn.commit()

    def event_open(self, run: str, event_id: str, segment: str, t: float):
        self.conn.execute("INSERT INTO events VALUES (?,?,?,?,NULL)", (run, event_id, segment, t))

    def event_close(self, run: str, event_id: str, t: float):
        self.conn.execute("UPDATE events SET t_close=? WHERE run=? AND event_id=?", (t, run, event_id))

    def alert(self, run: str, a: dict):
        self.conn.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            run, a["alert_id"], a["event_id"], a["segment"], a["t_min"], a["score"], a["severity"],
            a["fault"]["type"], a["fault"]["confidence"], a["trend"], a["action"], a["narrative"], json.dumps(a)))
        self.conn.commit()

    def df(self, table: str, run: str | None = None) -> pd.DataFrame:
        q = f"SELECT * FROM {table}" + (" WHERE run=?" if run else "")
        return pd.read_sql(q, self.conn, params=(run,) if run else None)

    def runs(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT run FROM scores").fetchall()]

    def close(self):
        self.flush()
        self.conn.close()
