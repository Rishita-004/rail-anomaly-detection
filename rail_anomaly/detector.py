"""Stage 3b — Scoring + decision logic.

Per window:
  raw_iforest, raw_lstm  ->  normalised [0,1]  ->  ensemble score s
Per segment state machine:
  event OPEN  when s > θ for persistence_k consecutive windows
  event CLOSE when s < θ for clear_k consecutive windows
θ = percentile of healthy ensemble scores (fit at training time; may be refreshed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentState:
    above: int = 0
    below: int = 0
    open: bool = False
    event_id: str | None = None
    history: list = field(default_factory=list)  # (t_min, score)


class Detector:
    def __init__(self, threshold: float, persistence_k: int = 4, clear_k: int = 20,
                 bands: dict | None = None):
        self.theta = threshold
        self.k = persistence_k
        self.clear_k = clear_k
        self.bands = bands or {"low": 0.6, "medium": 0.85}
        self._state: dict[str, SegmentState] = {}
        self._counter = 0

    def severity(self, s: float) -> str:
        if s < self.bands["low"]:
            return "low"
        if s < self.bands["medium"]:
            return "medium"
        return "high"

    def trend(self, seg: str, lookback_min: float = 360) -> str:
        h = self._state[seg].history
        if len(h) < 20:
            return "insufficient"
        t0 = h[-1][0] - lookback_min
        pts = [(t, s) for t, s in h if t >= t0]
        if len(pts) < 20:
            pts = h[-20:]
        t = np.array([p[0] for p in pts]); s = np.array([p[1] for p in pts])
        slope = np.polyfit(t, s, 1)[0] * 60  # per hour
        if slope > 0.05:
            return "rising"
        if slope < -0.05:
            return "falling"
        return "plateau" if np.mean(s) > self.theta else "transient"

    def step(self, seg: str, t_min: float, score: float) -> dict:
        st = self._state.setdefault(seg, SegmentState())
        st.history.append((t_min, score))
        if len(st.history) > 5000:
            st.history = st.history[-5000:]
        transition = None
        if score > self.theta:
            st.above += 1; st.below = 0
            if not st.open and st.above >= self.k:
                st.open = True
                self._counter += 1
                st.event_id = f"ev-{self._counter:05d}"
                transition = "open"
        else:
            st.below += 1; st.above = 0
            if st.open and st.below >= self.clear_k:
                st.open = False
                transition = "close"
        return {
            "segment": seg, "t_min": t_min, "score": round(float(score), 4),
            "theta": round(self.theta, 4), "flag": bool(score > self.theta),
            "in_event": st.open, "event_id": st.event_id if st.open or transition == "close" else None,
            "transition": transition, "severity": self.severity(score),
        }


def fit_threshold(healthy_scores: np.ndarray, percentile: float = 99.5) -> float:
    return float(np.percentile(healthy_scores, percentile))
