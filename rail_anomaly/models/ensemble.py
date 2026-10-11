"""Score normalisation (empirical CDF on healthy data) and weighted ensemble."""
from __future__ import annotations

import numpy as np


class ScoreNormalizer:
    """Maps a raw model score to [0,1] using the healthy distribution as reference.

    z = (raw - p50) / (p99.5 - p50);  s = sigmoid(k · (z - 1))
    so a healthy-median window scores ≈0.05, the healthy 99.5th percentile scores 0.5,
    and scores keep increasing smoothly beyond the healthy range (unlike a plain CDF,
    which saturates at 1.0 and loses severity information).
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self.p50 = 0.0
        self.p995 = 1.0

    def fit(self, healthy_raw: np.ndarray) -> "ScoreNormalizer":
        r = np.asarray(healthy_raw, dtype=float)
        self.p50 = float(np.percentile(r, 50))
        self.p995 = float(np.percentile(r, 99.5))
        if self.p995 - self.p50 < 1e-9:
            self.p995 = self.p50 + 1e-9
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        z = (np.asarray(raw, dtype=float) - self.p50) / (self.p995 - self.p50)
        return 1.0 / (1.0 + np.exp(-self.k * (z - 1.0)))


class EnsembleScorer:
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def combine(self, scores: dict[str, float]) -> float:
        avail = {k: v for k, v in scores.items() if v is not None}
        if not avail:
            return 0.0
        w = {k: self.weights.get(k, 0.0) for k in avail}
        tot = sum(w.values()) or 1.0
        return float(sum(avail[k] * w[k] for k in avail) / tot)
