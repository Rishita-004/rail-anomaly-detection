"""Stage 3a — Per-window feature extraction.

Input : tensor [T, 5]  (accel_v, accel_l, strain, temp, acoustic)
Output: ~40 named features. Feature *groups* (used by the explainer / diagnoser):
  accel_hf, accel_mf, accel_lf, accel_crest, lateral, strain_peak, strain_trend,
  temp, temp_strain_corr, acoustic, jitter
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal, stats

FEATURE_GROUPS = {
    "accel_hf": ["accel_v_band_hf", "accel_v_centroid", "accel_v_entropy"],
    "accel_mf": ["accel_v_band_mf"],
    "accel_lf": ["accel_v_band_lf"],
    "accel_energy": ["accel_v_rms", "accel_v_peak", "accel_strain_ratio"],
    "accel_crest": ["accel_v_crest", "accel_v_kurtosis"],
    "lateral": ["accel_l_rms", "accel_l_peak", "accel_l_mean", "lat_vert_ratio"],
    "strain_peak": ["strain_peak", "strain_rms"],
    "strain_trend": ["strain_mean", "strain_gradient"],
    "temp": ["temp_mean", "temp_gradient"],
    "temp_strain_corr": ["temp_strain_corr"],
    "acoustic": ["acoustic_rms", "acoustic_peak", "acoustic_burst_rate", "acoustic_kurtosis"],
    "jitter": ["accel_v_skew", "accel_v_impulse_width"],
}
FEATURE_NAMES = [f for g in FEATURE_GROUPS.values() for f in g]


def _band_energy(x: np.ndarray, fs: int) -> tuple[float, float, float, float, float]:
    f, p = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    p = p + 1e-12
    total = p.sum()
    lf = p[f < 20].sum() / total
    mf = p[(f >= 20) & (f < 80)].sum() / total
    hf = p[f >= 80].sum() / total
    centroid = (f * p).sum() / total
    pn = p / total
    entropy = -(pn * np.log(pn)).sum()
    return lf, mf, hf, centroid, entropy


def _crest(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return float(np.max(np.abs(x)) / rms)


def _impulse_width(x: np.ndarray, fs: int) -> float:
    """Mean width (s) of regions where |x| > 3·std — widens with loose fastening."""
    thr = 3 * np.std(x) + 1e-9
    above = np.abs(x) > thr
    if not above.any():
        return 0.0
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0.0
    n = min(len(starts), len(ends))
    widths = (ends[:n] - starts[:n])
    widths = widths[widths > 0]
    return float(widths.mean() / fs) if len(widths) else 0.0


class FeatureExtractor:
    """Stateful per segment: keeps a rolling context for trend features."""

    def __init__(self, fs: int, context_windows: int = 120):
        self.fs = fs
        self._ctx: dict[str, deque] = {}
        self.context_windows = context_windows

    def _context(self, seg: str) -> deque:
        if seg not in self._ctx:
            self._ctx[seg] = deque(maxlen=self.context_windows)
        return self._ctx[seg]

    def extract(self, seg: str, x: np.ndarray) -> dict[str, float]:
        fs = self.fs
        av, al, st, te, ac = (x[:, i].astype(float) for i in range(5))
        f: dict[str, float] = {}

        lf, mf, hf, cen, ent = _band_energy(av, fs)
        f["accel_v_band_lf"], f["accel_v_band_mf"], f["accel_v_band_hf"] = lf, mf, hf
        f["accel_v_centroid"], f["accel_v_entropy"] = cen, ent
        f["accel_v_rms"] = float(np.sqrt(np.mean(av ** 2)))
        f["accel_v_peak"] = float(np.max(np.abs(av)))
        f["accel_v_crest"] = _crest(av)
        f["accel_v_kurtosis"] = float(stats.kurtosis(av))
        f["accel_v_skew"] = float(stats.skew(av))
        f["accel_v_impulse_width"] = _impulse_width(av, fs)

        f["accel_l_rms"] = float(np.sqrt(np.mean(al ** 2)))
        f["accel_l_peak"] = float(np.max(np.abs(al)))
        f["accel_l_mean"] = float(np.mean(al))
        f["lat_vert_ratio"] = f["accel_l_rms"] / (f["accel_v_rms"] + 1e-9)

        f["strain_peak"] = float(np.max(st))
        f["strain_rms"] = float(np.sqrt(np.mean(st ** 2)))
        f["strain_mean"] = float(np.mean(st))
        f["accel_strain_ratio"] = f["accel_v_rms"] / (f["strain_rms"] + 1e-9)

        f["temp_mean"] = float(np.mean(te))

        f["acoustic_rms"] = float(np.sqrt(np.mean(ac ** 2)))
        f["acoustic_peak"] = float(np.max(ac))
        f["acoustic_kurtosis"] = float(stats.kurtosis(ac))
        thr = np.mean(ac) + 4 * np.std(ac) + 1e-9
        f["acoustic_burst_rate"] = float((ac > thr).sum() / len(ac))

        # rolling-context trend features
        ctx = self._context(seg)
        ctx.append((f["strain_mean"], f["temp_mean"]))
        if len(ctx) >= 10:
            s_hist = np.array([c[0] for c in ctx])
            t_hist = np.array([c[1] for c in ctx])
            k = np.arange(len(s_hist))
            f["strain_gradient"] = float(np.polyfit(k, s_hist, 1)[0] * 100)
            f["temp_gradient"] = float(np.polyfit(k, t_hist, 1)[0] * 100)
            f["temp_strain_corr"] = float(np.corrcoef(s_hist, t_hist)[0, 1]) if np.std(t_hist) > 1e-6 and np.std(s_hist) > 1e-6 else 0.0
        else:
            f["strain_gradient"] = f["temp_gradient"] = f["temp_strain_corr"] = 0.0

        return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in f.items()}

    def vector(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)


def group_of(feature: str) -> str:
    for g, names in FEATURE_GROUPS.items():
        if feature in names:
            return g
    return "other"
