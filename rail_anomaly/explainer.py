"""Stage 4 — Attribution.

Uses SHAP TreeExplainer on the Isolation Forest (exact, fast). If shap is not
installed, falls back to |z-score| attribution, which is a reasonable proxy for
an isolation-based model.

Per-channel reconstruction error from the LSTM-AE (if present) is blended in as
a sensor-level cross-check.

Output: contributions per feature (sum to 1), per group, per sensor.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, group_of

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False

SENSOR_OF_FEATURE = {
    **{f: "accel_v" for f in FEATURE_NAMES if f.startswith("accel_v") or f == "accel_strain_ratio"},
    **{f: "accel_l" for f in FEATURE_NAMES if f.startswith("accel_l") or f == "lat_vert_ratio"},
    **{f: "strain" for f in FEATURE_NAMES if f.startswith("strain")},
    **{f: "temp" for f in FEATURE_NAMES if f.startswith("temp")},
    **{f: "acoustic" for f in FEATURE_NAMES if f.startswith("acoustic")},
}
CHANNEL_NAMES = ["accel_v", "accel_l", "strain", "temp", "acoustic"]


class Explainer:
    def __init__(self, iforest, background: np.ndarray | None = None):
        self.iforest = iforest
        self._tree = None
        if SHAP_AVAILABLE:
            try:
                self._tree = shap.TreeExplainer(iforest.model)
            except Exception:
                self._tree = None

    def feature_contrib(self, x_raw: np.ndarray) -> np.ndarray:
        """Returns |contribution| per feature, L1-normalised."""
        xs = self.iforest.transform(x_raw.reshape(1, -1))
        if self._tree is not None:
            phi = self._tree.shap_values(xs)
            phi = np.asarray(phi).reshape(-1)
            # IsolationForest SHAP: negative = more anomalous → use magnitude
            c = np.abs(phi)
        else:
            c = np.abs(xs.reshape(-1))
        return c / (c.sum() + 1e-12)

    def explain(self, x_raw: np.ndarray, channel_error: np.ndarray | None = None) -> dict:
        c = self.feature_contrib(x_raw)
        feats = {n: float(c[i]) for i, n in enumerate(FEATURE_NAMES)}
        groups: dict[str, float] = {}
        sensors: dict[str, float] = {}
        for n, v in feats.items():
            groups[group_of(n)] = groups.get(group_of(n), 0.0) + v
            s = SENSOR_OF_FEATURE.get(n, "other")
            sensors[s] = sensors.get(s, 0.0) + v
        if channel_error is not None:
            ce = np.asarray(channel_error, dtype=float)
            ce = ce / (ce.sum() + 1e-12)
            for i, ch in enumerate(CHANNEL_NAMES):
                sensors[ch] = 0.5 * sensors.get(ch, 0.0) + 0.5 * float(ce[i])
        top = sorted(feats.items(), key=lambda kv: -kv[1])[:5]
        return {
            "features": feats,
            "groups": groups,
            "sensors": sensors,
            "top_features": [{"feature": k, "contrib": round(v, 4)} for k, v in top],
            "z": {n: float(v) for n, v in zip(FEATURE_NAMES, self.iforest.transform(x_raw.reshape(1, -1)).reshape(-1))},
        }
