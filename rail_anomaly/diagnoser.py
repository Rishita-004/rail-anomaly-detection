"""Stage 5 — Diagnosis + alert construction.

Compares the group-level attribution vector to fault prototypes (cosine similarity),
applies rule gates on z-scored features, and returns fault type + confidence.
Then renders the diagnostic alert JSON.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np

from .features import FEATURE_GROUPS


class Diagnoser:
    def __init__(self, signatures: dict):
        self.sig = signatures["faults"]
        self.groups = sorted({g for f in self.sig.values() for g in f["prototype"]})

    def _vec(self, d: dict) -> np.ndarray:
        v = np.array([d.get(g, 0.0) for g in self.groups], dtype=float)
        return v / (np.linalg.norm(v) + 1e-12)

    def _rule_ok(self, rules: dict, z: dict) -> bool:
        for feat, cond in rules.items():
            val = z.get(feat)
            if val is None:
                # rule on a group name: use max z within group
                continue
            if "max" in cond and val > cond["max"]:
                return False
            if "min" in cond and val < cond["min"]:
                return False
        return True

    def diagnose(self, attribution: dict, z_weight: float = 0.6) -> dict:
        """Evidence vector = blend of SHAP group attribution and log-compressed group |z|.

        SHAP tells us what moved the *model*; |z| tells us what moved the *signal*.
        Both are L2-normalised before blending so neither scale dominates.
        """
        z = attribution.get("z", {})
        gz = {g: float(np.log1p(np.mean([abs(z.get(f, 0.0)) for f in names])))
              for g, names in FEATURE_GROUPS.items()}
        a = (1 - z_weight) * self._vec(attribution["groups"]) + z_weight * self._vec(gz)
        a = a / (np.linalg.norm(a) + 1e-12)
        # map rule feature names that are groups (e.g. 'temp') to a representative feature
        z_ext = dict(z)
        z_ext.setdefault("temp", z.get("temp_mean", 0.0))
        z_ext.setdefault("strain_gradient", z.get("strain_gradient", 0.0))
        sims = {}
        for name, spec in self.sig.items():
            p = self._vec(spec["prototype"])
            s = float(a @ p)
            if not self._rule_ok(spec.get("rules", {}), z_ext):
                s *= 0.3  # rule violated → strongly down-weight
            sims[name] = s
        names = list(sims)
        vals = np.array([sims[n] for n in names])
        probs = np.exp(8 * vals) / np.exp(8 * vals).sum()  # sharpened softmax
        best = int(np.argmax(probs))
        return {
            "fault_type": names[best],
            "confidence": round(float(probs[best]), 3),
            "similarities": {n: round(float(v), 3) for n, v in sims.items()},
            "action": self.sig[names[best]]["action"],
        }


def build_alert(det: dict, attribution: dict, diag: dict, trend: str, model_version: str, km: float | None = None) -> dict:
    top = attribution["top_features"][:3]
    evidence = [{"feature": t["feature"], "contrib": t["contrib"]} for t in top]
    sensors = sorted(attribution["sensors"].items(), key=lambda kv: -kv[1])
    why = "; ".join(f"{k} contributed {v:.0%}" for k, v in sensors[:3])
    return {
        "alert_id": f"a-{uuid.uuid4().hex[:10]}",
        "event_id": det["event_id"],
        "segment": det["segment"],
        "km": km,
        "t_min": det["t_min"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score": det["score"],
        "severity": det["severity"],
        "fault": {"type": diag["fault_type"], "confidence": diag["confidence"]},
        "evidence": evidence,
        "sensor_contrib": {k: round(v, 3) for k, v in attribution["sensors"].items()},
        "trend": trend,
        "action": diag["action"],
        "narrative": (
            f"Segment {det['segment']} · severity {det['severity'].upper()} · score {det['score']:.2f}. "
            f"Probable fault: {diag['fault_type'].replace('_', ' ')} (confidence {diag['confidence']:.2f}). "
            f"Why: {why}. Trend: {trend}. Action: {diag['action']}."
        ),
        "model_version": model_version,
    }
