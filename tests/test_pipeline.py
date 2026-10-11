"""Smoke tests: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly.detector import Detector
from rail_anomaly.features import FEATURE_NAMES, FeatureExtractor
from rail_anomaly.models import IForestModel, ScoreNormalizer
from rail_anomaly.simulator import FaultSpec, TrackSimulator, severity_at


def test_severity_curves():
    assert severity_at(0, 10, 20, 1.0, "linear") == 0
    assert abs(severity_at(20, 10, 20, 1.0, "linear") - 0.5) < 1e-9
    assert severity_at(11, 10, 20, 0.7, "step") == 0.7


def test_simulator_shapes_and_labels():
    sim = TrackSimulator(["A"], fs=500, window_sec=2.0, faults=[FaultSpec("A", "rail_crack", onset_min=1, ramp_min=1)], seed=0)
    w0 = sim.window("A", 0.0)
    w1 = sim.window("A", 3.0)
    assert w0["tensor"].shape == (1000, 5) and w0["label"] == "healthy"
    assert w1["label"] == "rail_crack" and w1["severity"] == 1.0


def test_features_complete():
    sim = TrackSimulator(["A"], seed=1)
    fx = FeatureExtractor(500)
    f = fx.extract("A", sim.window("A", 0.0)["tensor"])
    assert set(FEATURE_NAMES) <= set(f)
    assert np.isfinite(fx.vector(f)).all()


def test_crack_scores_higher_than_healthy():
    sim = TrackSimulator(["A"], seed=2, faults=[FaultSpec("A", "rail_crack", onset_min=5, ramp_min=1)])
    fx = FeatureExtractor(500)
    X = np.stack([fx.vector(fx.extract("A", sim.window("A", t)["tensor"])) for t in np.linspace(0, 4.9, 300)])
    m = IForestModel(n_estimators=50).fit(X)
    norm = ScoreNormalizer().fit(m.raw_score(X))
    healthy = norm.transform(m.raw_score(X[:50])).mean()
    faulty = np.stack([fx.vector(fx.extract("A", sim.window("A", t)["tensor"])) for t in np.linspace(8, 9, 30)])
    assert norm.transform(m.raw_score(faulty)).mean() > healthy + 0.2  # ~30% of windows have no train pass, so the mean is diluted


def test_detector_persistence():
    d = Detector(0.5, persistence_k=3, clear_k=2)
    out = [d.step("A", t, s) for t, s in enumerate([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1])]
    assert out[2]["in_event"] is False          # spike of 2 does not open
    assert out[5]["transition"] == "open"       # 3 consecutive opens
    assert out[7]["transition"] == "close"      # 2 consecutive below closes
