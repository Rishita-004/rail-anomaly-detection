"""Train models on HEALTHY data only and fit threshold + normalisers.

Usage: python scripts/train.py [--scenario config/scenarios/healthy.yaml] [--no-lstm]
Produces artifacts/: iforest.joblib, norm_iforest.joblib, [lstm_ae.pt, norm_lstm.joblib], meta.json
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly import ROOT, load_settings, load_yaml
from rail_anomaly.detector import fit_threshold
from rail_anomaly.features import FEATURE_NAMES, FeatureExtractor
from rail_anomaly.models import EnsembleScorer, IForestModel, ScoreNormalizer, TORCH_AVAILABLE
from rail_anomaly.simulator import from_scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(ROOT / "config/scenarios/healthy.yaml"))
    ap.add_argument("--no-lstm", action="store_true")
    a = ap.parse_args()

    cfg = load_settings()
    art = ROOT / cfg["paths"]["artifacts"]
    art.mkdir(parents=True, exist_ok=True)
    scen = load_yaml(Path(a.scenario))
    assert not scen.get("faults"), "Train on a healthy-only scenario"

    fs, wsec, stride = cfg["sampling"]["rate_hz"], cfg["sampling"]["window_sec"], cfg["sampling"]["stride_sec"]
    print(f"Simulating healthy data: {scen['duration_min']} min × {len(scen['segments'])} segments …")
    sim = from_scenario(scen, fs, wsec)
    fx = FeatureExtractor(fs)
    X, tensors = [], []
    for w in sim.stream(scen["duration_min"], stride):
        X.append(fx.vector(fx.extract(w["segment"], w["tensor"])))
        tensors.append(w["tensor"])
    X = np.stack(X); T = np.stack(tensors)
    # split: 80% fit, 20% held-out for normaliser + threshold
    n = len(X); idx = np.random.default_rng(0).permutation(n); cut = int(0.8 * n)
    fit_i, val_i = idx[:cut], idx[cut:]
    print(f"  {n} windows, {X.shape[1]} features. fit={len(fit_i)} val={len(val_i)}")

    # --- Isolation Forest
    print("Training Isolation Forest …")
    ifm = IForestModel(**cfg["models"]["iforest"]).fit(X[fit_i])
    norm_if = ScoreNormalizer().fit(ifm.raw_score(X[val_i]))
    ifm.save(art / "iforest.joblib"); joblib.dump(norm_if, art / "norm_iforest.joblib")
    s_if_val = norm_if.transform(ifm.raw_score(X[val_i]))

    # --- LSTM Autoencoder (optional)
    s_lstm_val = None
    if TORCH_AVAILABLE and not a.no_lstm:
        from rail_anomaly.models import LSTMAutoencoder
        print("Training LSTM Autoencoder …")
        ae = LSTMAutoencoder(**cfg["models"]["lstm_ae"]).fit(T[fit_i])
        norm_lstm = ScoreNormalizer().fit(ae.raw_score(T[val_i]))
        ae.save(art / "lstm_ae.pt"); joblib.dump(norm_lstm, art / "norm_lstm.joblib")
        s_lstm_val = norm_lstm.transform(ae.raw_score(T[val_i]))
    else:
        print("LSTM Autoencoder skipped (torch not available or --no-lstm). Ensemble = Isolation Forest only.")
        for p in ("lstm_ae.pt", "norm_lstm.joblib"):
            (art / p).unlink(missing_ok=True)

    # --- threshold on ensemble score of held-out healthy data
    ens = EnsembleScorer(cfg["detection"]["ensemble_weights"])
    s_ens = np.array([ens.combine({"iforest": float(s_if_val[i]), "lstm_ae": None if s_lstm_val is None else float(s_lstm_val[i])})
                      for i in range(len(val_i))])
    theta = fit_threshold(s_ens, cfg["detection"]["threshold_percentile"])
    meta = {"version": f"if{'+ae' if s_lstm_val is not None else ''}-{date.today().isoformat()}",
            "theta": theta, "n_train": int(cut), "features": FEATURE_NAMES,
            "threshold_percentile": cfg["detection"]["threshold_percentile"]}
    (art / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"θ = {theta:.4f}  (p{cfg['detection']['threshold_percentile']} of healthy ensemble scores)")
    print(f"Artifacts written to {art}")


if __name__ == "__main__":
    main()
