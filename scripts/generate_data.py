"""Generate a labelled feature dataset from a scenario (used for training / offline analysis).

Usage: python scripts/generate_data.py config/scenarios/healthy.yaml [--out data/healthy.parquet]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly import ROOT, load_settings, load_yaml
from rail_anomaly.features import FEATURE_NAMES, FeatureExtractor
from rail_anomaly.simulator import from_scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-tensors", action="store_true", help="also save raw windows (.npz) for LSTM training")
    a = ap.parse_args()

    cfg = load_settings()
    scen = load_yaml(Path(a.scenario))
    fs, wsec, stride = cfg["sampling"]["rate_hz"], cfg["sampling"]["window_sec"], cfg["sampling"]["stride_sec"]
    sim = from_scenario(scen, fs, wsec)
    fx = FeatureExtractor(fs)

    rows, tensors = [], []
    for w in sim.stream(scen["duration_min"], stride):
        f = fx.extract(w["segment"], w["tensor"])
        rows.append({"segment": w["segment"], "t_min": w["t_min"], "label": w["label"], "severity": w["severity"], **f})
        if a.save_tensors:
            tensors.append(w["tensor"])
    df = pd.DataFrame(rows)
    out = Path(a.out or ROOT / cfg["paths"]["data"] / f"{scen['name']}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} windows -> {out}")
    if a.save_tensors:
        np.savez_compressed(out.with_suffix(".npz"), X=np.stack(tensors))
        print(f"wrote tensors -> {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
