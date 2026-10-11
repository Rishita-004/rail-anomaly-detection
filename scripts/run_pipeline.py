"""Run the full streaming pipeline on a scenario.

Usage: python scripts/run_pipeline.py config/scenarios/crack_s12.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rail_anomaly.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(Path(a.scenario), verbose=not a.quiet)
