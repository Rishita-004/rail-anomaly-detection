"""Explainable real-time anomaly detection for railway track (simulated IoT)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def load_settings(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


__all__ = ["ROOT", "CONFIG_DIR", "load_settings", "load_yaml"]
