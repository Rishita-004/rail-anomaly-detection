"""End-to-end pipeline runner (in-process, bus-driven).

simulator -> windows -> featurizer -> detector -> explainer -> diagnoser -> alerts
Every stage publishes to the bus so it can be split into separate processes
later without changing the logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import CONFIG_DIR, ROOT, load_settings, load_yaml
from .bus import InMemoryBus
from .detector import Detector
from .diagnoser import Diagnoser, build_alert
from .explainer import Explainer
from .features import FeatureExtractor
from .models import EnsembleScorer, IForestModel, TORCH_AVAILABLE
from .simulator import from_scenario
from .storage import Store

KM_OF = {"S-10": 39.8, "S-11": 40.5, "S-12": 41.3, "S-13": 42.0, "S-14": 42.7}


class Artifacts:
    """Loads everything produced by scripts/train.py."""

    def __init__(self, art_dir: Path):
        self.dir = Path(art_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.version = meta["version"]
        self.theta = meta["theta"]
        self.iforest = IForestModel.load(self.dir / "iforest.joblib")
        self.norm_if = joblib.load(self.dir / "norm_iforest.joblib")
        self.lstm = None
        self.norm_lstm = None
        if TORCH_AVAILABLE and (self.dir / "lstm_ae.pt").exists():
            from .models import LSTMAutoencoder
            self.lstm = LSTMAutoencoder.load(self.dir / "lstm_ae.pt")
            self.norm_lstm = joblib.load(self.dir / "norm_lstm.joblib")


def run(scenario_path: Path, art_dir: Path | None = None, db_path: Path | None = None,
        verbose: bool = True, max_alerts_per_event: int = 1) -> dict:
    cfg = load_settings()
    scen = load_yaml(scenario_path)
    art = Artifacts(art_dir or ROOT / cfg["paths"]["artifacts"])
    store = Store(db_path or ROOT / cfg["paths"]["data"] / "pipeline.db")
    run_name = scen["name"]

    fs, wsec, stride = cfg["sampling"]["rate_hz"], cfg["sampling"]["window_sec"], cfg["sampling"]["stride_sec"]
    sim = from_scenario(scen, fs, wsec)
    fx = FeatureExtractor(fs)
    det = Detector(art.theta, cfg["detection"]["persistence_k"], cfg["detection"]["clear_k"], cfg["detection"]["severity_bands"])
    ens = EnsembleScorer(cfg["detection"]["ensemble_weights"])
    expl = Explainer(art.iforest)
    diag = Diagnoser(load_yaml(CONFIG_DIR / "fault_signatures.yaml"))
    bus = InMemoryBus()

    alerts: list[dict] = []
    alerted_events: dict[str, int] = {}
    n = 0

    for tick in sim.ticks(scen["duration_min"], stride):
        # --- features for all segments in this tick, scored in one batch
        xvecs = np.stack([fx.vector(fx.extract(w["segment"], w["tensor"])) for w in tick])
        s_if_all = art.norm_if.transform(art.iforest.raw_score(xvecs))
        ch_err_all, s_lstm_all = None, None
        if art.lstm is not None:
            ch_err_all = art.lstm.channel_error(np.stack([w["tensor"] for w in tick]))
            s_lstm_all = art.norm_lstm.transform(ch_err_all.mean(axis=1))
        for i, w in enumerate(tick):
            n += 1
            seg, t = w["segment"], w["t_min"]
            xvec = xvecs[i]
            s_if = float(s_if_all[i])
            s_lstm = None if s_lstm_all is None else float(s_lstm_all[i])
            ch_err = None if ch_err_all is None else ch_err_all[i]
            s = ens.combine({"iforest": s_if, "lstm_ae": s_lstm})
            r = det.step(seg, t, s)
            bus.publish("scores", seg, r)
            store.score(run_name, r, w["label"], w["severity"], s_if, s_lstm)
            # --- events / explanation / alert
            if r["transition"] == "open":
                store.event_open(run_name, r["event_id"], seg, t)
            if r["transition"] == "close":
                store.event_close(run_name, r["event_id"], t)
            if r["in_event"] and alerted_events.get(r["event_id"], 0) < max_alerts_per_event:
                attribution = expl.explain(xvec, ch_err)
                d = diag.diagnose(attribution)
                a = build_alert(r, attribution, d, det.trend(seg), art.version, KM_OF.get(seg))
                a["ground_truth"] = {"label": w["label"], "severity": w["severity"]}
                bus.publish("alerts", seg, a)
                store.alert(run_name, a)
                alerts.append(a)
                alerted_events[r["event_id"]] = alerted_events.get(r["event_id"], 0) + 1
                if verbose:
                    print(f"\n🔔 ALERT  {a['narrative']}\n   (ground truth: {w['label']} @ severity {w['severity']})")
            if verbose and n % 3000 == 0:
                print(f"  … {n} windows processed (t = {t:.1f} min)")

    store.close()
    if verbose:
        print(f"\nRun '{run_name}' complete: {n} windows, {len(alerts)} alerts. Stored in {store.path}")
    return {"run": run_name, "windows": n, "alerts": alerts}
