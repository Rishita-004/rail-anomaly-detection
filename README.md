# Explainable Real-Time Anomaly Detection for Railway Track (Simulated IoT)

End-to-end pipeline: **simulate → ingest → detect → explain → diagnose → alert**.
Runs on a laptop with zero infrastructure (in-memory bus); switches to Kafka for
multi-process deployment without changing pipeline logic.

## Quick start

```bash
pip install -r requirements.txt          # add `torch` to enable the LSTM Autoencoder
python scripts/train.py                  # trains on healthy data only → artifacts/
python scripts/run_pipeline.py config/scenarios/crack_s12.yaml
python scripts/evaluate.py crack_s12
streamlit run dashboard/app.py           # track map, score trend, alert feed
uvicorn rail_anomaly.api:app --reload    # optional REST alert service
python -m pytest tests -q
```

## Layout

| Path | Stage | What it does |
|---|---|---|
| `rail_anomaly/simulator.py` | 1 Simulation | 5 channels (accel_v, accel_l, strain, temp, acoustic); healthy model = wheel-pass impulses ⊛ ring-down + noise + diurnal temperature; 5 fault transforms with severity curves (linear / exponential / step); hidden ground-truth label per window |
| `rail_anomaly/bus.py` | 2 Ingestion | `InMemoryBus` (default) and `KafkaBus` with identical interface; topics `windows, scores, events, alerts` |
| `rail_anomaly/features.py` | 3a Features | 27 features in 12 groups: band energies (Welch), crest/kurtosis, lateral stats, strain peak/trend, temp–strain correlation, acoustic burst rate, impulse width |
| `rail_anomaly/models/` | 3b Models | `IForestModel` (baseline), `LSTMAutoencoder` (optional, needs torch), `ScoreNormalizer` (sigmoid anchored on healthy p50/p99.5), `EnsembleScorer` |
| `rail_anomaly/detector.py` | 3b Decision | θ = p99.5 of healthy ensemble scores; event opens after k=4 consecutive windows above θ, closes after 20 below; severity bands; 6-h trend |
| `rail_anomaly/explainer.py` | 4 Explain | SHAP TreeExplainer → per-feature / per-group / per-sensor contributions (falls back to \|z\| if shap missing); blends LSTM per-channel error when present |
| `rail_anomaly/diagnoser.py` | 5 Diagnose | Evidence = 0.4·SHAP + 0.6·log(1+\|z\|) per group; cosine similarity to prototypes in `config/fault_signatures.yaml`; rule gates; softmax confidence; alert JSON + narrative |
| `rail_anomaly/pipeline.py` | Orchestrator | Runs all stages per tick (all segments batched) and persists to SQLite |
| `rail_anomaly/storage.py` | Storage | SQLite tables `scores`, `events`, `alerts` (swap for TimescaleDB in prod) |
| `dashboard/app.py`, `rail_anomaly/api.py` | Alert & dashboard | Streamlit UI; FastAPI endpoints `/alerts`, `/segments/{id}/scores` |
| `scripts/` | Tooling | `train.py`, `run_pipeline.py`, `evaluate.py`, `generate_data.py` |
| `config/` | Config | `settings.yaml`, `fault_signatures.yaml`, `scenarios/*.yaml` |

## Scenario files

```yaml
name: crack_s12
duration_min: 90
segments: [S-10, S-11, S-12]
seed: 7
faults:
  - {segment: S-12, type: rail_crack, onset_min: 30, growth: linear, max_severity: 1.0, ramp_min: 45}
```
Fault types: `rail_crack`, `loose_fastening`, `ballast_void`, `thermal_buckling`, `misalignment`.

## Alert contract (`alerts` topic / `alerts` table)

```json
{
  "alert_id": "a-…", "event_id": "ev-00001", "segment": "S-12", "km": 41.3,
  "score": 0.59, "severity": "low",
  "fault": {"type": "rail_crack", "confidence": 0.99},
  "evidence": [{"feature": "acoustic_burst_rate", "contrib": 0.21}, …],
  "sensor_contrib": {"accel_v": 0.50, "acoustic": 0.38, …},
  "trend": "rising", "action": "Ultrasonic inspection within 48 h; …",
  "narrative": "Segment S-12 · severity LOW · score 0.59. Probable fault: rail crack …",
  "model_version": "if-2026-09-10"
}
```

## Results (Isolation Forest + SHAP path; LSTM-AE not active)

| Scenario | Faults | Recall | False alarms | Mean delay | Diagnosis correct |
|---|---|---|---|---|---|
| `crack_s12` | 1 | 1.0 | 0 | 17.6 min | 1/1 |
| `mixed_faults` | 4 overlapping | 1.0 | 0 | 13.6 min | 4/4 |

Per-fault diagnosis accuracy at 60 % severity (flagged windows): crack 1.00, loose fastening 1.00,
ballast void 0.96, thermal buckling 0.90, misalignment 1.00.

## Known limitations / next steps

- Thermal buckling has the weakest signature; detected late (~80 % severity). Adding a longer
  strain-vs-temperature context (hours, not minutes) should bring this forward.
- LSTM Autoencoder is implemented but was not trained in this environment (no torch). Install
  torch and rerun `train.py`; the ensemble, explainer and dashboard already handle it.
- Single-process throughput ≈ 100 windows/s. For real time at scale, run each stage as its own
  process on `KafkaBus` (`docker compose up -d kafka`).
- Threshold refresh, drift monitoring (PSI) and model registry are stubs to be added.
