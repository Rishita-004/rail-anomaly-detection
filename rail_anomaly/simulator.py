"""Stage 1 — Track sensor simulator with physics-inspired fault injection.

Generates five channels per segment at a common sample rate:
  accel_v   vertical acceleration   (wheel-pass impulses + ring-down + noise)
  accel_l   lateral acceleration
  strain    rail bending strain     (quasi-static load per bogie + thermal term)
  temp      rail temperature        (diurnal sinusoid + AR(1) weather noise)
  acoustic  acoustic emission       (background + sparse benign clicks)

Each fault is a transform  x_faulty = f(x_healthy, severity(t))  with severity in [0, 1].
Every window carries a hidden ground-truth label used only by the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

CHANNELS = ["accel_v", "accel_l", "strain", "temp", "acoustic"]


# --------------------------------------------------------------------------- #
# Severity curves
# --------------------------------------------------------------------------- #
def severity_at(t_min: float, onset: float, ramp: float, max_sev: float, growth: str) -> float:
    if t_min < onset:
        return 0.0
    x = min(1.0, (t_min - onset) / max(ramp, 1e-6))
    if growth == "linear":
        s = x
    elif growth == "exponential":
        s = (np.exp(3 * x) - 1) / (np.exp(3) - 1)
    elif growth == "step":
        s = 1.0
    else:
        raise ValueError(growth)
    return float(max_sev * s)


# --------------------------------------------------------------------------- #
# Fault transforms (operate on one window of shape [T, 5])
# --------------------------------------------------------------------------- #
def _wheel_mask(T: int, wheel_idx: np.ndarray, width: int) -> np.ndarray:
    m = np.zeros(T)
    for i in wheel_idx:
        m[max(0, i - width): min(T, i + width)] = 1.0
    return m


def fault_rail_crack(x, s, fs, wheel_idx, rng):
    """High-frequency burst + acoustic click at each wheel pass, amplitude ∝ s."""
    T = x.shape[0]
    t = np.arange(T) / fs
    burst = np.zeros(T)
    for i in wheel_idx:
        n = int(0.01 * fs)  # 10 ms burst
        seg = slice(i, min(T, i + n))
        k = np.arange(seg.stop - seg.start)
        burst[seg] += np.sin(2 * np.pi * 180 * k / fs) * np.exp(-k / (0.25 * n))
    x[:, 0] += 2.5 * s * burst
    x[:, 4] += 4.0 * s * np.abs(burst) + 0.4 * s * rng.standard_normal(T) * _wheel_mask(T, wheel_idx, int(0.01 * fs))
    return x


def fault_loose_fastening(x, s, fs, wheel_idx, rng):
    """Impulse widening + phase jitter + mid-frequency rattle."""
    T = x.shape[0]
    rattle = np.zeros(T)
    for i in wheel_idx:
        j = int(i + rng.integers(-int(0.02 * s * fs), int(0.02 * s * fs) + 1))
        n = int(0.15 * fs)  # long mid-frequency rattle (rail chattering on the sleeper)
        seg = slice(max(0, j), min(T, j + n))
        k = np.arange(seg.stop - seg.start)
        rattle[seg] += np.sin(2 * np.pi * 110 * k / fs) * np.exp(-k / (0.8 * n))
    x[:, 0] += 3.0 * s * rattle
    return x


def fault_ballast_void(x, s, fs, wheel_idx, rng):
    """Sleeper 'hanging': larger low-frequency vertical displacement + strain peak."""
    T = x.shape[0]
    lf = np.zeros(T)
    for i in wheel_idx:
        n = int(0.25 * fs)
        seg = slice(i, min(T, i + n))
        k = np.arange(seg.stop - seg.start)
        lf[seg] += np.sin(np.pi * k / n)
    x[:, 0] += 1.5 * s * lf
    x[:, 2] += 1.2 * s * lf
    return x


def fault_thermal_buckling(x, s, fs, wheel_idx, rng):
    """Compressive strain drifting upward with temperature."""
    temp_norm = (x[:, 3] - 20.0) / 15.0
    x[:, 2] += 3.0 * s * np.clip(temp_norm, 0, None) + 2.0 * s
    x[:, 3] += 8.0 * s  # hot rail
    return x


def fault_misalignment(x, s, fs, wheel_idx, rng):
    """Lateral bias and uneven lateral response on wheel pass."""
    T = x.shape[0]
    x[:, 1] += 1.6 * s * _wheel_mask(T, wheel_idx, int(0.04 * fs)) * np.sign(np.sin(np.arange(T) / fs * 2 * np.pi * 0.7))
    x[:, 1] += 0.5 * s
    return x


FAULTS = {
    "rail_crack": fault_rail_crack,
    "loose_fastening": fault_loose_fastening,
    "ballast_void": fault_ballast_void,
    "thermal_buckling": fault_thermal_buckling,
    "misalignment": fault_misalignment,
}


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
@dataclass
class FaultSpec:
    segment: str
    type: str
    onset_min: float
    growth: str = "linear"
    max_severity: float = 1.0
    ramp_min: float = 30.0


@dataclass
class TrackSimulator:
    segments: list[str]
    fs: int = 500
    window_sec: float = 2.0
    faults: list[FaultSpec] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.T = int(self.fs * self.window_sec)
        # per-segment slow state
        self._temp_state = {s: 0.0 for s in self.segments}
        self._ring = self._ring_down()

    def _ring_down(self) -> np.ndarray:
        n = int(0.05 * self.fs)
        k = np.arange(n)
        return np.sin(2 * np.pi * 60 * k / self.fs) * np.exp(-k / (0.3 * n))

    def _healthy_window(self, seg: str, t_min: float) -> tuple[np.ndarray, np.ndarray]:
        rng, T, fs = self.rng, self.T, self.fs
        x = np.zeros((T, 5))
        # train pass: every window has a train with probability 0.7
        wheel_idx = np.array([], dtype=int)
        if rng.random() < 0.7:
            speed = rng.uniform(15, 40)              # m/s
            axle_spacing = rng.uniform(2.3, 2.8)     # m
            load = rng.uniform(0.7, 1.3)
            period = axle_spacing / speed            # s between axles
            start = rng.uniform(0, 0.4)
            wheel_idx = np.arange(int(start * fs), T, int(period * fs))
            # vertical accel: impulse train ⊛ ring-down
            imp = np.zeros(T)
            imp[wheel_idx] = load * rng.uniform(0.9, 1.1, len(wheel_idx))
            x[:, 0] += signal.fftconvolve(imp, self._ring, mode="full")[:T]
            x[:, 1] += 0.35 * signal.fftconvolve(imp * rng.uniform(0.8, 1.2, T), self._ring, mode="full")[:T]
            # strain: quasi-static load bump per bogie
            bump = signal.windows.hann(int(0.12 * fs))
            x[:, 2] += load * signal.fftconvolve(imp, bump, mode="full")[:T]
        # noise
        x[:, 0] += 0.08 * rng.standard_normal(T)
        x[:, 1] += 0.06 * rng.standard_normal(T)
        x[:, 2] += 0.03 * rng.standard_normal(T)
        # temperature: diurnal + AR(1)
        hour = (t_min / 60.0) % 24
        diurnal = 20 + 10 * np.sin(2 * np.pi * (hour - 9) / 24)
        self._temp_state[seg] = 0.98 * self._temp_state[seg] + 0.3 * rng.standard_normal()
        x[:, 3] += diurnal + self._temp_state[seg]
        x[:, 2] += 0.03 * (x[:, 3] - 20)  # mild thermal strain
        # acoustic: background + rare benign clicks
        x[:, 4] += 0.05 * np.abs(rng.standard_normal(T))
        for _ in range(rng.poisson(0.3)):
            i = rng.integers(0, T - 5)
            x[i:i + 5, 4] += 0.4
        return x, wheel_idx

    def _apply_faults(self, seg: str, t_min: float, x: np.ndarray, wheel_idx: np.ndarray) -> tuple[str, float]:
        label, sev_max = "healthy", 0.0
        for f in self.faults:
            if f.segment != seg:
                continue
            s = severity_at(t_min, f.onset_min, f.ramp_min, f.max_severity, f.growth)
            if s <= 0:
                continue
            x = FAULTS[f.type](x, s, self.fs, wheel_idx, self.rng)
            if s > sev_max:
                label, sev_max = f.type, s
        return label, sev_max

    def window(self, seg: str, t_min: float) -> dict:
        x, wheel_idx = self._healthy_window(seg, t_min)
        label, sev = self._apply_faults(seg, t_min, x, wheel_idx)
        return {
            "segment": seg,
            "t_min": round(t_min, 3),
            "tensor": x.astype(np.float32),
            "label": label,
            "severity": round(sev, 3),
        }

    def ticks(self, duration_min: float, stride_sec: float = 0.5):
        """Yield one list of windows (all segments) per stride tick."""
        n = int(duration_min * 60 / stride_sec)
        for i in range(n):
            t_min = i * stride_sec / 60.0
            yield [self.window(seg, t_min) for seg in self.segments]

    def stream(self, duration_min: float, stride_sec: float = 0.5):
        """Yield windows for all segments, interleaved in time order."""
        for tick in self.ticks(duration_min, stride_sec):
            yield from tick


def from_scenario(cfg: dict, fs: int, window_sec: float) -> TrackSimulator:
    faults = [FaultSpec(**f) for f in cfg.get("faults", [])]
    return TrackSimulator(segments=cfg["segments"], fs=fs, window_sec=window_sec, faults=faults, seed=cfg.get("seed", 0))
