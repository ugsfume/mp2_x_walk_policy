#!/usr/bin/env python3
"""Gait rhythm from foot height: fundamental frequency and harmonic power share.

Amplitude metrics (foot z range, clearance) cannot tell a clean 2 Hz trot from
aperiodic thrashing between the same extremes. This one can: it takes the
vertical position of each foot, finds the dominant frequency in the 0.5-12 Hz
band, and reports how much of the band power sits within +-0.35 Hz of that
frequency and its first two harmonics. A clean trot scores 0.95-0.99; erratic
leg motion with the same excursion scores far lower.

    python analysis/rhythm.py <stem>_gait_traces.npz [more.npz ...] [--json out.json]

Input is the ``*_gait_traces.npz`` written by ``scripts/evaluate_policy.py``
(``foot_pos_w`` of shape (T, feet, 3) or (envs, T, feet, 3), and ``step_dt``).
All env x foot series are pooled and the median over series is reported;
f0 is the phase-1/phase-2 gait-family gate (>= 1.8 Hz on the 490-step,
50 Hz evaluation).

Works on numpy alone; no simulator needed. On hardware logs, feed the same
function the world-frame foot heights from your own forward kinematics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def harmonic_power_share(z: np.ndarray, dt: float) -> tuple[float, float]:
    """(fundamental_hz, share) over the series in the columns of ``z`` (T, n).

    Each series is mean-removed and Hann-windowed; the dominant frequency is
    the rfft peak inside 0.5-12 Hz; the share is the power within +-0.35 Hz of
    f0, 2 f0 and 3 f0 over the band power. Medians over series.
    """
    shares, peaks = [], []
    for col in range(z.shape[1]):
        x = z[:, col] - z[:, col].mean()
        n = len(x)
        if n < 32:
            continue
        power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        freq = np.fft.rfftfreq(n, dt)
        band = (freq > 0.5) & (freq < 12.0)
        if not band.any() or power[band].sum() <= 0:
            continue
        f0 = freq[band][np.argmax(power[band])]
        harmonic = sum(power[np.abs(freq - h * f0) < 0.35].sum() for h in (1, 2, 3))
        shares.append(harmonic / power[band].sum())
        peaks.append(f0)
    if not shares:
        return float("nan"), float("nan")
    return float(np.median(peaks)), float(np.median(shares))


def rhythm_from_traces(npz_path: Path) -> dict:
    """Pooled f0 and harmonic share of a ``*_gait_traces.npz`` file."""
    data = np.load(npz_path, allow_pickle=False)
    foot = data["foot_pos_w"]
    step_dt = float(data["step_dt"])
    if foot.ndim == 3:
        foot = foot[None]
    envs, steps, nfeet, _ = foot.shape
    z = foot[:, :, :, 2]
    pooled = np.moveaxis(z, 1, 0).reshape(steps, envs * nfeet)
    f0, share = harmonic_power_share(pooled, step_dt)
    per_env = [harmonic_power_share(z[e], step_dt) for e in range(envs)]
    return {
        "path": str(npz_path),
        "envs": envs,
        "steps": steps,
        "step_dt_s": step_dt,
        "series": envs * nfeet,
        "f0_hz": f0,
        "harmonic_power_share": share,
        "per_env_f0_hz": [p[0] for p in per_env],
        "per_env_share": [p[1] for p in per_env],
    }


def self_test() -> None:
    """A 2 Hz sinusoid sampled at 50 Hz for 490 steps must read f0 = 2.04 Hz
    (the nearest FFT bin) with share ~1; white noise must read a low share."""
    t = np.arange(490) * 0.02
    clean = np.stack([0.01 * np.sin(2 * np.pi * 2.0 * t + k) for k in range(4)], axis=1)
    f0, share = harmonic_power_share(clean, 0.02)
    assert abs(f0 - 2.0408) < 1e-3, f0
    assert share > 0.99, share
    rng = np.random.default_rng(0)
    f0n, share_n = harmonic_power_share(rng.normal(size=(490, 4)), 0.02)
    assert share_n < 0.5, share_n
    print(f"RHYTHM_SELF_TEST: PASS (clean f0 {f0:.4f} Hz share {share:.4f}; noise share {share_n:.3f})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.traces:
            return 0
    results = [rhythm_from_traces(p) for p in args.traces]
    for r in results:
        print(f"{r['path']}: f0 {r['f0_hz']:.3f} Hz  harmonic share {r['harmonic_power_share']:.3f}  "
              f"({r['envs']} envs x {r['series'] // r['envs']} feet, {r['steps']} steps)")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
