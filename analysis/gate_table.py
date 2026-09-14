#!/usr/bin/env python3
"""Score an evaluation against the gait gates, next to the reference row.

    python analysis/gate_table.py <eval dir> [--reference phase1|phase2] [--noisy <eval dir>]

``<eval dir>`` is an output directory of ``scripts/evaluate_policy.py`` run on
a PlayRand id with 32 envs, 8 recorded, 490 steps, 0.10 m/s. ``--noisy`` is
the same checkpoint evaluated on ``MP2-Walk-E2E-Noisy-v0`` (full measured
observation noise) and scores the robustness gate.

The bars were frozen from a reference population of known steppers and known
foot-sliders, with a margin against the eval-to-eval scatter measured on
repeated evaluations of one checkpoint (vx 0.0037 m/s, f0 0.06 Hz, clearance
1.0 mm, vz 0.0021 m/s). A verdict within one scatter sd of its bar is not a
verdict: repeat the eval three times and take the median.

Gates (PlayRand, clean observations)
  health   torque >= 0.06 N m | vx >= 0.035 m/s | base height std <= 2.5 mm
           | vertical bounce (after 1 s) <= 0.0315 m/s
  family   f0 >= 1.80 Hz | swing clearance median <= 12 mm
  posture  |pitch| median <= 1.5 deg | per-foot lift spread <= 5 mm
  robust   (with --noisy) harmonic-share loss <= 15 pts AND vx_noisy >= 0.8 vx
Reported, not gated: contact cadence, foot path excess, spawn-frame
displacement. The 660 mm displacement bar applies to OPEN-LOOP replays
(scripts/openloop_replay.sh), not to closed-loop evals.

A trace with any episode reset (done or timeout) is invalid: the
displacement metrics are poisoned by the teleport.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhythm import rhythm_from_traces  # noqa: E402

SETTLE_STEPS = 50

GATES = [
    # key, label, unit, op, bar, tier
    ("torque", "torque mean", "N m", ">=", 0.06, "health"),
    ("vx", "forward speed", "m/s", ">=", 0.035, "health"),
    ("bz_mm", "base height std", "mm", "<=", 2.5, "health"),
    ("vz", "vertical bounce", "m/s", "<=", 0.0315, "health"),
    ("f0", "gait f0", "Hz", ">=", 1.80, "family"),
    ("clear_mm", "swing clearance med", "mm", "<=", 12.0, "family"),
    ("pitch_deg", "|pitch| median", "deg", "<=", 1.5, "posture"),
    ("lift_spread_mm", "per-foot lift spread", "mm", "<=", 5.0, "posture"),
]
REPORTED = [
    ("share", "harmonic share", ""),
    ("cadence", "contact cadence", "Hz"),
    ("fpe", "foot path excess", ""),
    ("disp_mm", "spawn-frame displacement", "mm"),
]
SCATTER_SD = {"vx": 0.0037, "f0": 0.06, "clear_mm": 1.04, "vz": 0.0021}

# What the shipped checkpoints scored on these gates (32 envs, 8 recorded,
# 490 steps, 0.10 m/s). Expect your own eval to land within scatter, not on it.
REFERENCE = {
    "phase1": {  # phase1/checkpoints/cpg_residual_s3_499.pt on MP2-Walk-CpgResidual-PlayRand-v0
        "torque": 0.0795, "vx": 0.1004, "bz_mm": 2.0, "vz": 0.0210, "f0": 2.041,
        "clear_mm": 6.2, "pitch_deg": 0.24, "lift_spread_mm": 2.9,
        "share": 0.974, "cadence": 1.913, "fpe": 1.110,
    },
    "phase2": {  # phase2/checkpoints/e2e_anchored_s42_1050.pt on MP2-Walk-E2E-PlayRand-v0
        "torque": 0.0773, "vx": 0.0863, "bz_mm": 2.21, "vz": 0.0264, "f0": 1.939,
        "clear_mm": 6.85, "pitch_deg": 0.32, "lift_spread_mm": 2.66,
        "share": 0.965, "cadence": 1.722, "fpe": 1.243, "disp_mm": 872.0,
        "noisy_vx": 0.0869, "noisy_share": 0.923,
    },
}


def bounce_vz(traces: Path) -> float:
    d = np.load(traces)
    root = d["root_pos_w"]
    dt = float(d["step_dt"])
    if root.ndim == 2:
        root = root[None]
    vz = [float(np.sqrt(np.mean((np.diff(root[e, :, 2]) / dt)[SETTLE_STEPS:] ** 2))) for e in range(root.shape[0])]
    return float(np.median(vz))


def find_outputs(eval_dir: Path) -> tuple[Path, Path]:
    summaries = sorted(eval_dir.glob("*_summary.json"))
    traces = sorted(eval_dir.glob("*_gait_traces.npz"))
    if not summaries or not traces:
        raise SystemExit(f"{eval_dir}: need one *_summary.json and one *_gait_traces.npz")
    return summaries[0], traces[0]


def score(eval_dir: Path) -> dict:
    summary_path, traces = find_outputs(eval_dir)
    s = json.loads(summary_path.read_text())
    r = rhythm_from_traces(traces)
    disp = s.get("forward_displacement_spawnframe_m")
    return {
        "eval_dir": str(eval_dir),
        "task": s.get("task"),
        "checkpoint": s.get("checkpoint"),
        "num_envs": s.get("num_envs"),
        "record_envs": s.get("record_envs"),
        "steps": s.get("steps"),
        "resets": float(s.get("done_count_mean", 0.0)) + float(s.get("timeout_count_mean", 0.0)),
        "torque": s["joint_torque_abs_mean_nm"],
        "vx": s["mean_forward_velocity_mps"],
        "bz_mm": 1e3 * s["base_height_std_m"],
        "vz": bounce_vz(traces),
        "f0": r["f0_hz"],
        "share": r["harmonic_power_share"],
        "clear_mm": 1e3 * s["swing_clearance_median_m"],
        "pitch_deg": s["abs_pitch_median_deg"],
        "lift_spread_mm": 1e3 * s["per_foot_lift_spread_m"],
        "cadence": s.get("cadence_hz", float("nan")),
        "fpe": s.get("foot_path_excess", float("nan")),
        "disp_mm": 1e3 * disp if disp is not None else float("nan"),
    }


def verdict(op: str, value: float, bar: float) -> bool:
    return value >= bar if op == ">=" else value <= bar


def fmt(v, unit: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    digits = 4 if abs(v) < 1 else 2 if abs(v) < 100 else 0
    return f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--reference", choices=sorted(REFERENCE), default=None)
    parser.add_argument("--noisy", type=Path, default=None, help="eval dir on MP2-Walk-E2E-Noisy-v0")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    row = score(args.eval_dir)
    ref = REFERENCE.get(args.reference, {}) if args.reference else {}
    print(f"{row['task']}  {row['checkpoint']}")
    print(f"{row['num_envs']} envs, {row['record_envs']} recorded, {row['steps']} steps")
    if row["resets"] > 0:
        print(f"INVALID: {row['resets']:.1f} episode resets per env; displacement metrics are not meaningful")
    all_pass = row["resets"] == 0
    head = f"{'gate':<24}{'value':>10}{'bar':>10}{'reference':>11}  verdict"
    print(head)
    print("-" * len(head))
    results = {}
    for key, label, unit, op, bar, tier in GATES:
        v = row[key]
        ok = verdict(op, v, bar)
        near = key in SCATTER_SD and abs(v - bar) <= SCATTER_SD[key]
        note = "PASS" if ok else "FAIL"
        if near:
            note += "  (within scatter of the bar: median of 3)"
        all_pass &= ok
        results[key] = {"value": v, "bar": bar, "pass": ok}
        print(f"{label:<24}{fmt(v, unit):>10}{op + ' ' + fmt(bar, unit):>10}{fmt(ref.get(key), unit):>11}  {note}")
    for key, label, unit in REPORTED:
        print(f"{label:<24}{fmt(row[key], unit):>10}{'':>10}{fmt(ref.get(key), unit):>11}  reported")

    if args.noisy:
        noisy = score(args.noisy)
        loss_pts = 100.0 * (row["share"] - noisy["share"])
        ratio = noisy["vx"] / row["vx"] if row["vx"] else float("nan")
        ok = loss_pts <= 15.0 and ratio >= 0.8
        all_pass &= ok
        results["robustness"] = {"share_loss_pts": loss_pts, "vx_ratio": ratio, "pass": ok}
        print("-" * len(head))
        print(f"noise robustness ({noisy['task']}):")
        print(f"{'  harmonic share loss':<24}{loss_pts:>9.1f}p{'<= 15.0':>10}"
              f"{fmt(100 * (ref.get('share', np.nan) - ref.get('noisy_share', np.nan)), ''):>11}  {'PASS' if loss_pts <= 15 else 'FAIL'}")
        print(f"{'  vx retained':<24}{ratio:>10.3f}{'>= 0.800':>10}"
              f"{fmt(ref.get('noisy_vx', np.nan) / ref['vx'] if 'vx' in ref else None, ''):>11}  {'PASS' if ratio >= 0.8 else 'FAIL'}")

    print("GATE_TABLE:", "PASS" if all_pass else "FAIL")
    if args.json:
        args.json.write_text(json.dumps({"row": row, "results": results}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
