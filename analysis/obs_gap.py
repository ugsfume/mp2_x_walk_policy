#!/usr/bin/env python3
"""Observation-noise profile: how much noisier are the policy's real inputs
than its simulated ones, along the SAME open-loop gait?

Method (docs/methodology.md): capture a policy's observations while it walks
in simulation (scripts/capture_policy_vectors.py), play the resulting joint-
target stream back on the robot open-loop -- no policy in the loop -- and
rebuild the 45 observation channels from the robot's IMU and servo readings
with the deployment formulas (deploy/contract.py). Then, per channel:

    excess_sd = sqrt(max(0, var_real - var_sim))       over the replay rows
    channel figure = mean of excess_sd over the channel's dims
    uniform-noise amplitude = sqrt(3) x channel figure   (U(-a, a) has sd a/sqrt(3))

Variances, not paired differences: the two streams are the same gait but not
phase-locked to the sample (transport delay), so a row-paired residual would
count timing error as noise. The gyro bias is taken from a still segment
recorded right before playback.

    python analysis/obs_gap.py --sim runs/capture/e2e.json --real my_replay_log.jsonl [--command-x 0.10] [--json out.json]

``--real`` here is our own log format (one JSON object per line, records with
``"event": "sysid_tick"``, ``"segment": "playback"|"playback_settle"``,
``joint_position`` (12, rad, ROS order) and ``telemetry.imu_raw`` (accel g
xyz, gyro deg/s xyz in the sensor frame)). Adapt ``load_real_log`` to your
own logging; ``excess_profile`` works on plain arrays.

Numbers this produced for one unit: ang_vel 1.0 rad/s, gravity 0.30,
joint_vel 4.8 rad/s (gait-independent), joint_pos 0.147-0.175 rad
(gait-dependent: measure along the candidate's own gait). The IMU channels
are upper bounds when the replayed body path diverges from the sim path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.contract import STAND_POSE  # noqa: E402
from deploy.fixtures import load_bundle  # noqa: E402

CHANNELS = {
    "base_ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_pos_rel": slice(9, 21),
    "joint_vel": slice(21, 33),
}
# IMU sensor frame -> body frame on the unit these policies were tuned on.
# Yours may differ; it is part of the deployment contract you supply.
SENSOR_TO_BODY = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
DEG2RAD = np.pi / 180.0
RATE_HZ = 50.0


def real_channels(joint_pos: np.ndarray, imu_raw: np.ndarray, settle_gyro: np.ndarray,
                  sensor_to_body: np.ndarray = SENSOR_TO_BODY) -> dict[str, np.ndarray]:
    """Rebuild the four measured observation channels from a replay log.

    joint_pos (T,12) rad ROS order; imu_raw (T,6) accel then gyro in the
    sensor frame (g, deg/s); settle_gyro (S,3) gyro readings while still.
    """
    bias = (settle_gyro @ sensor_to_body.T).mean(axis=0)
    acc_b = imu_raw[:, :3] @ sensor_to_body.T
    return {
        "projected_gravity": -acc_b / np.linalg.norm(acc_b, axis=1, keepdims=True),
        "base_ang_vel": ((imu_raw[:, 3:] @ sensor_to_body.T) - bias) * DEG2RAD,
        "joint_pos_rel": joint_pos - STAND_POSE.astype(float),
        "joint_vel": np.vstack([np.zeros((1, 12)), np.diff(joint_pos, axis=0) * RATE_HZ]),
    }


def excess_profile(sim_obs: np.ndarray, real: dict[str, np.ndarray]) -> dict:
    """Per-channel excess sd and the uniform-noise amplitude it implies."""
    n = min(sim_obs.shape[0], min(len(v) for v in real.values()))
    out = {"rows": int(n)}
    for name, sl in CHANNELS.items():
        sv, rv = sim_obs[:n, sl], real[name][:n]
        excess = np.sqrt(np.maximum(0.0, rv.var(axis=0) - sv.var(axis=0)))
        out[name] = {
            "sim_sd": float(np.sqrt(sv.var(axis=0)).mean()),
            "real_sd": float(np.sqrt(rv.var(axis=0)).mean()),
            "excess_sd": float(excess.mean()),
            "uniform_amplitude": float(np.sqrt(3.0) * excess.mean()),
        }
    return out


def load_real_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Our replay log: JSON lines with sysid_tick events."""
    q, imu, settle = [], [], []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") != "sysid_tick":
            continue
        if d.get("segment") == "playback_settle":
            settle.append(d["telemetry"]["imu_raw"][3:])
        elif d.get("segment") == "playback":
            q.append(d["joint_position"])
            imu.append(d["telemetry"]["imu_raw"])
    if not q or not settle:
        raise SystemExit(f"{path}: no playback rows or no still segment for the gyro bias")
    return np.asarray(q, float), np.asarray(imu, float), np.asarray(settle, float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", type=Path, required=True, help="capture bundle manifest (.json) the stream was cut from")
    ap.add_argument("--real", type=Path, required=True, nargs="+", help="one or more robot replay logs of that stream")
    ap.add_argument("--command-x", type=float, default=0.10)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    arrays, _ = load_bundle(args.sim)
    rows = np.isclose(arrays["command_x"], args.command_x)
    sim_obs = arrays["observations"][rows].astype(float)
    if sim_obs.shape[1] not in (45, 47):
        raise SystemExit(f"unexpected observation width {sim_obs.shape[1]}")

    results = {}
    for log in args.real:
        q, imu, settle = load_real_log(log)
        results[str(log)] = excess_profile(sim_obs, real_channels(q, imu, settle))
    for log, r in results.items():
        print(log)
        for name in CHANNELS:
            c = r[name]
            print(f"  {name:<18} sim sd {c['sim_sd']:.4f}  real sd {c['real_sd']:.4f}  excess {c['excess_sd']:.4f}"
                  f"  -> uniform amplitude {c['uniform_amplitude']:.4f}")
    if len(results) > 1:
        amps = {name: [r[name]["uniform_amplitude"] for r in results.values()] for name in CHANNELS}
        print("across logs (mean / max):")
        for name, a in amps.items():
            print(f"  {name:<18} {np.mean(a):.4f} / {np.max(a):.4f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
