#!/usr/bin/env python3
"""Verify an ONNX policy bundle against its fixtures through the numpy contract.

    python deploy/check_onnx.py deploy/models/e2e_anchored_s42_1050
    python deploy/check_onnx.py <dir with policy.onnx> --fixtures deploy/models/e2e_anchored_s42_1050

Three checks, all through ``deploy/contract.py``:
  1. fixtures: every stored observation through onnxruntime reproduces the
     stored raw action (tolerance 1e-5; the fixtures were produced by the
     TorchScript export on CPU, so this is float noise, not model drift);
  2. fixtures transform_* rows: ``decode_action`` reproduces the stored
     clipped action, CPG offsets and joint targets exactly;
  3. trajectory fixtures: a closed-loop simulation rollout's observations
     reproduce its raw actions (same tolerance).

Pass ``--fixtures`` to check a freshly exported policy.onnx against the
shipped fixtures of the same checkpoint: that proves your export path.
Requires onnxruntime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.contract import OnnxPolicy, contract_for_observation_size, decode_action  # noqa: E402
from deploy.fixtures import load_bundle  # noqa: E402

TOL = 1e-5


def run_batch(policy: OnnxPolicy, observations: np.ndarray) -> np.ndarray:
    return np.stack([policy(row) for row in observations])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="directory containing policy.onnx")
    parser.add_argument("--fixtures", type=Path, default=None, help="directory with fixtures.json / trajectory_fixtures.json (default: model_dir)")
    args = parser.parse_args()
    fixtures_dir = args.fixtures or args.model_dir

    policy = OnnxPolicy(args.model_dir / "policy.onnx")
    contract = policy.contract
    print(f"{args.model_dir / 'policy.onnx'}: input width {policy.observation_size} -> contract {contract.name}")
    ok = True

    fx, meta = load_bundle(fixtures_dir / "fixtures.json")
    if meta.get("contract") not in (None, contract.name):
        print(f"FAIL fixtures were made for contract {meta['contract']}, model is {contract.name}")
        return 1
    pred = run_batch(policy, fx["observations"])
    err = float(np.max(np.abs(pred - fx["expected_raw_actions"])))
    print(f"parity     {len(pred):4d} rows  max |onnx - expected| = {err:.3e}  {'PASS' if err <= TOL else 'FAIL'}")
    ok &= err <= TOL

    n = len(fx["transform_ticks"])
    worst = {"clipped": 0.0, "cpg": 0.0, "target": 0.0}
    for i in range(n):
        clipped, cpg, target = decode_action(
            contract, fx["transform_raw_actions"][i], fx["transform_velocity_commands"][i], int(fx["transform_ticks"][i])
        )
        worst["clipped"] = max(worst["clipped"], float(np.max(np.abs(clipped - fx["transform_clamped_actions"][i]))))
        worst["cpg"] = max(worst["cpg"], float(np.max(np.abs(cpg - fx["transform_cpg_offsets"][i]))))
        worst["target"] = max(worst["target"], float(np.max(np.abs(target - fx["transform_targets"][i]))))
    decode_ok = all(v <= 1e-6 for v in worst.values())
    print(f"decoding   {n:4d} rows  max |delta| clipped {worst['clipped']:.1e}  cpg {worst['cpg']:.1e}  target {worst['target']:.1e}  {'PASS' if decode_ok else 'FAIL'}")
    ok &= decode_ok

    traj_manifest = fixtures_dir / "trajectory_fixtures.json"
    if traj_manifest.exists():
        tr, tmeta = load_bundle(traj_manifest)
        pred = run_batch(policy, tr["observations"])
        err = float(np.max(np.abs(pred - tr["expected_raw_actions"])))
        print(f"trajectory {len(pred):4d} rows  max |onnx - expected| = {err:.3e}  {'PASS' if err <= TOL else 'FAIL'}"
              f"   ({tmeta.get('task')}, commands {tmeta.get('commands_m_s')})")
        ok &= err <= TOL

    print("ONNX_CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
