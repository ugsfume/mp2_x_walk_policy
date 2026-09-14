#!/usr/bin/env python3
"""Turn one command block of a capture bundle into the two open-loop streams.

    python analysis/capture_to_stream.py --capture runs/capture/e2e.json --command-x 0.10 --output runs/capture/e2e_010

writes

    <output>_replay.npz     the RAW ACTIONS of the block, for
                            evaluate_policy --scripted_agent replay (key
                            ``q_measured``, (T, 12) in [-1, 1]; ``rate_hz`` 50).
                            Replayed on the same task family the capture came
                            from, the action term rebuilds the identical joint
                            targets, so this is the open-loop simulation twin.
    <output>_targets.json   the JOINT TARGETS in radians (T x 12, ROS order),
                            decoded through deploy/contract.py with tick = row
                            index -- what a robot plays back open-loop at 50 Hz
                            for the observation-noise measurement
                            (docs/methodology.md).

The block must start at step 0 and be contiguous; for the CPG contract it
must also be a whole number of 25-tick gait periods so the clock phase of
row i is the phase the policy saw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.contract import CPG_PERIOD_TICKS, contract_for_observation_size, decode_action  # noqa: E402
from deploy.fixtures import load_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True, help="capture bundle manifest (.json)")
    ap.add_argument("--command-x", type=float, default=0.10)
    ap.add_argument("--output", type=Path, required=True, help="output stem")
    args = ap.parse_args()

    arrays, meta = load_bundle(args.capture)
    obs = arrays["observations"]
    raw = arrays["expected_raw_actions"].astype(np.float32)
    cmd = arrays["command_x"].astype(float)
    steps = arrays["step_indices"].astype(int)

    rows = np.nonzero(np.isclose(cmd, args.command_x, atol=1e-9))[0]
    if rows.size == 0 or not np.all(np.diff(rows) == 1):
        raise SystemExit(f"command block {args.command_x} missing or not contiguous")
    block_steps = steps[rows]
    if block_steps[0] != 0 or not np.all(np.diff(block_steps) == 1):
        raise SystemExit("step_indices do not restart densely at the block start")
    contract = contract_for_observation_size(obs.shape[1])
    if contract.uses_cpg_prior and rows.size % CPG_PERIOD_TICKS != 0:
        raise SystemExit(f"block length {rows.size} is not a whole number of {CPG_PERIOD_TICKS}-tick gait periods")

    q = raw[rows]
    command_vec = np.asarray([args.command_x, 0.0, 0.0], dtype=np.float32)
    targets = [decode_action(contract, q[i], command_vec, tick=i)[2].tolist() for i in range(len(q))]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay_path = Path(str(args.output) + "_replay.npz")
    np.savez(replay_path, q_measured=q, rate_hz=np.float64(50.0))
    targets_path = Path(str(args.output) + "_targets.json")
    targets_path.write_text(json.dumps(targets) + "\n")

    info = {
        "capture": str(args.capture),
        "task": meta.get("task"),
        "contract": contract.name,
        "command_x": args.command_x,
        "rows": int(len(q)),
        "rate_hz": 50.0,
        "replay_npz": str(replay_path),
        "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "targets_json": str(targets_path),
        "targets_sha256": hashlib.sha256(targets_path.read_bytes()).hexdigest(),
    }
    Path(str(args.output) + ".meta.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
