#!/usr/bin/env python3
"""Check that a training run's dumped config equals the recorded one.

Isaac Lab writes the fully resolved environment and agent configs of every
run to ``<run dir>/params/{env,agent}.yaml``. ``phase*/record/`` holds those
files from the runs that produced the shipped checkpoints. This script
compares a fresh run's dump against them -- values AND key order, because
term order is where rewards are summed and where random draws happen -- and
lists every difference that is not on the allow-list below.

    python tools/check_record.py <run dir> phase1/record
    python tools/check_record.py <run dir> phase2/record

Exit status 0 means the config that ran is the config that was recorded.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# Keys whose values legitimately differ between the recorded run and a fresh
# one. Paths are dotted; a trailing ``*`` matches any suffix.
ALLOW_ENV = {
    "seed",
    "log_dir",
    "scene.robot.spawn.usd_path",
    "sim.device",
    "sim.visualizer_cfgs*",
    "export_io_descriptors",
}
ALLOW_AGENT = {
    "seed",
    "device",
    "max_iterations",
    "experiment_name",
    "run_name",
    "resume",
    "load_run",
    "load_checkpoint",
    "algorithm.anchor_path",
    "algorithm.anchor_sha256",
}
# The recorded runs used the package the code was developed in; module paths
# in function references are rewritten before comparing.
MODULE_REWRITES = [
    (r"mini_pupper_isaaclab\.tasks\.mini_pupper_velocity\.mdp:", "mp2_x_walk_policy.mdp:"),
    (r"mini_pupper_isaaclab\.actuators:", "mp2_x_walk_policy.actuators:"),
    (r"mini_pupper_isaaclab\.modifiers:", "mp2_x_walk_policy.modifiers:"),
    (r"mini_pupper_isaaclab\.actions:", "mp2_x_walk_policy.actions:"),
    (r"mini_pupper_isaaclab\.algorithms:", "mp2_x_walk_policy.algorithms:"),
]


def load(path: Path):
    # The dumps carry python-specific tags (tuple, slice); the unsafe loader
    # is needed to read them. These files come from our own runs.
    return yaml.load(path.read_text(), Loader=yaml.UnsafeLoader)


def rewrite(value):
    if isinstance(value, str):
        for pattern, repl in MODULE_REWRITES:
            value = re.sub(pattern, repl, value)
    return value


def allowed(path: str, allow: set[str]) -> bool:
    for entry in allow:
        if entry.endswith("*"):
            if path.startswith(entry[:-1]):
                return True
        elif path == entry:
            return True
    return False


def diff(a, b, path: str, allow: set[str], out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = list(a.keys()), list(b.keys())
        if ka != kb:
            if set(ka) == set(kb):
                out.append(f"{path or '<root>'}: key ORDER differs\n    recorded {ka}\n    run      {kb}")
            else:
                missing = [k for k in ka if k not in kb]
                extra = [k for k in kb if k not in ka]
                out.append(f"{path or '<root>'}: keys differ (missing {missing}, extra {extra})")
        for k in ka:
            if k in b:
                diff(a[k], b[k], f"{path}.{k}" if path else str(k), allow, out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", allow, out)
        return
    a, b = rewrite(a), rewrite(b)
    if a != b and not allowed(path, allow):
        out.append(f"{path}: recorded {a!r}, run {b!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run directory containing params/env.yaml and params/agent.yaml")
    parser.add_argument("record_dir", type=Path, help="phase1/record or phase2/record")
    args = parser.parse_args()

    failures = 0
    for name, allow in (("env", ALLOW_ENV), ("agent", ALLOW_AGENT)):
        recorded = load(args.record_dir / f"{name}.yaml")
        run = load(args.run_dir / "params" / f"{name}.yaml")
        out: list[str] = []
        diff(recorded, run, "", allow, out)
        status = "PASS" if not out else "FAIL"
        print(f"{name}.yaml: {status}")
        for line in out:
            print("  " + line)
        failures += bool(out)
    print("RECORD_CHECK:", "PASS" if failures == 0 else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
