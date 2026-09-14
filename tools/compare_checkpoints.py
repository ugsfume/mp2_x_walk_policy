#!/usr/bin/env python3
"""Compare two rsl_rl checkpoints tensor by tensor.

    python tools/compare_checkpoints.py a.pt b.pt

Prints, for the actor and critic state dicts, every tensor that differs and
the largest absolute difference; exits 0 only if every tensor is bit-identical
(``torch.equal``) and the iteration counters match. Used to prove that two
training launches that should be the same computation (same task, seed, env
count and start checkpoint) really are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def compare(a_path: Path, b_path: Path) -> bool:
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    ok = True
    for key in ("actor_state_dict", "critic_state_dict"):
        sa, sb = a[key], b[key]
        if list(sa.keys()) != list(sb.keys()):
            print(f"{key}: parameter names differ")
            ok = False
            continue
        for name in sa:
            ta, tb = sa[name], sb[name]
            if ta.shape != tb.shape or not torch.equal(ta, tb):
                delta = (ta.double() - tb.double()).abs().max().item() if ta.shape == tb.shape else float("nan")
                print(f"{key}.{name}: DIFFERS (max |delta| = {delta:.3e})")
                ok = False
        print(f"{key}: {len(sa)} tensors {'identical' if ok else 'checked'}")
    if a.get("iter") != b.get("iter"):
        print(f"iter differs: {a.get('iter')} vs {b.get('iter')}")
        ok = False
    print("CHECKPOINTS:", "BIT-IDENTICAL" if ok else "DIFFERENT")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(64)
    sys.exit(0 if compare(Path(sys.argv[1]), Path(sys.argv[2])) else 1)
