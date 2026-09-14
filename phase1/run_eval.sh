#!/usr/bin/env bash
# Evaluate a phase-1 checkpoint (default: the shipped one) and print the gate table.
#   phase1/run_eval.sh [checkpoint.pt] [out dir]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:-$HERE/checkpoints/cpg_residual_s3_499.pt}"
OUT="${2:-$HERE/../runs/eval_phase1_$(basename "$CKPT" .pt)}"
exec "$HERE/../scripts/eval.sh" MP2-Walk-CpgResidual-PlayRand-v0 "$CKPT" "$OUT" --reference phase1
