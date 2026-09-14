#!/usr/bin/env bash
# Evaluate a phase-2 checkpoint (default: the shipped one): gate table on clean
# observations plus the robustness gate under the full measured noise.
#   phase2/run_eval.sh [checkpoint.pt] [out dir]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:-$HERE/checkpoints/e2e_anchored_s42_1050.pt}"
OUT="${2:-$HERE/../runs/eval_phase2_$(basename "$CKPT" .pt)}"
exec "$HERE/../scripts/eval.sh" MP2-Walk-E2E-PlayRand-v0 "$CKPT" "$OUT" --reference phase2 --noisy
