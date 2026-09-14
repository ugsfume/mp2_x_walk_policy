#!/usr/bin/env bash
# Distil the phase-1 teacher into the 45-dim clone, then evaluate the clone.
#
#   phase2/run_distill.sh [--teacher <phase-1 checkpoint.pt>] [--out runs/distill] [--epochs 40] [--num-envs 512] [--steps 400]
#
# 1. export the teacher to TorchScript (its deterministic mean is the label source)
# 2. collect 512 envs x 400 steps x {0.05, 0.075, 0.10} m/s pairs (~1 min GPU)
# 3. fit the clone, 40 epochs with input jitter (~seconds)
# 4. evaluate the clone closed-loop on the end-to-end PlayRand id and print the gate table
# The clone that ships (phase2/checkpoints/bc_clone.pt) came out of exactly
# this recipe; collection is unseeded, so yours will differ slightly.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

TEACHER="$REPO_ROOT/phase1/checkpoints/cpg_residual_s3_499.pt"; OUT="$REPO_ROOT/runs/distill"; EPOCHS=40; ENVS=512; STEPS=400
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher) TEACHER="$(readlink -f "$2")"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-envs) ENVS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
mkdir -p "$OUT"

python -m mp2_x_walk_policy.scripts.export_policy --task MP2-Walk-CpgResidual-Play-v0 --checkpoint "$TEACHER" \
  --output_dir "$OUT/teacher_export" --device "$DEVICE" --viz none > "$OUT/export.log" 2>&1 || {
  echo "teacher export failed, see $OUT/export.log" >&2; exit 1; }
grep -q EXPORT_INTERFACE_VERIFIED "$OUT/export.log" || { echo "export not verified" >&2; exit 1; }

python "$REPO_ROOT/distill/bc_collect.py" --teacher "$OUT/teacher_export/policy.pt" --output "$OUT/pairs" \
  --num-envs "$ENVS" --steps "$STEPS" --device "$DEVICE" --viz none > "$OUT/collect.log" 2>&1 || {
  echo "collection failed, see $OUT/collect.log" >&2; exit 1; }
grep -q "BC_COLLECT: PASS" "$OUT/collect.log" || { echo "collection did not pass" >&2; exit 1; }
grep BC_COLLECT_META "$OUT/collect.log" | sed 's/BC_COLLECT_META //' | python -c "import json,sys; m=json.load(sys.stdin); print(f\"pairs: {m['rows']} rows, clipped labels {100*m['clip_fraction_overall']:.2f}%\")"

python "$REPO_ROOT/distill/bc_train.py" --dataset "$OUT/pairs.npz" --output "$OUT/bc_clone.pt" --epochs "$EPOCHS" --augment \
  > "$OUT/train.log" 2>&1 || { echo "clone training failed, see $OUT/train.log" >&2; exit 1; }
tail -2 "$OUT/train.log"
sha256sum "$OUT/bc_clone.pt"

"$REPO_ROOT/scripts/eval.sh" MP2-Walk-E2E-PlayRand-v0 "$OUT/bc_clone.pt" "$OUT/eval" || true
echo "clone: $OUT/bc_clone.pt  (pass it to phase2/run_train.sh --anchor)"
