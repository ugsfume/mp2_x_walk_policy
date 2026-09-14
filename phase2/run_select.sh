#!/usr/bin/env bash
# Score a phase-2 checkpoint ladder and pick a candidate.
#
#   phase2/run_select.sh <run dir> [--from 550] [--every 100] [--out runs/select_<tag>]
#
# Evaluates model_<k>.pt for k >= --from on the PlayRand id (gate table each),
# then runs the open-loop replay bar on the fastest full-pass checkpoint.
# Checkpoints before the noise ramp has finished (iteration 450) trained at
# the start amplitudes and are not candidates; --from 550 is the default.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

RUN="${1:?usage: run_select.sh <run dir> [--from N] [--every N] [--out dir]}"; shift
FROM=550; EVERY=100; OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
OUT="${OUT:-$RUNS_DIR/select_$(basename "$RUN")}"
mkdir -p "$OUT"

best=""; best_vx=0
for ckpt in $(ls "$RUN"/model_*.pt | sort -t_ -k2 -n); do
  k=$(basename "$ckpt" .pt); k=${k#model_}
  (( k >= FROM )) || continue
  (( (k - FROM) % EVERY == 0 )) || [[ "$ckpt" == "$(ls "$RUN"/model_*.pt | sort -t_ -k2 -n | tail -1)" ]] || continue
  echo "== model_$k"
  if "$REPO_ROOT/scripts/eval.sh" MP2-Walk-E2E-PlayRand-v0 "$ckpt" "$OUT/model_$k" --reference phase2 > "$OUT/model_$k.table" 2>&1; then
    vx=$(python -c "import json; print(json.load(open('$OUT/model_$k/gate_table.json'))['row']['vx'])")
    echo "   full pass, vx $vx"
    if python -c "import sys; sys.exit(0 if float('$vx') > float('$best_vx') else 1)"; then best="$ckpt"; best_vx="$vx"; fi
  else
    grep -E "FAIL|INVALID" "$OUT/model_$k.table" | head -3 | sed 's/^/   /'
  fi
done
[[ -n "$best" ]] || { echo "no checkpoint passed every gate"; exit 1; }
echo "fastest full-pass checkpoint: $best (vx $best_vx); running the open-loop replay bar"
"$REPO_ROOT/scripts/openloop_replay.sh" MP2-Walk-E2E-Play-v0 MP2-Walk-E2E-PlayRand-v0 "$best" "$OUT/openloop_$(basename "$best" .pt)"
