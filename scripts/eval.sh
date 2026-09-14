#!/usr/bin/env bash
# Evaluate a checkpoint on a PlayRand id and score it against the gates.
#
#   scripts/eval.sh <task-id> <checkpoint.pt> <out dir> [--reference phase1|phase2] [--noisy]
#
# 32 randomised envs, 8 recorded, 490 steps (9.8 s) at 0.10 m/s, contact
# threshold 2 N -- the settings the gate bars were frozen with. Writes
# <out dir>/model_*_{summary.json,gait_traces.npz,eval.csv,gait.png} and
# prints the gate table. With --noisy the E2E checkpoint is also run on
# MP2-Walk-E2E-Noisy-v0 (into <out dir>_noisy) for the robustness gate.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

TASK="${1:?usage: eval.sh <task-id> <checkpoint> <out dir> [--reference phaseN] [--noisy]}"
CKPT="$(readlink -f "${2:?checkpoint}")"
OUT="${3:?out dir}"; shift 3
REF=(); NOISY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference) REF=(--reference "$2"); shift 2 ;;
    --noisy) NOISY=1; shift ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

evaluate() { # evaluate <task> <out>
  python -m mp2_x_walk_policy.scripts.evaluate_policy --task "$1" --scripted_agent policy \
    --checkpoint "$CKPT" --num_envs 32 --record_envs 8 --steps 490 --command_x 0.10 \
    --contact_threshold 2.0 --output_dir "$2" --device "$DEVICE" --viz none > "$2.log" 2>&1 || {
    echo "evaluation failed, see $2.log" >&2; exit 1; }
  grep -q "Parsing configuration from" "$2.log" || { echo "task did not load, see $2.log" >&2; exit 1; }
}

mkdir -p "$(dirname "$OUT")"
evaluate "$TASK" "$OUT"
NOISY_ARG=()
if [[ $NOISY -eq 1 ]]; then
  evaluate MP2-Walk-E2E-Noisy-v0 "${OUT}_noisy"
  NOISY_ARG=(--noisy "${OUT}_noisy")
fi
python "$REPO_ROOT/analysis/gate_table.py" "$OUT" "${REF[@]}" "${NOISY_ARG[@]}" --json "$OUT/gate_table.json"
