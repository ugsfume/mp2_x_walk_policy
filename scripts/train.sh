#!/usr/bin/env bash
# Train one task for one or more seeds, sequentially.
#
#   scripts/train.sh <task-id> [--seeds "42 1 2 3"] [--iters N] [--envs N]
#                    [--tag NAME] [-- <extra isaaclab train args>]
#
# Checkpoints land in $ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk/<timestamp>_<tag>_s<seed>/,
# console output in runs/<tag>_s<seed>.log. Extra args after `--` go straight
# to `isaaclab.sh train` (resume flags, `agent.*` / `env.*` overrides).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

TASK="${1:?usage: train.sh <task-id> [options] [-- extra args]}"; shift
SEEDS="42 1 2 3"; ITERS=""; ENVS=512; TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --envs) ENVS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
TAG="${TAG:-${TASK//MP2-Walk-/}}"
ITER_ARG=(); [[ -n "$ITERS" ]] && ITER_ARG=(--max_iterations="$ITERS")

for seed in $SEEDS; do
  run_name="${TAG}_s${seed}"
  log="$RUNS_DIR/${run_name}.log"
  echo "== $TASK seed $seed -> $run_name ($log)"
  ( cd "$ISAACLAB_DIR" && ./isaaclab.sh train --rl_library rsl_rl \
      --task="$TASK" --external_callback mp2_x_walk_policy.register_tasks \
      --num_envs="$ENVS" "${ITER_ARG[@]}" --seed="$seed" --device="$DEVICE" \
      --viz none --run_name "$run_name" "$@" ) >"$log" 2>&1 || {
    echo "FAILED seed=$seed (see $log)" >&2; exit 1; }
  # The Isaac Lab CLI can exit 0 after a configuration error; require evidence
  # that training actually ran.
  grep -q "Learning iteration" "$log" || { echo "no training iterations in $log" >&2; exit 1; }
  grep -E "Training time|Total time" "$log" | tail -1 || true
done
echo "TRAIN_COMPLETE $TASK seeds=[$SEEDS]"
