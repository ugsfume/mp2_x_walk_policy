#!/usr/bin/env bash
# Phase 2: anchored PPO from the clone. 4 seeds x 1300 iterations, 512 envs.
#
#   phase2/run_train.sh [--anchor <clone.pt>] [--seeds "42 1 2 3"] [--iters 1300] [--tag e2e_anchored]
#
# The actor is warm-started from the clone AND anchored to it; both use the
# same file. Default: the shipped phase2/checkpoints/bc_clone.pt. Pass
# --anchor to use one you distilled (phase2/run_distill.sh); its sha256 is
# computed here and recorded in the run's params/agent.yaml.
# ~20 min per seed on an RTX 5080; checkpoints every 25 iterations.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"

ANCHOR="$REPO_ROOT/phase2/checkpoints/bc_clone.pt"; SEEDS="42 1 2 3"; ITERS=1300; TAG=e2e_anchored
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anchor) ANCHOR="$(readlink -f "$2")"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
[[ -f "$ANCHOR" ]] || { echo "clone not found: $ANCHOR" >&2; exit 66; }
SHA="$(sha256sum "$ANCHOR" | cut -d' ' -f1)"
echo "clone: $ANCHOR"
echo "sha256: $SHA"

# rsl_rl resolves --load_run under the experiment's log root, so the clone is
# placed there as a one-checkpoint "run".
WARM="$LOG_ROOT/warmstart_bc_clone"
mkdir -p "$WARM"
cp -f "$ANCHOR" "$WARM/model_0.pt"

exec "$REPO_ROOT/scripts/train.sh" MP2-Walk-E2E-v0 --seeds "$SEEDS" --iters "$ITERS" --tag "$TAG" -- \
  --resume --load_run 'warmstart_bc_clone$' --checkpoint 'model_0.pt$' \
  "agent.algorithm.anchor_path=$ANCHOR" "agent.algorithm.anchor_sha256=$SHA"
