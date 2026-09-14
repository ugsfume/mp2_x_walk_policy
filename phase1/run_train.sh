#!/usr/bin/env bash
# Phase 1: train the CPG-residual policy. 4 seeds x 500 iterations, 512 envs.
#
#   phase1/run_train.sh [--seeds "42 1 2 3"] [--iters 500] [--tag cpg_residual]
#
# ~6 min per seed on an RTX 5080. Checkpoints every 50 iterations under
# $ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk/<timestamp>_<tag>_s<seed>/.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="42 1 2 3"; ITERS=500; TAG=cpg_residual
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
exec "$HERE/../scripts/train.sh" MP2-Walk-CpgResidual-v0 --seeds "$SEEDS" --iters "$ITERS" --tag "$TAG"
