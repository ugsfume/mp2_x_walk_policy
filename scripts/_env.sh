# Shared by every script in scripts/, phase1/ and phase2/. Source, do not run.
#
# Assumes the Isaac Lab python environment is already active (conda/venv) and
# that ISAACLAB_DIR points at the Isaac Lab checkout. Nothing else is needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
DEVICE="${DEVICE:-cuda:0}"

[[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || {
  echo "ISAACLAB_DIR=$ISAACLAB_DIR does not contain isaaclab.sh" >&2; exit 64; }

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MP2_X_WALK_ROOT="$REPO_ROOT"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs}"
mkdir -p "$RUNS_DIR"

# Isaac Lab keeps rsl_rl checkpoints under its own tree.
LOG_ROOT="$ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk"
