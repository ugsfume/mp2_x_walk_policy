#!/usr/bin/env bash
# Render one 9.8 s clip of a checkpoint walking at 0.10 m/s (1280x720, 50 fps).
#
#   scripts/render.sh <play-task-id> <checkpoint.pt> <out dir> [camera view]
#
# Camera views: gait (default; foot-level side view framing ~1 m of travel),
# gait_wide, side, rear_quarter, top_oblique. The camera is static in recorded
# video. Output: <out dir>/videos/model_*-step-0.mp4 plus the usual metrics.
# Needs an RTX-capable GPU; ~1 min.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

TASK="${1:?usage: render.sh <play-task-id> <checkpoint> <out dir> [camera view]}"
CKPT="$(readlink -f "${2:?checkpoint}")"
OUT="${3:?out dir}"
VIEW="${4:-gait}"
python -m mp2_x_walk_policy.scripts.evaluate_policy --task "$TASK" --scripted_agent policy \
  --checkpoint "$CKPT" --num_envs 1 --record_envs 1 --steps 490 --command_x 0.10 \
  --contact_threshold 2.0 --video --camera_view "$VIEW" --output_dir "$OUT" --device "$DEVICE" \
  > "$OUT.log" 2>&1 || { echo "render failed, see $OUT.log" >&2; exit 1; }
ls "$OUT"/videos/*.mp4
