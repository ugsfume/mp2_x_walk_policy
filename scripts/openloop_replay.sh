#!/usr/bin/env bash
# Open-loop replay bar: how far does the policy's own 0.10 m/s action stream
# carry the robot in simulation when played back blind?
#
#   scripts/openloop_replay.sh <play-task-id> <playrand-task-id> <checkpoint.pt> <out dir>
#
# 1. export the checkpoint (TorchScript + ONNX)
# 2. capture 400 steps per command on the Play id (1 env)
# 3. cut the 0.10 m/s block into a replay stream (and the robot target stream)
# 4. replay it on the PlayRand id, 32 envs, 400 steps, and report the median
#    spawn-frame displacement. Bar: >= 660 mm. This is the selection bar for a
#    checkpoint that will be put on the robot; a policy that only covers
#    distance by reacting to its observations does not pass it.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

PLAY="${1:?usage: openloop_replay.sh <play-id> <playrand-id> <checkpoint> <out dir>}"
PLAYRAND="${2:?playrand id}"
CKPT="$(readlink -f "${3:?checkpoint}")"
OUT="${4:?out dir}"
mkdir -p "$OUT"

python -m mp2_x_walk_policy.scripts.export_policy --task "$PLAY" --checkpoint "$CKPT" \
  --output_dir "$OUT/export" --device "$DEVICE" --viz none > "$OUT/export.log" 2>&1 || {
  echo "export failed, see $OUT/export.log" >&2; exit 1; }
grep -q EXPORT_INTERFACE_VERIFIED "$OUT/export.log" || { echo "export not verified" >&2; exit 1; }

python -m mp2_x_walk_policy.scripts.capture_policy_vectors --task "$PLAY" --checkpoint "$CKPT" \
  --export "$OUT/export/policy.pt" --label policy --output "$OUT/capture" --steps-per-command 400 \
  --device "$DEVICE" --viz none > "$OUT/capture.log" 2>&1 || { echo "capture failed, see $OUT/capture.log" >&2; exit 1; }
[[ -f "$OUT/capture.json" ]] || { echo "capture wrote no bundle" >&2; exit 1; }

python "$REPO_ROOT/analysis/capture_to_stream.py" --capture "$OUT/capture.json" --command-x 0.10 --output "$OUT/stream_010" > "$OUT/stream.log" 2>&1

python -m mp2_x_walk_policy.scripts.evaluate_policy --task "$PLAYRAND" --scripted_agent replay \
  --replay_file "$OUT/stream_010_replay.npz" --num_envs 32 --record_envs 32 --steps 400 --command_x 0.10 \
  --contact_threshold 2.0 --output_dir "$OUT/replay" --device "$DEVICE" --viz none > "$OUT/replay.log" 2>&1 || {
  echo "replay failed, see $OUT/replay.log" >&2; exit 1; }

python - "$OUT/replay" <<'PY'
import glob, json, sys
s = json.load(open(glob.glob(sys.argv[1] + "/*_summary.json")[0]))
resets = float(s.get("done_count_mean", 0)) + float(s.get("timeout_count_mean", 0))
disp = 1e3 * s["forward_displacement_spawnframe_m"]
print(f"open-loop replay: median spawn-frame displacement {disp:.1f} mm over 400 steps, {s['num_envs']} envs, resets {resets:.1f}")
print("OPENLOOP_BAR:", "PASS" if disp >= 660 and resets == 0 else "FAIL", "(bar 660 mm, no resets)")
PY
