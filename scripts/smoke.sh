#!/usr/bin/env bash
# Instantiate every task id and step it with a zero agent (8 envs, 30 steps).
# Catches import, asset and config errors in ~30 s per id, before a training
# run is launched. The Isaac Lab CLI can turn an import error into a clean
# exit, so the check is on the log, not the exit code.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

IDS=(MP2-Walk-CpgResidual-v0 MP2-Walk-CpgResidual-Play-v0 MP2-Walk-CpgResidual-PlayRand-v0
     MP2-Walk-E2E-v0 MP2-Walk-E2E-Play-v0 MP2-Walk-E2E-PlayRand-v0 MP2-Walk-E2E-Noisy-v0)
FAIL=0
for id in "${IDS[@]}"; do
  out="$RUNS_DIR/smoke/$id"; mkdir -p "$out"
  python -m mp2_x_walk_policy.scripts.evaluate_policy --task "$id" --scripted_agent zero \
    --num_envs 8 --steps 30 --device "$DEVICE" --viz none --output_dir "$out" > "$out.log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]] && grep -q "Parsing configuration from" "$out.log" && ls "$out"/*_summary.json >/dev/null 2>&1; then
    echo "ok    $id"
  else
    echo "FAIL  $id (rc=$rc, see $out.log)"; FAIL=1
  fi
done
[[ $FAIL -eq 0 ]] && echo "SMOKE: PASS" || { echo "SMOKE: FAIL"; exit 1; }
