# Provenance

What the shipped artifacts are and where they came from. Internal run names
appear only here.

## Checkpoints

| file | trained as | sha256 |
|---|---|---|
| `phase1/checkpoints/cpg_residual_s3_499.pt` | run `mp2_e2e_v35_s3_500iter`, seed 3, iteration 499, 500 iterations x 512 envs | `adfee7b7f664229994d1b4a45159ed9d41da13e56de4ed433235e992b5ead35c` |
| `phase2/checkpoints/bc_clone.pt` | clone of the file above (614,400 pairs, 40 epochs, seed 0, input jitter) | `92cb5922abd4e942a741a196c8c06f10a47e9e6151af9c94d29d03d6ad6cc74b` |
| `phase2/checkpoints/e2e_anchored_s42_1050.pt` | run `mp2_e2e_v39_s42_1300iter`, seed 42, iteration 1050, warm-started from and anchored to the clone | `1f8de3f9cdb7db27deb70fdcd70167f1f722eab7548b3e558668d0daacdd40cd` |

The checkpoints were produced by an earlier, layered version of the task
package. `phase*/record/{env,agent}.yaml` are the resolved configs Isaac
Lab dumped for those two runs (absolute paths replaced by placeholders; the
agent config still carries the original experiment and run names). The
configs in this repo are checked against them by `tools/check_record.py`,
and 10 training iterations at seed 42 under either package give
bit-identical checkpoints (`tools/compare_checkpoints.py`).

## Exported policies (deploy/models/)

| bundle | from | policy.onnx sha256 |
|---|---|---|
| `cpg_residual_s3_499` | the phase-1 checkpoint above | `39d098129c3111e5e03f247bb81011c85a55e990c4dc7048156fa337da48f370` |
| `e2e_anchored_s42_1050` | the phase-2 checkpoint above | `6cc19ed577298ca81ee118355df04cdd1f7b1e20d583b3dc9c138af439e36d8e` |

Exported with rsl_rl 5.0.1 `export_policy_to_onnx` (torch 2.10.0+cu128,
onnx 1.21). The fixtures in each bundle were produced from the same
checkpoint by the TorchScript export on CPU (`fixtures`) and by a
closed-loop rollout in simulation (`trajectory_fixtures`).

## Asset

`assets/mini_pupper_2_isaac_v27/`: URDF converter output of the
mass-corrected description (see `assets/build/README.md`), contact
reporting enabled on 30 bodies. Generator comments carrying local paths were
removed and the binary geometry layer re-saved; prim-by-prim content is
unchanged. Tree hash (sha256 of the sorted per-file sha256 list):
`252c761731f99a4352b12fa46a0de92b05e252811c6ecaf10a98f8abd597ff36`.

## Floor results quoted in the README

Glossy wood, fresh battery, 10 s at 0.10 m/s commanded, tape-measured
forward travel of the body, three runs each, same day as the scripted
controller runs quoted next to them. Evaluated 2026-08-13 (phase 1) and
2026-08-16 (phase 2).
