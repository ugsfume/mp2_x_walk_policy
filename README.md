# mp2_x_walk_policy

Forward-walking policies for the Mini Pupper V2, trained in Isaac Lab with
PPO, and everything needed to train them again, evaluate them, and put them
on a robot.

Two policies ship, and the second is built from the first:

| | phase 1 — CPG-residual | phase 2 — end-to-end |
|---|---|---|
| observation | 47: IMU, joint pos/vel, command, previous action, gait clock | 45: the same without the clock |
| action | residual on a fixed 2 Hz diagonal-trot prior (±0.12 rad) | joint targets around the stand pose (±0.25 rad) |
| training | PPO from scratch, 500 iterations | PPO warm-started from a clone of phase 1 and pulled toward it by a constant anchor loss, 1300 iterations |
| glossy wood, 10 s at 0.10 m/s | 915 / 885 / 870 mm, heading within ±8° | 1030 / 1005 / 1000 mm, heading +33…+47° (left) |
| scripted controller, same floor, same day | 515 mm | 470–515 mm |

The end-to-end policy walks 2.15x as far as the stock scripted controller
and plants its feet; its one open defect is the left arc. Both were
developed, tuned and tested on **one** Mini Pupper V2 unit. The simulated
plant and the observation-noise profile are measurements of that unit
(`plant/`), and the arc is partly that unit's left-side servo weakness.
Expect to re-measure before expecting the same numbers.

## How the pieces fit

```
                measured plant + noise profile (plant/, docs/methodology.md)
                                     |
   phase1/run_train.sh   PPO  ------+------>  cpg_residual_s3_499.pt   (teacher)
                                     |                 |
   phase2/run_distill.sh  roll the teacher, fit a 45-dim clone  ---->  bc_clone.pt
                                     |                 |
   phase2/run_train.sh   PPO from the clone, anchored to the clone -->  e2e_anchored_s42_1050.pt
                                     |
   scripts/eval.sh, analysis/gate_table.py      pick a checkpoint
   scripts/openloop_replay.sh                   ... that also walks blind
   scripts/export_policy, deploy/               ONNX + a numpy contract for the robot
```

Why the detour through the clone: trained from scratch, the 45-dim task
either collapses to standing (under realistic observation noise) or settles
into a foot-sliding gait that the reward likes and the floor pays for less.
The phase-1 prior fixes the gait family; the clone carries it into the
end-to-end contract; the anchor keeps PPO from drifting back out of it.
Details and the evidence for each choice: `docs/phase2_e2e_anchored.md`.

## Quick start

Install Isaac Sim 6.0.1, Isaac Lab (the commit in `docs/environment.md`) and
rsl_rl 5.0.1 first. Then, with that environment active:

```bash
export ISAACLAB_DIR=~/IsaacLab
scripts/smoke.sh                                    # every task instantiates, ~4 min
phase2/run_eval.sh                                  # the shipped policy against the gates, ~2 min
phase1/run_train.sh                                 # 4 seeds x 500 iterations, ~35 min on an RTX 5080
phase2/run_distill.sh --teacher <phase-1 model.pt>  # teacher -> clone, ~3 min
phase2/run_train.sh --anchor runs/distill/bc_clone.pt   # 4 seeds x 1300, ~1.5 h
phase2/run_select.sh $ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk/<run>   # ladder -> candidate
```

`scripts/render.sh MP2-Walk-E2E-Play-v0 <model.pt> runs/clip` records a
9.8 s side-view clip. `docs/deployment.md` covers the export and the exact
observation/action contract the robot side has to implement;
`deploy/check_onnx.py` verifies an export against the shipped fixtures.

## What is where

```
mp2_x_walk_policy/   the Isaac Lab task package: tasks/{cpg_residual,e2e_anchored}.py on
                     tasks/base.py; plant.py (measured numbers); algorithms.py (anchored PPO);
                     scripts/ (evaluate, export, capture)
phase1/, phase2/     run scripts, shipped checkpoints, the recorded training configs
distill/             clone dataset collector and trainer
analysis/            gate table, rhythm scorer, open-loop stream tools, noise estimator
deploy/              numpy contract, ONNX bundles with fixtures, ONNX check
plant/               the measured plant and noise profile, with how each number was obtained
assets/              the robot USD (mass-corrected) and how it was built
tools/               check_record.py (configs vs the recorded training dumps), compare_checkpoints.py
docs/                environment, phase 1, phase 2, methodology, deployment
```

## Reproduced from this repo

Everything below was run from this repository as published, on one RTX 5080
(Isaac Sim 6.0.1, Isaac Lab `ffff603`, rsl_rl 5.0.1), with the scripts above
and no other input than the shipped checkpoints.

**Phase 1** — `phase1/run_train.sh`, 4 seeds × 500 iterations, 8–10 min
each; `model_499` on `MP2-Walk-CpgResidual-PlayRand-v0`:

| seed | speed m/s | f0 Hz | clearance mm | pitch ° | gates |
|---|---|---|---|---|---|
| 42 | 0.105 | 2.04 | 6.6 | 1.00 | pass |
| 1 | 0.098 | 2.04 | 5.6 | 1.16 | pass |
| 2 | 0.108 | 2.04 | 8.5 | 1.49 | pass |
| 3 | 0.100 | 2.04 | 6.1 | 0.22 | pass — **byte-identical to the shipped checkpoint** |

**Clone** — `phase2/run_distill.sh` from the seed-3 checkpoint: 614,400
pairs, 3.54% clipped labels (shipped clone: 3.54%), closed-loop 0.058 m/s,
f0 2.04 Hz, clearance 6.1 mm, all gates pass (shipped clone: 0.061 m/s).

**Phase 2** — `phase2/run_train.sh` from the shipped clone, 4 seeds × 1300
iterations, 21–24 min each; `model_1050` on `MP2-Walk-E2E-PlayRand-v0` and
the robustness gate on `MP2-Walk-E2E-Noisy-v0`:

| seed | speed m/s | f0 Hz | clearance mm | rhythm loss under noise | speed kept | gates |
|---|---|---|---|---|---|---|
| 42 | 0.083 | 1.94 | 8.6 | 3.7 pts | 1.05 | pass — **byte-identical to the shipped checkpoint** |
| 1 | 0.085 | 1.84 | 10.5 | 2.8 pts | 0.90 | pass |
| 2 | 0.082 | 1.94 | 8.9 | 6.7 pts | 0.90 | pass |
| 3 | 0.085 | 1.73 | 11.3 | 2.3 pts | 0.86 | f0 below 1.80 at 1050 (three evals: 1.73 / 1.73 / 1.89); passes at iteration 1150 (f0 1.84, 0.084 m/s) |

`phase2/run_select.sh` on the seed-42 run: 7 of 8 post-ramp checkpoints
pass every gate (one f0 within scatter of the bar), the fastest (iteration
550, 0.087 m/s) clears the open-loop bar at 755 mm. Speed differences of a
few mm/s between post-ramp checkpoints are evaluation scatter, not
learning; the shipped iteration-1050 checkpoint is one full pass among
several.

Training is deterministic on one machine, so the two byte-identical
checkpoints are expected; on other hardware expect the same tables within
evaluation scatter, not the same bytes.

## Evaluation in simulation, and what it can't tell you

`scripts/eval.sh` runs 32 randomised 9.8 s episodes at 0.10 m/s and scores
health (torque, speed, bounce), gait family (fundamental frequency, swing
clearance), posture, and robustness to the measured observation noise
(`analysis/gate_table.py`). The shipped phase-2 checkpoint on those gates:
0.086 m/s, 1.94 Hz, 6.9 mm clearance, 4-point rhythm loss under full noise.

Simulation is a good filter and a poor ranker. It did not predict the left
arc, it does not reproduce how a foot catches on a high-friction mat, and it
cannot see a foot sliding through stance the way a camera can. Two things
told us more than any sim number: tape on the floor, and video of the feet.

## License

Apache-2.0. The robot description the asset derives from is MangDang's
(Apache-2.0); see `NOTICE`.
