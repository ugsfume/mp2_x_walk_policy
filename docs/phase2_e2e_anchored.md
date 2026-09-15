# Phase 2 — end-to-end policy, cloned from phase 1 and anchored to the clone

Task `MP2-Walk-E2E-v0` (`mp2_x_walk_policy/tasks/e2e_anchored.py`), agent
`agents.AnchoredPPORunnerCfg`, algorithm `algorithms.AnchorPPO`.

## Why not just train it from scratch

The 45-dim task (no clock, no prior) trained from scratch on this plant does
one of two things. Under the measured observation noise it collapses to
standing: the reward landscape from stance out to a walk is uphill through a
noise tax the policy never pays off, and 12 seeds across three noise
configurations all stood still. Under clean observations it walks, but into a
gait that slides its hind feet through stance — the reward's actual optimum
on this simulator, and one that reality pays somewhat less for.

Initialising from a clone of the phase-1 policy gets past the standing
collapse, but PPO drifts back into the sliding gait within ~50 iterations;
the first thing it harvests is foot clearance. Adding a loss term that pulls
the actor's mean toward the frozen clone holds the stepping family. With that
term decayed to zero over training the drift returns as it fades (clean at
β = 1.2, sliding by ~0.6, fully back by ~0.2), so β is held constant for the
whole run. The product is a policy trained under a standing anchor; that is
a deliberate trade.

## Step 1 — the clone

`phase2/run_distill.sh` (`distill/bc_collect.py`, `distill/bc_train.py`):

1. Export the phase-1 checkpoint to TorchScript; its deterministic mean
   action is the label source.
2. Roll it on the clean randomised-plant evaluation env: 512 envs × 400
   steps at each of 0.05 / 0.075 / 0.10 m/s = 614,400 pairs. The student
   sees the teacher's 47-dim observation minus the clock, with the
   previous-action slice rewritten to the student's own previous output
   (zeros at reset). The label is the executed joint target re-expressed in
   the student's contract: `clip((target − stand_pose) / 0.25, −1, 1)`;
   3.5% of labels clip.
3. Fit the actor MLP (128-128-64) by MSE, 40 epochs, with uniform input
   jitter at the clean noise floor. Seconds on a GPU. Save as an rsl_rl
   checkpoint (trained actor, fresh critic, fresh optimizer, std 0.8).

Validation MSE is not a useful gate: a clone with a lower MSE stood still
because its previous-action slice had been left in the teacher's encoding.
Judge a clone closed-loop on `MP2-Walk-E2E-PlayRand-v0`; the shipped clone
reads 0.061 m/s at a 2 Hz cadence with a clean foot path. Collection is unseeded, so a
fresh clone is statistically the same, not bit-identical — which is why the
shipped one is in the repo and its hash is pinned in the agent config.

## Step 2 — anchored PPO

**Observation (45)** — the phase-1 layout without the clock. Noise on the
four measured channels starts at the clean floor (0.05 / 0.02 / 0.01 / 0.2)
and ramps linearly to the measured amplitudes (1.0 / 0.30 / 0.174 / 4.8)
between iterations 150 and 450; observation delay 0–2 steps and hip
stiffness U(0.8, 1.0) as in phase 1. A second observation group, `anchor`,
is the same 45 values with no noise and no delay; only the anchor loss reads
it.

**Action (12)** — `target = stand_pose + 0.25 · clip(a, −1, 1)`.

**Command** — vx ∈ U(0.05, 0.25) m/s.

**Rewards** — the phase-1 tracking and regularisation terms; straightness
at half weight (yaw −0.40, lateral position −1.5, lateral velocity −0.40,
ang-vel tracking +0.50); and two gait terms instead of the rear-leg set:
`swing_progress` (+0.80): a lifted foot earns for moving forward relative to
the body, up to two feet at a time, only if it has been on the ground within
the last 0.4 s (a parked foot stops earning); `stance_foot_slip_l2` (−2.0,
2 N contact threshold).

**Reset randomisation** — stiffness and damping × U(0.5, 1.5), joint
friction × U(0.5, 2.0), all joints; hip stiffness × U(0.8, 1.0).

**Anchor** — every PPO minibatch adds `β · mean((μ_actor(obs_anchor) −
μ_clone(obs_anchor))²)` to the loss, β = 1.2012 constant. The value was
sized so that the anchor gradient on the actor equals the surrogate
gradient at 30% of the distance between the clone and an unanchored policy.
The critic never sees it. At β = 0 the update is stock PPO to the bit.

**Warm start** — the actor is loaded from the clone (`--resume`), the critic
and optimizer are fresh. 1300 iterations, checkpoint every 25.

```bash
phase2/run_train.sh --anchor runs/distill/bc_clone.pt     # or the shipped clone; ~21 min per seed
phase2/run_select.sh $ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk/<run>
```

## Step 3 — selection

Candidates are checkpoints after the noise ramp (≥ 550). Each is scored on
`MP2-Walk-E2E-PlayRand-v0` (`analysis/gate_table.py`): health, f0 ≥ 1.8 Hz,
clearance ≤ 12 mm, posture, and robustness under the full noise
(`MP2-Walk-E2E-Noisy-v0`). Among full passes, the fastest wins, and it must
then clear the open-loop bar: its own 0.10 m/s action stream, played back
blind on the randomised env, carries the body ≥ 660 mm in 400 steps
(`scripts/openloop_replay.sh`). A policy that only covers distance by
reacting to its observations fails that bar, and on the robot the
reaction is what breaks first.

The shipped checkpoint (seed 42, iteration 1050): 0.086 m/s, f0 1.94 Hz,
clearance 6.9 mm, 4-point rhythm loss under full noise, open-loop 681 mm.
All four seeds produce full-pass checkpoints after the ramp; one seed sits
on the f0 bar (1.73–1.89 across repeated evals of one checkpoint) and
needs the median-of-3 rule, which is why the rule exists. The seed also
decides how symmetric the gait comes out; train the sweep and select, do
not judge the recipe from one seed (`docs/environment.md`,
Reproducibility).

## On the floor

Glossy wood, 10 s at 0.10 m/s: 1030 / 1005 / 1000 mm forward, hind feet
planted (checked on video), heading +45 / +47 / +33° to the left. Scripted
controller the same day: 470–515 mm. The arc is open: the policy has no
heading or position in its observation, so it can only run symmetric and
hope; correction has to come from outside (a yaw-rate command term), not
from training. Its open-loop replay on the same floor went 630 / 650 mm
nearly straight (+10 / +2°), so the arc is amplified by the closed loop, on
this unit's asymmetry.
