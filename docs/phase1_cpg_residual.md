# Phase 1 — CPG-residual policy

Task `MP2-Walk-CpgResidual-v0` (`mp2_x_walk_policy/tasks/cpg_residual.py`).
The policy adds a bounded residual to a fixed diagonal-trot trajectory. The
prior decides the gait family (2 Hz trot, 0.22 s stances); PPO learns balance
and propulsion around it.

## Task

**Observation (47, ROS joint order)**

| slice | term | training noise (uniform ±) |
|---|---|---|
| 0:3 | base angular velocity, body frame | 1.0 rad/s |
| 3:6 | gravity direction in the body frame | 0.30 |
| 6:9 | velocity command (vx, 0, 0) | — |
| 9:21 | joint position − stand pose | 0.05 rad |
| 21:33 | joint velocity | 4.8 rad/s |
| 33:45 | previous action (clipped) | — |
| 45:47 | gait clock [sin φ, cos φ], φ = 2π·2 Hz·t | — |

The four noisy terms also pass through a per-env observation delay of 0–2
control steps (0–40 ms), resampled every episode. Amplitudes are measured,
not tuned (`plant/obs_noise_profile.yaml`).

**Action (12)** — residual `a ∈ [-1, 1]`, applied as

```
target = stand_pose + cpg(φ, vx) + 0.12 · a
cpg:  hip  += 0.16 · s · cos(φ + φ_leg)          φ_leg = 0 (LF, RB), π (RF, LB)
      knee −= 0.20 · s · max(sin(φ + φ_leg), 0)   s = clip(|vx| / 0.10, 0.5, 1.5)
```

**Command** — vx ∈ U(0.05, 0.15) m/s per episode, vy = ωz = 0.

**Rewards** (Isaac Lab terms unless named; std 0.10 on the exp terms)

| term | weight |
|---|---|
| track_lin_vel_xy_exp | +2.0 |
| track_ang_vel_z_exp | +0.75 |
| lin_vel_z_l2 / ang_vel_xy_l2 | −1.0 / −0.10 |
| dof_torques_l2 / dof_acc_l2 / action_rate_l2 | −1e-4 / −1e-7 / −0.02 |
| flat_orientation_l2 / dof_pos_limits | −1.0 / −0.05 |
| yaw_l2 / lateral_position_l2 / lateral_velocity_l2 (world frame) | −0.80 / −3.0 / −0.80 |
| stance_foot_slip_l2 (all feet, 1 N) / front feet only | −0.05 / −0.03 |
| rear_contact_transitions / rear_air_fraction | +0.05 / +0.01 |
| front_rear_transition_imbalance_l2 | −0.02 |
| rear_relative_swing_clearance (10 mm) / front_rear_relative_lift_imbalance_l2 | +0.08 / −0.05 |
| diagonal_trot_contact_match / diagonal_trot_swing_clearance | +0.04 / +0.04 |

The rear-leg terms exist because without them the residual learned to walk
on the front legs and drag the rear. The two diagonal-trot terms reward
matching the prior's contact schedule.

**Terminations** — chassis contact (any of 5 body sensors > 1 N), tilt > 0.7
rad, 10 s timeout. **Reset randomisation** — hip stiffness scaled U(0.8, 1.0)
per hip per episode (the unit delivers less swing on one side); otherwise
deterministic resets in the stand pose. **Plant** — `plant/measured_plant.yaml`.

**PPO** (`agents.PPORunnerCfg`) — actor/critic MLP 128-128-64 ELU, init std
0.8, lr 1e-3 adaptive (KL 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.005,
5 epochs × 4 minibatches, 24 steps × 512 envs per iteration, 500 iterations.

## Train

```bash
phase1/run_train.sh                    # seeds 42 1 2 3, ~8 min each on an RTX 5080
phase1/run_eval.sh <run>/model_499.pt  # gate table vs the shipped checkpoint's row
```

All four seeds train; the shipped checkpoint is seed 3 at iteration 499. On
`MP2-Walk-CpgResidual-PlayRand-v0` (32 randomised starts, 0.10 m/s) it
reads 0.100 m/s, f0 2.04 Hz, harmonic share 0.97, swing clearance 6 mm,
pitch 0.2°. The clock makes f0 a near-certainty; what varies across seeds
is speed and clearance.

## On the floor

Glossy wood, 10 s at 0.10 m/s: 915 / 885 / 870 mm forward, heading +3 / −1
/ −8°, rhythm held. The same-day scripted controller: 515 mm. The policy
corrects the unit's left-side weakness by feel: the same prior played back
blind arcs left, and an earlier policy on this prior trained without the
noise profile arced further and lost 40% of its distance to observation
thrash. This one goes straight.

On a padded high-friction mat it did not transfer: 560–740 mm with a
30–70° left arc, the weak side catching instead of sliding. This is a
wood-proven policy.

## Where it leads

Its clock-locked gait is the reason it is the teacher for phase 2: it has the
stepping form we want, and the clone (`phase2/run_distill.sh`) carries that
form into the 45-dim contract without the clock.
