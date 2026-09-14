# Deployment — the policy contract

What the network expects and what it emits, so a robot-side loop can run an
exported policy exactly as the training environment did. `deploy/contract.py`
is the reference implementation (numpy + onnxruntime) and
`deploy/check_onnx.py` proves an export against fixtures.

## Export

```bash
python -m mp2_x_walk_policy.scripts.export_policy --task MP2-Walk-E2E-Play-v0 \
    --checkpoint phase2/checkpoints/e2e_anchored_s42_1050.pt --output_dir runs/export --viz none
python deploy/check_onnx.py runs/export --fixtures deploy/models/e2e_anchored_s42_1050
```

`policy.onnx` (+ `policy.onnx.data`) takes `obs` of shape (1, 45) or (1, 47)
float32 and returns `actions` (1, 12). The input width identifies the
contract. The second command replays the shipped fixtures through your
export: synthetic observations → actions (≤ 1e-5 from the TorchScript
reference), the action decoding rows (exact), and a 1200-row closed-loop
rollout. Run it on the target CPU too; ONNX on ARM differs from x86 by
~1e-6 and no more.

## Observation (float32, ROS joint order: LF, RF, LB, RB × abduction, hip, knee)

| slice | content | unit / convention |
|---|---|---|
| 0:3 | base angular velocity, body frame | rad/s, gyro bias removed |
| 3:6 | gravity direction in the body frame | unit vector; upright ≈ (0, 0, −1) |
| 6:9 | velocity command | (vx, 0, 0) m/s; the policies were trained on vx only |
| 9:21 | joint position − stand pose | rad; stand pose = (0, 0.994, −1.767) per leg |
| 21:33 | joint velocity | rad/s |
| 33:45 | previous action **after clipping** to [−1, 1] | zeros on reset |
| 45:47 | gait clock [sin φ, cos φ], φ = 2π·(tick mod 25)/25 | CPG contract only; tick 0 on reset |

Nothing is normalised. `projected_gravity = −accel_body / |accel_body|`;
angular velocity = body-frame gyro minus a bias estimated while the robot
stands still before the run, converted deg/s → rad/s.

## Action

```
clipped = clip(raw, −1, 1)
target  = stand_pose + cpg(tick, vx) + 0.12 · clipped     (47-dim, CPG contract)
target  = stand_pose + 0.25 · clipped                     (45-dim, end-to-end)

cpg, per leg with phase offset 0 (LF, RB) or π (RF, LB), s = clip(|vx|/0.10, 0.5, 1.5):
    hip  += 0.16 · s · cos(φ + offset)
    knee −= 0.20 · s · max(sin(φ + offset), 0)
```

Targets are joint angles in radians in the same order as the observation.
The previous-action slice for the next step is `clipped`, not `raw` and not
the target.

## The loop

50 Hz, wall-clock paced (`tick` advances once per step whether or not a
fresh sensor sample arrived). On reset: tick = 0, previous action = 0,
robot standing in the stand pose. `PolicyRunner` in `deploy/contract.py`
holds exactly this state.

Two things the training environment does that a robot does not do by
itself:

- **The episode starts in the stand pose.** Command the stand pose and let
  the robot settle before the first policy step. Nothing downstream limits
  the rate of a position command, so move into the pose with a ramp (a 2 s
  smoothstep from the measured pose is enough), not a step.
- **Stopping means standing, not zero command.** The policies were never
  trained to stand still; on stop, hold the stand pose. Servo drivers
  commonly latch the last command, so an exiting process that does not
  publish a resting pose leaves the robot frozen mid-stride.

## What you supply

Everything below depends on your driver and mounting. The values our unit
used are given as an example, not as a specification.

- **Servo ticks ↔ radians and knee convention.** Our driver:
  `tick = 512 − (550/π)·(θ − neutral)·direction`, neutral (0, π/4, −π/4) for
  (abduction, hip, knee), with the knee angle *serial* (relative to the
  hip link) in the policy contract while the servo bus wanted the absolute
  knee (hip + serial knee). Get this wrong and the policy sees the right
  numbers and moves the wrong joints; a joint-order or sign error is the
  first thing to rule out when an exported policy misbehaves.
- **IMU mounting frame.** A fixed 3×3 rotation from the sensor axes to the
  body frame (x forward, y left, z up). Ours was `[[0,−1,0],[−1,0,0],[0,0,−1]]`.
  Check it by tilting the robot: `projected_gravity` must follow the tilt
  and read (0, 0, −1) upright.
- **Joint velocity.** The servo reports positions only. We fitted a slope
  over the last three position samples (60 ms) and held the previous value
  when a sample was stale. Whatever you do, measure its noise with the
  replay procedure in `docs/methodology.md` — the training noise model
  (4.8 rad/s) is for a 50 Hz finite difference of a 10-bit readout.
- **Loop timing.** Measure it. Our reference loop held 50.0 ± 0.1 Hz with a
  p99 tick period of 22 ms on a Raspberry Pi 4 running onnxruntime on the
  CPU; a loop that slips to 40 Hz changes the plant the policy was trained
  for.

## Checking your integration without a policy

1. Command the stand pose; read back positions; the observation slice 9:21
   must be ≈ 0 and slice 3:6 ≈ (0, 0, −1).
2. Play a joint-target stream from `analysis/capture_to_stream.py` at 50 Hz
   with no policy in the loop. The robot should walk (this is how the
   noise profile is measured); if it does not, the fault is in the target
   path, not in the policy.
3. Run the policy lifted (legs free, upright) before the first floor run.
   The feet should trot at 2 Hz. Belly-up is not lifted: gravity reads
   (0, 0, +1) there and the policy is off its training distribution.
