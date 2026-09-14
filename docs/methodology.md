# Methodology — making the simulator honest about one small robot

The Mini Pupper V2 is a 450 g quadruped on hobby servos with a 10-bit
position readout, a ~100 ms sensing-to-motion round trip and a leg chain
that flexes by millimetres under its own weight. A default Isaac Lab plant
(rigid links, ideal PD servos at datasheet torque, clean observations) is a
different machine: policies that dominate in it stall or thrash on the
floor, and the simulator ranks candidates in the wrong order. Everything in
`plant/` exists to fix that, and every number there was measured on the
robot rather than tuned to make training work. This page says how.

## 1. Plant identification

Robot lifted, legs free, wall power, one excitation axis at a time on all
four legs; commands and servo readings logged at 50 Hz.

| quantity | how | result | in the config |
|---|---|---|---|
| transport delay | chirp 0.2–6 Hz, joint by joint; gain flat to ~5 Hz, constant equivalent lag ⇒ pure delay | ~70 ms (knee 78–84, hip 59–71, abd 66–74) | actuator delay 25–45 physics steps (50–90 ms), randomised per episode |
| velocity limit | constant-speed triangles, amplitude scaled so each traverse has ≥ 8 samples; sustained = p75, transient = p95 | 11–12 rad/s sustained, 25–31 transient | 16 rad/s |
| static deadband | same target approached from both directions, 1 s dwell | 0.13 ticks ≈ 0 | none |
| mass | scale | 448 g (URDF summed to 497 g: the base mass was duplicated) | asset base link corrected |
| geometry | tape vs URDF forward kinematics | 1–3% | unchanged |
| toe chain stiffness | robot powered and held, string pull at the toe with 150 / 300 g | 736 N/m ⇒ ~7.6 N m/rad at the joint, 3–4 mm deflection at 300 g | see below |
| lost motion | free toe travel before the servo resists | 3–4 mm ⇒ 0.05 rad | backlash 0.05 rad |
| hip torque under load | speed ladders with a weighed bottle hung on one toe | 0.25–0.42 N m at the 0.35 rating; the left front hip lagged more than the right at 300 g | the hip-stiffness randomisation range |

Two lessons that generalise:

- **The datasheet torque was not the limit, the modelled inertia was.** The
  first plants used the servo's rotor inertia (armature) and datasheet
  torque; in that plant the scripted reference gait cannot lift a front leg.
  Randomising the armature collapsed every training seed; removing it
  recovered three of four. The effort limit was set to 0.70 N m, a working
  value above the lift threshold, not a measurement.
- **The bench stiffness does not close the loop.** Putting the measured 7.6
  N m/rad in as joint compliance still left the simulated reference gait
  1.8x too fast. A joint-space stiffness of 2.0 / damping 0.21 is what
  makes the scripted controller cover the same distance in simulation as on
  the floor (0.97x at 0.10 m/s, 1.03x at 0.15 m/s, the second out of
  sample). It is a fitted cell, and the remaining error sits in loaded
  multi-leg contact that component tests do not reach.

The acceptance test for the plant was never a policy: it was the stock
scripted controller, which has no learning in it, walking the same distance
in both worlds.

## 2. The observation-noise profile

The policy sees the IMU and the servo positions, and those signals on the
robot are far noisier than their simulated counterparts — the 10-bit servo
readout makes a finite-difference joint velocity jump in steps of
0.29 rad/s, the IMU rides on a rattling shell. A policy trained on clean
observations reacts to that noise as if it were information. The first
phase-1 policy trained without a noise model lost 40% of its distance and
21–27 points of gait rhythm on the robot relative to the same joint stream
played back blind.

How much noise to add is a measurement, and the measurement has to be made
along the same gait in both worlds:

1. In simulation, run the candidate policy closed-loop for 400 steps at
   0.10 m/s and record its observations and actions
   (`scripts/capture_policy_vectors.py`).
2. Turn the actions into the joint-target stream the robot would execute
   (`analysis/capture_to_stream.py`; joint targets in radians, 50 Hz).
3. On the robot, stand still for a few seconds (gyro bias), then play the
   stream back **open-loop** — no policy in the loop — and log servo
   positions and raw IMU at 50 Hz.
4. Rebuild the 45 observation channels from the log with the deployment
   formulas (`analysis/obs_gap.py`, `deploy/contract.py`), then per channel:

   ```
   excess_sd = sqrt(max(0, var_real − var_sim))      over the replay rows
   amplitude = sqrt(3) · mean(excess_sd over the channel's dims)
   ```

   Variances rather than paired differences: the two streams are the same
   gait but not phase-locked to the sample, so a row-paired residual would
   count the transport delay as noise.

The replay is the important part. Noise estimated from the robot's own
closed-loop run is dominated by the thrashing the noise model is meant to
prevent, and raw hardware-minus-simulation differences double-count real
dynamics. Only an open-loop replay of one stream in both worlds isolates
what the sensors add.

Results on this unit (`plant/obs_noise_profile.yaml`): angular velocity
1.0 rad/s, gravity direction 0.30, joint velocity 4.8 rad/s
(gait-independent), joint position 0.05–0.17 rad (gait-dependent; measured
along the phase-2 gait it is 0.174, and the trained phase-2 policy's own
gait re-measures at 0.16, inside it). The validation that the model is
right: the phase-1 policy trained *without* it, evaluated in simulation
*with* these amplitudes, loses ~23 rhythm points — the number it lost on the
robot.

The remaining round-trip latency the plant does not carry (~10–40 ms on the
sensing side) is a per-episode random observation delay of 0–2 control
steps. The unit's weaker left hip becomes a per-episode hip-stiffness scale
of 0.8–1.0, so the policy learns to sense and correct an asymmetry rather
than memorise one.

## 3. What the simulator can and cannot rank

It can filter: a policy that stands still, bounces, or drags its feet in
simulation does so on the floor too, and the gate table
(`analysis/gate_table.py`) is built on that. It could not, on this project:

- **predict heading.** The phase-2 policy that arcs 40° on wood ranks as
  straight as any other in simulation; the plant is symmetric and the
  yaw-rate noise between repeated evaluations of one checkpoint is larger
  than the effect being measured.
- **see stance quality.** A policy passed every simulation gate and slid
  its hind feet through stance on the floor. Video of the feet decided it.
- **reproduce a high-friction floor.** On a padded mat the weak side catches
  and rotates the body; the simulator, with the same asymmetry injected,
  produces no such event at any friction coefficient.

So candidates are selected in simulation and ranked on the floor. A useful
intermediate is the open-loop bar (`scripts/openloop_replay.sh`): a
policy's own action stream, played back blind on the randomised env, must
carry the body ≥ 660 mm in 400 steps. Policies that clear it also walk when
their observations are worse than expected; policies that only make
distance by reacting to their observations do not.

## 4. Floor protocol

Glossy wood, tape on the floor, phone camera on a tripod. Fresh battery,
voltage logged. Command 0.10 m/s for 10 s (500 control steps), start in the
stand pose. Measure forward travel of the body, lateral offset at the nose
and tail, and heading change; three runs per policy; the scripted controller
run the same day as the denominator. Every run filmed — foot placement is
judged from the video. Write the expected numbers down before the runs.

## 5. One robot

Every measurement above, and every floor result in this repo, is from one
Mini Pupper V2 unit. Its left-side servo weakness is in the noise profile,
in the hip randomisation range and in the phase-2 arc. Another unit will
share the plant class and the noise mechanisms but not the numbers.
Re-measure sections 1 and 2 on yours before reading a floor difference as a
policy difference.
