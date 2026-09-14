"""The measured plant: actuator cell, observation-noise profile, physics preset.

Every number here was measured on one Mini Pupper V2 unit (see
docs/methodology.md and plant/*.yaml for how). Both task configs import
from this module rather than repeating the values.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .actuators import BacklashDCMotorCfg


@configclass
class MiniPupperPhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = default


# --------------------------------------------------------------- actuator cell
#
# stiffness 2.0 / damping 0.21   joint-space series compliance. The servo horn
#                                tracks its command; what yields is the
#                                horn-to-toe chain. Tuned so a scripted
#                                reference gait covers the same distance in
#                                sim as on the floor (0.97x at 0.10 m/s).
# backlash 0.05 rad              3-4 mm of free toe travel before the servo
#                                resists, measured by hand at the toe.
# delay 25-45 physics steps      50-90 ms command-to-motion transport delay,
#                                from a chirp test (~70 ms, pure delay to
#                                5 Hz); randomised per env per episode.
# effort 0.70 N m                the datasheet 0.35 N m sits just below the
#                                torque needed to lift a front leg in sim;
#                                0.70 is a working value above that cliff.
# velocity_limit 16.0 rad/s      slew ladder: 11-12 rad/s sustained,
#                                25-31 rad/s transient.
PLANT_EFFORT_LIMIT_NM = 0.70
PLANT_VELOCITY_LIMIT_RAD_S = 16.0
PLANT_STIFFNESS = 2.0
PLANT_DAMPING = 0.21
PLANT_DELAY_STEPS = (25, 45)  # PHYSICS steps at sim.dt = 0.002
PLANT_BACKLASH_RAD = 0.05


def measured_plant_actuator(joint_names_expr, friction: float, armature: float) -> BacklashDCMotorCfg:
    """The actuator both tasks train against."""
    return BacklashDCMotorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=PLANT_EFFORT_LIMIT_NM,
        saturation_effort=PLANT_EFFORT_LIMIT_NM,
        velocity_limit=PLANT_VELOCITY_LIMIT_RAD_S,
        velocity_limit_sim=PLANT_VELOCITY_LIMIT_RAD_S,
        stiffness=PLANT_STIFFNESS,
        damping=PLANT_DAMPING,
        friction=friction,
        armature=armature,
        min_delay=PLANT_DELAY_STEPS[0],
        max_delay=PLANT_DELAY_STEPS[1],
        backlash_rad=PLANT_BACKLASH_RAD,
    )


# ------------------------------------------------------- observation noise
#
# Uniform additive noise on the four measured channels. Amplitudes are
# sqrt(3) times the per-channel excess standard deviation of the REAL policy
# input over the simulated one, measured by replaying the same joint-target
# stream open-loop on the robot and in sim (docs/methodology.md). They are
# not tuning knobs.
#
#   base_ang_vel      1.0 rad/s   IMU gyro through the gait (excess ~0.6)
#   projected_gravity 0.30        IMU accelerometer normalised (excess ~0.175)
#   joint_vel         4.8 rad/s   finite-difference of 10-bit servo readings
#                                 at 50 Hz (1 LSB = 0.286 rad/s); excess ~2.75
#   joint_pos         0.05        CPG task: a third of the measured excess,
#                                 kept partial so proprioception stays usable
#                     0.174       end-to-end task: the full excess measured
#                                 along that task's own gait
NOISE_ANG_VEL = 1.0
NOISE_GRAVITY = 0.30
NOISE_JOINT_VEL = 4.8
NOISE_JOINT_POS_CPG = 0.05
NOISE_JOINT_POS_E2E = 0.174

# The clean floor the end-to-end task starts its ramp from (also the
# amplitudes the observation terms are declared with).
NOISE_FLOOR_ANG_VEL = 0.05
NOISE_FLOOR_GRAVITY = 0.02
NOISE_FLOOR_JOINT_POS = 0.01
NOISE_FLOOR_JOINT_VEL = 0.2

# Observation-side share of the transport delay: 0-2 policy steps (0-40 ms),
# resampled per env per episode.
OBS_DELAY_STEPS = (0, 2)

# Per-hip stiffness scale at reset. The unit delivers less swing on one side;
# 0.80-1.0 spans roughly 0-2x the measured in-gait deficit.
HIP_STIFFNESS_RANGE = (0.80, 1.0)
HIP_JOINT_NAMES = ["lf1_lf2", "rf1_rf2", "lb1_lb2", "rb1_rb2"]

NOISY_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"]


# ---------------------------------------------------- evaluation randomisation


def apply_play_randomisation(cfg) -> None:
    """Non-zero reset ranges for the PlayRand evaluation ids.

    Play ids inherit reset ranges of exactly zero, so their environments would
    be identical copies and every metric n = 1. Small pose/velocity offsets and
    joint offsets give a spread. Joints use ``reset_joints_by_offset`` rather
    than scale because the abduction joints default to 0.0 and no scale
    factor can perturb them.

    Evaluation only. Yaw is randomised, so use spawn-frame displacement and
    yaw-drift metrics (differences from the initial pose). Randomising yaw in
    training would change the task: yaw_l2 and lateral_position_l2 are
    defined in the world frame.
    """
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "yaw": (-0.15, 0.15),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (0.0, 0.0),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
    }
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-0.10, 0.10)


def strip_training_only_terms(cfg) -> None:
    """Make a training config a clean-observation evaluation config: no
    corruption, no observation delay, no hip randomisation, no noise ramp."""
    cfg.observations.policy.enable_corruption = False
    for term_name in NOISY_TERMS:
        term = getattr(cfg.observations.policy, term_name)
        term.modifiers = [
            m for m in (term.modifiers or [])
            if type(m).__name__ != "RandomizedObservationDelayCfg"
        ] or None
    cfg.events.randomize_hip_delivery = None
    if hasattr(cfg.curriculum, "obs_noise_ramp"):
        cfg.curriculum.obs_noise_ramp = None
