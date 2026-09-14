"""Phase 2: end-to-end policy, trained from and anchored to the distilled clone.

Observation (45): base_ang_vel(3), projected_gravity(3), velocity_command(3),
joint_pos - stand(12), joint_vel(12), previous action(12). No gait clock.
Action (12): joint target in [-1, 1], applied as ``stand + 0.25 * action``.

The environment declares a second, clean copy of the observation
(``anchor``) that the anchored PPO reads to query the frozen clone; the
actor and critic see only ``policy``. Observation noise starts at the clean
floor and ramps to the measured amplitudes (hold 150 iterations, ramp 300),
so the warm-started actor is not hit with full noise on its first update.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .. import mdp as mp2_mdp
from ..assets import MINI_PUPPER_FOOT_BODY_NAMES, MINI_PUPPER_ROS_JOINT_ORDER
from ..modifiers import RandomizedObservationDelayCfg
from ..plant import (
    HIP_JOINT_NAMES,
    HIP_STIFFNESS_RANGE,
    NOISE_ANG_VEL,
    NOISE_FLOOR_ANG_VEL,
    NOISE_FLOOR_GRAVITY,
    NOISE_FLOOR_JOINT_POS,
    NOISE_FLOOR_JOINT_VEL,
    NOISE_GRAVITY,
    NOISE_JOINT_POS_E2E,
    NOISE_JOINT_VEL,
    NOISY_TERMS,
    OBS_DELAY_STEPS,
    apply_play_randomisation,
    strip_training_only_terms,
)
from .base import FOOT_SENSOR_NAMES, MiniPupperBaseEnvCfg, make_play

NOISE_START_AMPS = [NOISE_FLOOR_ANG_VEL, NOISE_FLOOR_GRAVITY, NOISE_FLOOR_JOINT_POS, NOISE_FLOOR_JOINT_VEL]
NOISE_END_AMPS = [NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_JOINT_POS_E2E, NOISE_JOINT_VEL]
NOISE_RAMP_HOLD_ITERS = 150
NOISE_RAMP_ITERS = 300
STEPS_PER_ITER = 24  # rsl_rl num_steps_per_env


def _anchor_group() -> ObsGroup:
    """A fresh, clean mirror of the 45-dim policy observation, in the same order."""
    joint_cfg = SceneEntityCfg("robot", joint_names=MINI_PUPPER_ROS_JOINT_ORDER, preserve_order=True)
    group = ObsGroup()
    group.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    group.projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
    group.velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    group.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": joint_cfg})
    group.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": joint_cfg})
    group.actions = ObsTerm(func=mdp.last_action)
    group.enable_corruption = False
    group.concatenate_terms = True
    return group


@configclass
class E2EAnchoredEnvCfg(MiniPupperBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- command range and straightness
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.25)
        self.rewards.track_ang_vel_z_exp.weight = 0.50
        self.rewards.yaw_l2 = RewTerm(func=mp2_mdp.yaw_l2, weight=-0.40)
        self.rewards.lateral_position_l2 = RewTerm(func=mp2_mdp.lateral_position_l2, weight=-1.50)
        self.rewards.lateral_velocity_l2 = RewTerm(func=mp2_mdp.lateral_velocity_l2, weight=-0.40)

        # --- gait shaping: pay a lifted foot for moving forward relative to the
        # body (only if it recently touched down), penalise stance slip hard
        self.rewards.swing_progress = RewTerm(
            func=mp2_mdp.SwingProgressTouchdown,
            weight=0.80,
            params={
                "foot_body_names": list(MINI_PUPPER_FOOT_BODY_NAMES),
                "command_name": "base_velocity",
                "lift_threshold": 0.008,
                "target_lift": 0.020,
                "reference_speed": 0.15,
                "max_rewarded_feet": 2.0,
                "command_deadband": 0.02,
                "touchdown_window_s": 0.40,
                "planted_tolerance": 0.005,
            },
        )
        self.rewards.stance_foot_slip_l2 = RewTerm(
            func=mp2_mdp.stance_foot_slip_l2,
            weight=-2.0,
            params={
                "sensor_names": FOOT_SENSOR_NAMES,
                "foot_body_names": MINI_PUPPER_FOOT_BODY_NAMES,
                "threshold": 2.0,
            },
        )

        # --- plant randomisation at reset
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.5, 1.5),
                "damping_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.randomize_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.5, 2.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # --- observation realism: delay and hip asymmetry as in phase 1; noise
        # is declared at the clean floor and ramped by the curriculum term below
        for term_name, amp in zip(NOISY_TERMS, NOISE_START_AMPS):
            getattr(self.observations.policy, term_name).noise = Unoise(n_min=-amp, n_max=amp)
        for term_name in NOISY_TERMS:
            term = getattr(self.observations.policy, term_name)
            existing = list(term.modifiers) if term.modifiers else []
            term.modifiers = existing + [
                RandomizedObservationDelayCfg(
                    params={"min_steps": OBS_DELAY_STEPS[0], "max_steps": OBS_DELAY_STEPS[1]}
                )
            ]
        self.events.randomize_hip_delivery = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_NAMES),
                "stiffness_distribution_params": HIP_STIFFNESS_RANGE,
                "damping_distribution_params": None,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.curriculum.obs_noise_ramp = CurrTerm(
            func=mp2_mdp.ramp_observation_noise,
            params={
                "term_names": NOISY_TERMS,
                "start_amps": NOISE_START_AMPS,
                "end_amps": NOISE_END_AMPS,
                "hold_iters": NOISE_RAMP_HOLD_ITERS,
                "ramp_iters": NOISE_RAMP_ITERS,
                "steps_per_iter": STEPS_PER_ITER,
            },
        )

        # --- the clean observation the anchor loss is computed on
        self.observations.anchor = _anchor_group()


@configclass
class E2EAnchoredPlayEnvCfg(E2EAnchoredEnvCfg):
    """Clean-observation evaluation: 16 identical envs at 0.10 m/s."""

    def __post_init__(self):
        super().__post_init__()
        make_play(self)
        strip_training_only_terms(self)


@configclass
class E2EAnchoredPlayRandEnvCfg(E2EAnchoredPlayEnvCfg):
    """Clean-observation evaluation over 32 randomised starts."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        apply_play_randomisation(self)


@configclass
class E2EAnchoredNoisyEnvCfg(E2EAnchoredEnvCfg):
    """Evaluation under the full measured observation noise (static, no ramp),
    with the observation delay and hip randomisation on. This is what the
    noise-robustness bar in analysis/gate_table.py is scored on."""

    def __post_init__(self):
        super().__post_init__()
        for term_name, amp in zip(NOISY_TERMS, NOISE_END_AMPS):
            getattr(self.observations.policy, term_name).noise = Unoise(n_min=-amp, n_max=amp)
        self.curriculum.obs_noise_ramp = None
