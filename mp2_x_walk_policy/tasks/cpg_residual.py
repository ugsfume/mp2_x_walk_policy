"""Phase 1: residual policy around a fixed diagonal-trot prior.

Observation (47): base_ang_vel(3), projected_gravity(3), velocity_command(3),
joint_pos - stand(12), joint_vel(12), previous action(12), gait clock
[sin, cos](2). Action (12): residual in [-1, 1], applied as
``stand + CPG(phase, vx) + 0.12 * residual``.

The prior fixes the gait family (2 Hz trot, 0.22 s stances); the residual
learns balance and propulsion on top. Rewards are the stock tracking terms
plus straightness and a set of small shaping terms that keep the rear legs
participating.

Contract used by ``distill/bc_collect.py``: the action term is named
``joint_pos`` and the gait clock is the LAST observation term.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .. import mdp as mp2_mdp
from ..actions_cfg import CPGResidualJointPositionActionCfg
from ..assets import MINI_PUPPER_FOOT_BODY_NAMES, MINI_PUPPER_ROS_JOINT_ORDER
from ..modifiers import RandomizedObservationDelayCfg
from ..plant import (
    HIP_JOINT_NAMES,
    HIP_STIFFNESS_RANGE,
    NOISE_ANG_VEL,
    NOISE_GRAVITY,
    NOISE_JOINT_POS_CPG,
    NOISE_JOINT_VEL,
    NOISY_TERMS,
    OBS_DELAY_STEPS,
    apply_play_randomisation,
    strip_training_only_terms,
)
from .base import (
    FOOT_SENSOR_NAMES,
    FRONT_FEET,
    FRONT_SENSOR_NAMES,
    REAR_FEET,
    REAR_SENSOR_NAMES,
    MiniPupperBaseEnvCfg,
    make_play,
)

CPG_FREQUENCY_HZ = 2.0
CPG_HIP_AMPLITUDE_RAD = 0.16
CPG_KNEE_LIFT_RAD = 0.20
RESIDUAL_SCALE = 0.12


@configclass
class CpgResidualEnvCfg(MiniPupperBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- straightness and rear-leg shaping
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.yaw_l2 = RewTerm(func=mp2_mdp.yaw_l2, weight=-0.80)
        self.rewards.lateral_position_l2 = RewTerm(func=mp2_mdp.lateral_position_l2, weight=-3.00)
        self.rewards.lateral_velocity_l2 = RewTerm(func=mp2_mdp.lateral_velocity_l2, weight=-0.80)
        self.rewards.stance_foot_slip_l2 = RewTerm(
            func=mp2_mdp.stance_foot_slip_l2,
            weight=-0.05,
            params={
                "sensor_names": FOOT_SENSOR_NAMES,
                "foot_body_names": MINI_PUPPER_FOOT_BODY_NAMES,
                "threshold": 1.0,
            },
        )
        # Two absolute-height terms were tried and replaced by the relative-lift
        # terms below; the slots stay (as None) so the term order matches the
        # recorded training config.
        self.rewards.rear_swing_clearance = None
        self.rewards.rear_contact_transitions = RewTerm(
            func=mp2_mdp.rear_contact_transitions,
            weight=0.05,
            params={"rear_sensor_names": REAR_SENSOR_NAMES, "command_name": "base_velocity"},
        )
        self.rewards.front_rear_foot_height_imbalance_l2 = None
        self.rewards.front_stance_foot_slip_l2 = RewTerm(
            func=mp2_mdp.stance_foot_slip_l2,
            weight=-0.03,
            params={
                "sensor_names": FRONT_SENSOR_NAMES,
                "foot_body_names": FRONT_FEET,
                "threshold": 1.0,
            },
        )
        self.rewards.front_rear_transition_imbalance_l2 = RewTerm(
            func=mp2_mdp.contact_transition_imbalance_l2,
            weight=-0.02,
            params={
                "front_sensor_names": FRONT_SENSOR_NAMES,
                "rear_sensor_names": REAR_SENSOR_NAMES,
            },
        )
        self.rewards.rear_air_fraction = RewTerm(
            func=mp2_mdp.rear_air_fraction,
            weight=0.01,
            params={
                "rear_sensor_names": REAR_SENSOR_NAMES,
                "command_name": "base_velocity",
                "threshold": 1.0,
            },
        )
        self.rewards.rear_relative_swing_clearance = RewTerm(
            func=mp2_mdp.rear_relative_swing_clearance,
            weight=0.08,
            params={
                "all_foot_body_names": MINI_PUPPER_FOOT_BODY_NAMES,
                "rear_sensor_names": REAR_SENSOR_NAMES,
                "rear_foot_body_names": REAR_FEET,
                "command_name": "base_velocity",
                "threshold": 1.0,
                "target_lift": 0.010,
            },
        )
        self.rewards.front_rear_relative_lift_imbalance_l2 = RewTerm(
            func=mp2_mdp.front_rear_relative_lift_imbalance_l2,
            weight=-0.05,
            params={
                "all_foot_body_names": MINI_PUPPER_FOOT_BODY_NAMES,
                "front_foot_body_names": FRONT_FEET,
                "rear_foot_body_names": REAR_FEET,
                "target_lift": 0.010,
            },
        )

        # --- gait clock in the observation, and rewards that follow it
        self.observations.policy.gait_phase_clock = ObsTerm(
            func=mp2_mdp.gait_phase_clock,
            params={"frequency": CPG_FREQUENCY_HZ},
        )
        self.rewards.diagonal_trot_contact_match = RewTerm(
            func=mp2_mdp.diagonal_trot_contact_match,
            weight=0.04,
            params={
                "foot_sensor_names": FOOT_SENSOR_NAMES,
                "command_name": "base_velocity",
                "frequency": CPG_FREQUENCY_HZ,
                "threshold": 1.0,
            },
        )
        self.rewards.diagonal_trot_swing_clearance = RewTerm(
            func=mp2_mdp.diagonal_trot_swing_clearance,
            weight=0.04,
            params={
                "all_foot_body_names": MINI_PUPPER_FOOT_BODY_NAMES,
                "command_name": "base_velocity",
                "frequency": CPG_FREQUENCY_HZ,
                "target_lift": 0.010,
            },
        )

        # --- the residual action around the CPG prior
        self.actions.joint_pos = CPGResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=MINI_PUPPER_ROS_JOINT_ORDER,
            preserve_order=True,
            scale=RESIDUAL_SCALE,
            use_default_offset=True,
            frequency=CPG_FREQUENCY_HZ,
            hip_amplitude=CPG_HIP_AMPLITUDE_RAD,
            knee_lift_amplitude=CPG_KNEE_LIFT_RAD,
            abduction_amplitude=0.0,
        )

        # --- observation realism: measured noise, observation delay, hip asymmetry
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-NOISE_ANG_VEL, n_max=NOISE_ANG_VEL)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-NOISE_GRAVITY, n_max=NOISE_GRAVITY)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-NOISE_JOINT_VEL, n_max=NOISE_JOINT_VEL)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-NOISE_JOINT_POS_CPG, n_max=NOISE_JOINT_POS_CPG)
        # One modifier cfg per term: the manager binds the instantiated modifier
        # to the cfg object, so sharing one across terms of different widths
        # would feed 3-dim data into a 12-dim buffer.
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


@configclass
class CpgResidualPlayEnvCfg(CpgResidualEnvCfg):
    """Clean-observation evaluation: 16 identical envs at 0.10 m/s."""

    def __post_init__(self):
        super().__post_init__()
        make_play(self)
        strip_training_only_terms(self)


@configclass
class CpgResidualPlayRandEnvCfg(CpgResidualPlayEnvCfg):
    """Clean-observation evaluation over 32 randomised starts."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        apply_play_randomisation(self)
