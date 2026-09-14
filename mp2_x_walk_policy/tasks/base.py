"""What the two tasks share: flat ground, the measured plant, contact sensors,
the forward-only command, the hardware-reproducible observation set, and the
stock tracking / regularisation rewards.

Attribute assignment ORDER in ``__post_init__`` is deliberate. Isaac Lab's
managers iterate configclass attributes in insertion order, and that order is
where rewards are summed and where random draws happen; both task files add
their terms after ``super().__post_init__()`` so the resolved configs match
the recorded training configs term for term (``tools/check_record.py``).
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .. import mdp as mp2_mdp
from ..assets import (
    MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES,
    MINI_PUPPER_CONTACT_BODY_REL_PATHS,
    MINI_PUPPER_CONTACT_SENSOR_NAMES,
    MINI_PUPPER_ROS_JOINT_ORDER,
    MINI_PUPPER_V2_CFG,
)
from ..plant import (
    NOISE_FLOOR_ANG_VEL,
    NOISE_FLOOR_GRAVITY,
    NOISE_FLOOR_JOINT_POS,
    NOISE_FLOOR_JOINT_VEL,
    MiniPupperPhysicsCfg,
    measured_plant_actuator,
)

FOOT_SENSOR_NAMES = [MINI_PUPPER_CONTACT_SENSOR_NAMES[n] for n in ("lffoot", "rffoot", "lbfoot", "rbfoot")]
FRONT_FEET = ["lffoot", "rffoot"]
REAR_FEET = ["lbfoot", "rbfoot"]
FRONT_SENSOR_NAMES = [MINI_PUPPER_CONTACT_SENSOR_NAMES[n] for n in FRONT_FEET]
REAR_SENSOR_NAMES = [MINI_PUPPER_CONTACT_SENSOR_NAMES[n] for n in REAR_FEET]


@configclass
class MiniPupperBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=MiniPupperPhysicsCfg())

    def __post_init__(self):
        super().__post_init__()

        # --- scene: flat plane, the mass-corrected asset on the measured plant
        self.scene.num_envs = 512
        self.scene.env_spacing = 1.0
        self.scene.robot = MINI_PUPPER_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        stock = self.scene.robot.actuators["legs"]
        self.scene.robot.actuators["legs"] = measured_plant_actuator(
            stock.joint_names_expr, friction=stock.friction, armature=stock.armature
        )
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.scene.contact_forces = None

        # --- timing: 50 Hz policy over 500 Hz physics, 10 s episodes
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # --- command: forward only. The policy is trained on vx alone.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.15)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # --- action: joint targets in ROS order around the stand pose
        joint_cfg = SceneEntityCfg("robot", joint_names=MINI_PUPPER_ROS_JOINT_ORDER, preserve_order=True)
        self.actions.joint_pos.joint_names = MINI_PUPPER_ROS_JOINT_ORDER
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True

        # --- observation: only what the robot can measure. Noise here is the
        # clean floor; each task sets its own amplitudes.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-NOISE_FLOOR_ANG_VEL, n_max=NOISE_FLOOR_ANG_VEL),
        )
        self.observations.policy.projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-NOISE_FLOOR_GRAVITY, n_max=NOISE_FLOOR_GRAVITY),
        )
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joint_cfg},
            noise=Unoise(n_min=-NOISE_FLOOR_JOINT_POS, n_max=NOISE_FLOOR_JOINT_POS),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joint_cfg},
            noise=Unoise(n_min=-NOISE_FLOOR_JOINT_VEL, n_max=NOISE_FLOOR_JOINT_VEL),
        )
        self.observations.policy.enable_corruption = True
        self.observations.policy.concatenate_terms = True

        # --- events: deterministic resets; each task adds its randomisation
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # --- rewards: stock tracking and regularisation
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.10
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -0.05

        self.terminations.base_contact = None
        self.curriculum.terrain_levels = None

        # --- contact sensors on the chassis (termination) and the feet (rewards)
        for body_name, rel_path in MINI_PUPPER_CONTACT_BODY_REL_PATHS.items():
            setattr(
                self.scene,
                MINI_PUPPER_CONTACT_SENSOR_NAMES[body_name],
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{rel_path}",
                    update_period=self.sim.dt,
                    history_length=self.decimation,
                    track_air_time=True,
                ),
            )
        self.terminations.base_contact = DoneTerm(
            func=mp2_mdp.any_contact,
            params={
                "sensor_names": [
                    MINI_PUPPER_CONTACT_SENSOR_NAMES[body_name]
                    for body_name in MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES
                ],
                "threshold": 1.0,
            },
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.7},
        )


def make_play(cfg) -> None:
    """Evaluation settings shared by every Play/PlayRand id: 16 envs, a fixed
    0.10 m/s command, clean observations."""
    cfg.scene.num_envs = 16
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.10)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
