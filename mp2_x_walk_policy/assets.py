"""Mini Pupper V2 articulation for Isaac Lab: asset path, joint order, stand pose.

The USD in ``assets/`` is the mass-corrected conversion of the upstream
mini_pupper_ros description (see ``assets/build/README.md``). Every checkpoint
in this repository was trained on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _asset_dir() -> Path:
    override = os.environ.get("MP2_X_WALK_ASSET_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "assets"


MINI_PUPPER_USD = _asset_dir() / "mini_pupper_2_isaac_v27" / "mini_pupper_2_isaac_v27.usda"

# Joint order used everywhere a 12-vector appears (actions, joint_pos, joint_vel):
# leg by leg, front-left / front-right / back-left / back-right; within a leg
# abduction, hip, knee. This is the order the robot's ROS driver uses.
MINI_PUPPER_ROS_JOINT_ORDER = [
    "base_lf1",
    "lf1_lf2",
    "lf2_lf3",
    "base_rf1",
    "rf1_rf2",
    "rf2_rf3",
    "base_lb1",
    "lb1_lb2",
    "lb2_lb3",
    "base_rb1",
    "rb1_rb2",
    "rb2_rb3",
]

# Isaac's articulation order after importing the USD. Terms are declared with
# preserve_order=True against the ROS order, so this map is informational.
MINI_PUPPER_ISAAC_JOINT_ORDER = [
    "base_lf1",
    "base_rf1",
    "base_rb1",
    "base_lb1",
    "lf1_lf2",
    "rf1_rf2",
    "rb1_rb2",
    "lb1_lb2",
    "lf2_lf3",
    "rf2_rf3",
    "rb2_rb3",
    "lb2_lb3",
]

ROS_TO_ISAAC_INDEX = [MINI_PUPPER_ISAAC_JOINT_ORDER.index(name) for name in MINI_PUPPER_ROS_JOINT_ORDER]
ISAAC_TO_ROS_INDEX = [MINI_PUPPER_ROS_JOINT_ORDER.index(name) for name in MINI_PUPPER_ISAAC_JOINT_ORDER]

MINI_PUPPER_FOOT_BODY_NAMES = ["lffoot", "rffoot", "lbfoot", "rbfoot"]
MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES = ["base_link", "base_inertia", "Body_front", "Board_PCB", "Shell_Side"]
MINI_PUPPER_CONTACT_BODY_REL_PATHS = {
    "base_link": "Geometry/base_link",
    "base_inertia": "Geometry/base_link/base_inertia",
    "Body_front": "Geometry/base_link/Body_front",
    "Board_PCB": "Geometry/base_link/Board_PCB",
    "Shell_Side": "Geometry/base_link/Shell_Side",
    "lffoot": "Geometry/base_link/lf1/lf2/lf3/lffoot",
    "rffoot": "Geometry/base_link/rf1/rf2/rf3/rffoot",
    "lbfoot": "Geometry/base_link/lb1/lb2/lb3/lbfoot",
    "rbfoot": "Geometry/base_link/rb1/rb2/rb3/rbfoot",
}
MINI_PUPPER_CONTACT_SENSOR_NAMES = {
    body_name: f"contact_{body_name}" for body_name in MINI_PUPPER_CONTACT_BODY_REL_PATHS
}

# The stand pose. Also the offset every action is added to (use_default_offset)
# and the reference for joint_pos_rel in the observation.
MINI_PUPPER_STAND_JOINT_POS = {
    "base_lf1": 0.0,
    "lf1_lf2": 0.994,
    "lf2_lf3": -1.767,
    "base_rf1": 0.0,
    "rf1_rf2": 0.994,
    "rf2_rf3": -1.767,
    "base_lb1": 0.0,
    "lb1_lb2": 0.994,
    "lb2_lb3": -1.767,
    "base_rb1": 0.0,
    "rb1_rb2": 0.994,
    "rb2_rb3": -1.767,
}


def _spawn_cfg(usd_path: Path) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )


def _initial_state(root_height: float) -> ArticulationCfg.InitialStateCfg:
    return ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, root_height),
        joint_pos=MINI_PUPPER_STAND_JOINT_POS,
        joint_vel={".*": 0.0},
    )


# Datasheet-style servo model. Both tasks replace this actuator with the
# measured plant cell (plant.py); friction and armature are carried over.
MINI_PUPPER_V2_CFG = ArticulationCfg(
    spawn=_spawn_cfg(MINI_PUPPER_USD),
    init_state=_initial_state(0.085),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=MINI_PUPPER_ROS_JOINT_ORDER,
            effort_limit=0.35,
            saturation_effort=0.35,
            velocity_limit=10.47,
            velocity_limit_sim=10.47,
            stiffness=70.0,
            damping=1.2,
            friction=0.03,
            armature=0.005,
        ),
    },
)
