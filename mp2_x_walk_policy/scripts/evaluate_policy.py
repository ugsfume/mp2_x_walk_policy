"""Evaluate a checkpoint or a scripted agent on one of this repo's tasks.

Writes <stem>_eval.csv (per-step rows for env 0), <stem>_gait_traces.npz and
<stem>_summary.json (gait and contact metrics), an optional <stem>_gait.png and an
optional video. Set MP2_X_WALK_RECORD_FORCES=1 to also write <stem>_foot_forces.npz
(per-foot world-frame force vectors, foot velocities and substep normal forces).

Example:
    python -m mp2_x_walk_policy.scripts.evaluate_policy --task MP2-Walk-E2E-PlayRand-v0 --checkpoint phase2/checkpoints/e2e_anchored_s42_1050.pt --num_envs 32 --record_envs 8 --steps 490 --command_x 0.10 --contact_threshold 2.0 --output_dir runs/eval --viz none
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

from mp2_x_walk_policy import register_tasks
from mp2_x_walk_policy.assets import (
    MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES,
    MINI_PUPPER_CONTACT_SENSOR_NAMES,
    MINI_PUPPER_FOOT_BODY_NAMES,
    MINI_PUPPER_ROS_JOINT_ORDER,
)
from mp2_x_walk_policy.gait_metrics import GaitRecorder, aggregate_summaries, write_traces_multi

register_tasks()

parser = argparse.ArgumentParser(description="Evaluate Mini Pupper RSL-RL checkpoints.")
parser.add_argument("--task", default="MP2-Walk-E2E-PlayRand-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument(
    "--scripted_agent",
    choices=("policy", "zero", "random", "replay"),
    default="policy",
    help=(
        "Use an RSL-RL policy checkpoint, all-zero actions, uniform random actions in [-1, 1], "
        "or an open-loop REPLAY of a recorded action sequence (--replay_file). 'replay' feeds "
        "the rows of the NPZ (q_measured, shape (T, 12)) directly as the policy's raw actions "
        "into env.step, holding the last row past the end. The shipped tasks' action term "
        "applies its own scale and default offset, so the rows must be RAW POLICY ACTIONS in "
        "[-1, 1] (e.g. captured with scripts/capture_policy_vectors.py), not joint angles."
    ),
)
parser.add_argument(
    "--replay_file",
    default=None,
    help=(
        "NPZ with q_measured (T, 12) raw policy actions in [-1, 1] and scalar rate_hz, "
        "e.g. from scripts/capture_policy_vectors.py. rate_hz must equal the task control "
        "rate (50 Hz on the shipped tasks); this script refuses to resample."
    ),
)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--command_x", type=float, default=0.10)
parser.add_argument(
    "--command_yaw", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="Sample ang_vel_z commands from U(MIN, MAX) instead of pinning 0; the per-env "
         "values are written to the summary as command_z_per_env.")
parser.add_argument("--video", action="store_true", help="Record an RGB video through Gym RecordVideo.")
parser.add_argument("--video_length", type=int, default=None)
parser.add_argument("--show_command_markers", action="store_true", help="Show velocity command debug arrows.")
parser.add_argument("--camera_distance", type=float, default=0.45)
parser.add_argument("--camera_height", type=float, default=0.25)
parser.add_argument("--camera_target_x", type=float, default=0.10)
parser.add_argument("--camera_target_z", type=float, default=0.06)
parser.add_argument(
    "--camera_view",
    choices=("custom", "side", "rear_quarter", "top_oblique", "gait", "gait_wide"),
    default="custom",
    help=(
        "Preset diagnostic camera view. Every preset is STATIC in recorded video: neither "
        "sim.set_camera_view() nor ViewerCfg.origin_type affects the rgb_array render product, "
        "so --follow_robot_camera only moves an interactive viewport. 'gait' is a close "
        "foot-level side view framing ~1 m of travel along y=0; 'gait_wide' pulls back and "
        "offsets the aim to keep a turning policy in shot. The older presets sit ~2.2 m away, "
        "which is too far to judge swing clearance."
    ),
)
parser.add_argument("--follow_robot_camera", action="store_true")
parser.add_argument("--fall_height", type=float, default=0.055)
parser.add_argument("--fall_tilt_deg", type=float, default=90.0)
parser.add_argument("--unhealthy_final_tilt_deg", type=float, default=45.0)
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument(
    "--record_envs",
    type=int,
    default=1,
    help=(
        "Number of environments to record gait metrics for. Use with a PlayRand task, "
        "whose reset ranges are non-zero, to get an actual distribution -- on a Play "
        "task all envs are bit-identical and extra envs add nothing."
    ),
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import launch_simulation, resolve_task_config
import importlib.metadata as metadata


def _tensor_row(tensor: torch.Tensor, env_id: int = 0) -> list[float]:
    return [float(v) for v in tensor[env_id].detach().cpu().flatten()]


def _quat_xyzw_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab xyzw quaternions to roll/pitch/yaw radians.

    Isaac Lab stores ``root_quat_w`` as (x, y, z, w).
    """

    x, y, z, w = quat.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


def _tilt_from_projected_gravity(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Return body tilt from upright in radians using projected gravity."""

    return torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0))


def _per_body_contact_tensors(env, threshold: float) -> dict[str, object] | None:
    sensor_names = MINI_PUPPER_CONTACT_SENSOR_NAMES
    required = MINI_PUPPER_FOOT_BODY_NAMES + MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES
    if not all(sensor_names[name] in env.unwrapped.scene.sensors for name in required):
        return None

    foot_forces = []
    chassis_forces = []
    for name in MINI_PUPPER_FOOT_BODY_NAMES:
        sensor = env.unwrapped.scene.sensors[sensor_names[name]]
        force_norm = torch.linalg.norm(sensor.data.net_forces_w_history.torch.detach(), dim=-1).amax(dim=(1, 2))
        foot_forces.append(force_norm)
    for name in MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES:
        sensor = env.unwrapped.scene.sensors[sensor_names[name]]
        force_norm = torch.linalg.norm(sensor.data.net_forces_w_history.torch.detach(), dim=-1).amax(dim=(1, 2))
        chassis_forces.append(force_norm)

    foot_force_tensor = torch.stack(foot_forces, dim=1)
    chassis_force_tensor = torch.stack(chassis_forces, dim=1)
    # Instantaneous per-foot net force VECTORS (world frame). The norm-and-max
    # reduction above discards direction; the tangential component is the
    # propulsion question, so keep the vectors available for recording.
    foot_force_vecs = torch.stack(
        [
            env.unwrapped.scene.sensors[sensor_names[name]].data.net_forces_w.torch.detach()[:, 0, :]
            for name in MINI_PUPPER_FOOT_BODY_NAMES
        ],
        dim=1,
    )
    # Substep-resolution normal force: the sensor history buffer holds every
    # physics substep of the last control step (history_length == decimation,
    # update_period == sim.dt), so this is the full-rate contact record. The
    # instantaneous channels above sample only every 10th substep, which
    # aliases contact chatter.
    foot_normal_hist = torch.stack(
        [
            env.unwrapped.scene.sensors[sensor_names[name]]
            .data.net_forces_w_history.torch.detach()[:, :, 0, 2]
            for name in MINI_PUPPER_FOOT_BODY_NAMES
        ],
        dim=2,
    )  # (E, H, F)
    return {
        "foot_forces": foot_force_tensor,
        "chassis_forces": chassis_force_tensor,
        "foot_force_vecs": foot_force_vecs,
        "foot_normal_hist": foot_normal_hist,
        "foot_contacts": foot_force_tensor > threshold,
        "chassis_contacts": chassis_force_tensor > threshold,
    }


def _per_body_contact_state(env, threshold: float) -> dict[str, object]:
    tensors = _per_body_contact_tensors(env, threshold)
    if tensors is None:
        return {"present": False}

    foot_force_tensor = tensors["foot_forces"]
    chassis_force_tensor = tensors["chassis_forces"]
    foot_contact_tensor = tensors["foot_contacts"]
    chassis_contact_tensor = tensors["chassis_contacts"]
    return {
        "present": True,
        "foot_contact_fraction_mean": float(foot_contact_tensor.float().mean()),
        "foot_contact_count_mean": float(foot_contact_tensor.sum(dim=1).float().mean()),
        "chassis_contact_any": bool(chassis_contact_tensor.any()),
        "max_foot_force_n": float(foot_force_tensor.max()),
        "max_chassis_force_n": float(chassis_force_tensor.max()),
        "foot_contact_by_name": {
            name: bool(foot_contact_tensor[:, index].any()) for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        },
        "foot_force_by_name": {
            name: float(foot_force_tensor[:, index].mean())
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        },
        "chassis_force_by_name": {
            name: float(chassis_force_tensor[:, index].mean())
            for index, name in enumerate(MINI_PUPPER_CHASSIS_CONTACT_BODY_NAMES)
        },
    }


LEG_JOINTS = {
    "lf": ["base_lf1", "lf1_lf2", "lf2_lf3"],
    "rf": ["base_rf1", "rf1_rf2", "rf2_rf3"],
    "lb": ["base_lb1", "lb1_lb2", "lb2_lb3"],
    "rb": ["base_rb1", "rb1_rb2", "rb2_rb3"],
}


def _leg_joint_indices(names: list[str]) -> dict[str, list[int]]:
    return {leg: [names.index(joint_name) for joint_name in joint_names] for leg, joint_names in LEG_JOINTS.items()}


def _camera_eye_lookat(root_pos: torch.Tensor | None = None) -> tuple[list[float], list[float]]:
    follows = args_cli.follow_robot_camera or args_cli.camera_view in ("gait", "gait_wide")
    if follows or args_cli.camera_view != "custom":
        x = float(root_pos[0, 0]) if follows and root_pos is not None else 0.0
        y = float(root_pos[0, 1]) if follows and root_pos is not None else 0.0
    else:
        x = 0.0
        y = 0.0

    if args_cli.camera_view == "gait_wide":
        # Same foot-level intent as "gait", pulled back and aimed at the middle
        # of the *lateral* envelope as well as the forward one. The "gait"
        # preset assumes the robot travels along y = 0; a policy that turns or
        # drifts sideways walks out of that close frame partway through the run.
        eye = [x + 0.45, y - 1.55, 0.16]
        lookat = [x + 0.45, y - 0.20, 0.05]
    elif args_cli.camera_view == "gait":
        # Close, foot-level, side-on, framed on the middle of a ~1 m run so the
        # robot stays in shot without a following camera. About 5x closer than
        # the older presets, so a 20-30 mm foot lift is actually visible.
        eye = [x + 0.45, y - 0.95, 0.09]
        lookat = [x + 0.45, y, 0.05]
    elif args_cli.camera_view == "side":
        eye = [x + 0.50, y - 2.20, 0.45]
        lookat = [x + 0.45, y, 0.07]
    elif args_cli.camera_view == "rear_quarter":
        eye = [x - 1.00, y - 1.70, 0.55]
        lookat = [x + 0.45, y, 0.07]
    elif args_cli.camera_view == "top_oblique":
        eye = [x + 0.55, y - 2.30, 2.20]
        lookat = [x + 0.45, y, 0.05]
    elif args_cli.follow_robot_camera:
        d = args_cli.camera_distance
        eye = [x + d, y - d, args_cli.camera_height]
        lookat = [x + args_cli.camera_target_x, y, args_cli.camera_target_z]
    else:
        d = args_cli.camera_distance
        eye = [d, -d, args_cli.camera_height]
        lookat = [args_cli.camera_target_x, 0.0, args_cli.camera_target_z]
    return eye, lookat


def _set_camera_view(env, root_pos: torch.Tensor) -> None:
    if not hasattr(env.unwrapped.sim, "set_camera_view"):
        return
    eye, lookat = _camera_eye_lookat(root_pos)
    env.unwrapped.sim.set_camera_view(eye, lookat)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else None
    if args_cli.scripted_agent == "policy":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --scripted_agent=policy")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    elif checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args_cli.output_dir:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
    else:
        if checkpoint is None:
            output_dir = Path.cwd() / ("eval_video_scripted" if args_cli.video else "eval_scripted")
        else:
            output_dir = checkpoint.parent / ("eval_video" if args_cli.video else "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    run_stem = checkpoint.stem if checkpoint is not None else f"{args_cli.scripted_agent}_actions"

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    installed_version = metadata.version("rsl-rl-lib")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.log_dir = str(checkpoint.parent if checkpoint is not None else output_dir)
        env_cfg.commands.base_velocity.debug_vis = args_cli.show_command_markers
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        if args_cli.command_yaw is not None:
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (
                float(args_cli.command_yaw[0]), float(args_cli.command_yaw[1]))
        else:
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        viewer_eye, viewer_lookat = _camera_eye_lookat()
        env_cfg.viewer.eye = tuple(viewer_eye)
        env_cfg.viewer.lookat = tuple(viewer_lookat)
        # Camera-following is not available in this render path: neither the
        # per-step sim.set_camera_view() call below nor
        # ViewerCfg.origin_type="asset_root" affects the offscreen render
        # product used by render_mode="rgb_array", so recorded video always has
        # a static camera. The "gait" preset frames the whole expected travel
        # path from close range at foot level, which keeps a 0.2 m robot in
        # shot for the full run.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        render_mode = "rgb_array" if args_cli.video else None
        base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        if args_cli.video:
            video_dir.mkdir(parents=True, exist_ok=True)
            base_env = gym.wrappers.RecordVideo(
                base_env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length or args_cli.steps,
                disable_logger=True,
                name_prefix=run_stem,
            )

        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        _set_camera_view(env, env.unwrapped.scene["robot"].data.root_pos_w.torch.detach())

        policy = None
        if args_cli.scripted_agent == "policy":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint), map_location=agent_cfg.device)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        elif args_cli.scripted_agent == "replay":
            if not args_cli.replay_file:
                raise ValueError("--scripted_agent replay requires --replay_file")
            replay_npz = np.load(args_cli.replay_file)
            replay_traj = torch.as_tensor(
                replay_npz["q_measured"], dtype=torch.float32, device=env.unwrapped.device
            )
            replay_rate = float(replay_npz["rate_hz"])
            env_rate = 1.0 / env.unwrapped.step_dt
            if abs(replay_rate - env_rate) > 0.001 * env_rate:
                raise ValueError(
                    f"replay rate {replay_rate:.3f} Hz != task control rate {env_rate:.3f} Hz; "
                    "resample the trajectory, do not stretch it"
                )
            if replay_traj.shape[1] != env.num_actions:
                raise ValueError(f"replay has {replay_traj.shape[1]} channels, task takes {env.num_actions}")
            print(
                f"REPLAY {args_cli.replay_file}: {replay_traj.shape[0]} samples at "
                f"{replay_rate:.2f} Hz ({replay_traj.shape[0] / replay_rate:.2f} s); "
                f"holding last sample beyond the end"
            )

        robot = env.unwrapped.scene["robot"]
        obs = env.get_observations().to(env.unwrapped.device)
        initial_pos = robot.data.root_pos_w.torch.detach().clone()
        initial_yaw_like = robot.data.projected_gravity_b.torch.detach().clone()
        foot_body_ids = [robot.body_names.index(name) for name in MINI_PUPPER_FOOT_BODY_NAMES]
        num_feet = len(MINI_PUPPER_FOOT_BODY_NAMES)
        action_leg_ids = _leg_joint_indices(MINI_PUPPER_ROS_JOINT_ORDER)
        robot_joint_ids_ros = [robot.joint_names.index(name) for name in MINI_PUPPER_ROS_JOINT_ORDER]
        robot_leg_ids = _leg_joint_indices(MINI_PUPPER_ROS_JOINT_ORDER)

        # One recorder per recorded env so metrics can be reported as a
        # distribution rather than n = 1.
        num_record_envs = max(1, min(args_cli.record_envs, env.num_envs))
        gait_recorders = [
            GaitRecorder(MINI_PUPPER_FOOT_BODY_NAMES, float(env.unwrapped.step_dt))
            for _ in range(num_record_envs)
        ]
        gait_recorder = gait_recorders[0]
        # Optional per-step foot force VECTOR recording (world frame). Norm-based
        # metrics discard direction, and the tangential component is the
        # propulsion question.
        record_forces = bool(os.environ.get("MP2_X_WALK_RECORD_FORCES"))
        force_vec_rows: list[np.ndarray] = []
        foot_vel_rows: list[np.ndarray] = []
        normal_hist_rows: list[np.ndarray] = []
        print(f"PHYSICS_BACKEND {type(env_cfg.sim.physics).__name__}")
        csv_path = output_dir / f"{run_stem}_eval.csv"
        rows = []
        reward_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
        done_count = torch.zeros(env.num_envs, device=env.unwrapped.device)
        timeout_count = torch.zeros(env.num_envs, device=env.unwrapped.device)
        raw_min = float("inf")
        raw_max = float("-inf")
        raw_clip_count = 0
        raw_action_count = 0
        clipped_raw_min = float("inf")
        clipped_raw_max = float("-inf")
        proc_min = float("inf")
        proc_max = float("-inf")
        min_height = float("inf")
        max_height = float("-inf")
        max_tilt_rad = 0.0
        foot_contact_fraction_sum = 0.0
        max_foot_force = 0.0
        max_chassis_force = 0.0
        chassis_contact_steps = 0
        foot_contact_steps = torch.zeros(num_feet, device=env.unwrapped.device)
        foot_transition_counts = torch.zeros(num_feet, device=env.unwrapped.device)
        support_count_sum = 0.0
        support_count_min = float("inf")
        support_count_max = float("-inf")
        support_count_hist = {str(index): 0 for index in range(num_feet + 1)}
        all_feet_contact_steps = 0
        no_feet_contact_steps = 0
        diagonal_pair_match_steps = 0
        diagonal_pair_opposed_steps = 0
        front_pair_match_steps = 0
        rear_pair_match_steps = 0
        left_pair_match_steps = 0
        right_pair_match_steps = 0
        stance_planar_speed_sum = torch.zeros(num_feet, device=env.unwrapped.device)
        stance_planar_speed_count = torch.zeros(num_feet, device=env.unwrapped.device)
        swing_planar_speed_sum = torch.zeros(num_feet, device=env.unwrapped.device)
        swing_planar_speed_count = torch.zeros(num_feet, device=env.unwrapped.device)
        foot_z_min = torch.full((num_feet,), float("inf"), device=env.unwrapped.device)
        foot_z_max = torch.full((num_feet,), float("-inf"), device=env.unwrapped.device)
        foot_xy_travel = torch.zeros(num_feet, device=env.unwrapped.device)
        previous_foot_contacts = None
        previous_foot_pos = None
        leg_raw_action_abs_sum = {leg: 0.0 for leg in LEG_JOINTS}
        leg_raw_action_sq_sum = {leg: 0.0 for leg in LEG_JOINTS}
        leg_clipped_action_abs_sum = {leg: 0.0 for leg in LEG_JOINTS}
        joint_raw_action_abs_sum = torch.zeros(len(MINI_PUPPER_ROS_JOINT_ORDER), device=env.unwrapped.device)
        joint_clipped_action_abs_sum = torch.zeros(len(MINI_PUPPER_ROS_JOINT_ORDER), device=env.unwrapped.device)
        joint_raw_clip_counts = torch.zeros(len(MINI_PUPPER_ROS_JOINT_ORDER), device=env.unwrapped.device)
        joint_raw_action_min = torch.full(
            (len(MINI_PUPPER_ROS_JOINT_ORDER),), float("inf"), device=env.unwrapped.device
        )
        joint_raw_action_max = torch.full(
            (len(MINI_PUPPER_ROS_JOINT_ORDER),), float("-inf"), device=env.unwrapped.device
        )
        leg_target_min = {leg: float("inf") for leg in LEG_JOINTS}
        leg_target_max = {leg: float("-inf") for leg in LEG_JOINTS}
        leg_joint_pos_min = {leg: float("inf") for leg in LEG_JOINTS}
        leg_joint_pos_max = {leg: float("-inf") for leg in LEG_JOINTS}
        leg_joint_vel_abs_sum = {leg: 0.0 for leg in LEG_JOINTS}
        leg_metric_steps = 0

        for step in range(args_cli.steps):
            with torch.inference_mode():
                if args_cli.scripted_agent == "policy":
                    actions = policy(obs)
                elif args_cli.scripted_agent == "replay":
                    actions = replay_traj[min(step, replay_traj.shape[0] - 1)].expand(
                        env.num_envs, env.num_actions
                    )
                elif args_cli.scripted_agent == "zero":
                    actions = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
                else:
                    actions = 2.0 * torch.rand((env.num_envs, env.num_actions), device=env.unwrapped.device) - 1.0
                raw_min = min(raw_min, float(actions.min()))
                raw_max = max(raw_max, float(actions.max()))
                raw_clip_count += int(torch.sum(torch.abs(actions) > 1.0))
                raw_action_count += actions.numel()
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs = obs.to(env.unwrapped.device)
                if policy is not None:
                    policy.reset(dones)

            reward_sum += rewards.to(env.unwrapped.device)
            timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).to(env.unwrapped.device).bool()
            non_timeout_dones = dones.to(env.unwrapped.device).bool() & ~timeouts
            done_count += non_timeout_dones.float()
            timeout_count += timeouts.float()
            action_term = env.unwrapped.action_manager._terms["joint_pos"]
            received_actions = action_term.raw_actions.detach()
            processed_actions = action_term.processed_actions.detach()
            clipped_raw_min = min(clipped_raw_min, float(received_actions.min()))
            clipped_raw_max = max(clipped_raw_max, float(received_actions.max()))
            proc_min = min(proc_min, float(processed_actions.min()))
            proc_max = max(proc_max, float(processed_actions.max()))
            joint_raw_action_abs_sum += torch.abs(actions[0].detach())
            joint_clipped_action_abs_sum += torch.abs(received_actions[0].detach())
            joint_raw_clip_counts += (torch.abs(actions[0].detach()) > 1.0).float()
            joint_raw_action_min = torch.minimum(joint_raw_action_min, actions[0].detach())
            joint_raw_action_max = torch.maximum(joint_raw_action_max, actions[0].detach())
            joint_pos_ros = robot.data.joint_pos.torch.detach()[:, robot_joint_ids_ros]
            joint_vel_ros = robot.data.joint_vel.torch.detach()[:, robot_joint_ids_ros]
            leg_metric_steps += 1
            for leg, ids in action_leg_ids.items():
                raw_leg = actions[0, ids].detach()
                clipped_leg = received_actions[0, ids].detach()
                target_leg = processed_actions[0, ids].detach()
                joint_pos_leg = joint_pos_ros[0, robot_leg_ids[leg]].detach()
                joint_vel_leg = joint_vel_ros[0, robot_leg_ids[leg]].detach()
                leg_raw_action_abs_sum[leg] += float(torch.mean(torch.abs(raw_leg)))
                leg_raw_action_sq_sum[leg] += float(torch.mean(torch.square(raw_leg)))
                leg_clipped_action_abs_sum[leg] += float(torch.mean(torch.abs(clipped_leg)))
                leg_target_min[leg] = min(leg_target_min[leg], float(target_leg.min()))
                leg_target_max[leg] = max(leg_target_max[leg], float(target_leg.max()))
                leg_joint_pos_min[leg] = min(leg_joint_pos_min[leg], float(joint_pos_leg.min()))
                leg_joint_pos_max[leg] = max(leg_joint_pos_max[leg], float(joint_pos_leg.max()))
                leg_joint_vel_abs_sum[leg] += float(torch.mean(torch.abs(joint_vel_leg)))

            root_pos = robot.data.root_pos_w.torch.detach()
            root_quat = robot.data.root_quat_w.torch.detach()
            root_rpy = _quat_xyzw_to_rpy(root_quat)
            projected_gravity = robot.data.projected_gravity_b.torch.detach()
            tilt = _tilt_from_projected_gravity(projected_gravity)
            min_height = min(min_height, float(root_pos[:, 2].min()))
            max_height = max(max_height, float(root_pos[:, 2].max()))
            max_tilt_rad = max(max_tilt_rad, float(tilt.max()))
            root_lin_vel_b = robot.data.root_lin_vel_b.torch.detach()
            root_ang_vel_b = robot.data.root_ang_vel_b.torch.detach()
            if args_cli.follow_robot_camera or args_cli.camera_view in ("gait", "gait_wide"):
                _set_camera_view(env, root_pos)
            command = env.unwrapped.command_manager.get_command("base_velocity").detach()
            contact_state = _per_body_contact_state(env, args_cli.contact_threshold)
            contact_tensors = _per_body_contact_tensors(env, args_cli.contact_threshold)
            if contact_state.get("present"):
                foot_contact_fraction_sum += float(contact_state["foot_contact_fraction_mean"])
                max_foot_force = max(max_foot_force, float(contact_state["max_foot_force_n"]))
                max_chassis_force = max(max_chassis_force, float(contact_state["max_chassis_force_n"]))
                chassis_contact_steps += int(bool(contact_state["chassis_contact_any"]))
            if contact_tensors is not None:
                foot_contacts = contact_tensors["foot_contacts"][0].to(env.unwrapped.device)
                support_count = int(foot_contacts.sum())
                support_count_sum += support_count
                support_count_min = min(support_count_min, support_count)
                support_count_max = max(support_count_max, support_count)
                support_count_hist[str(support_count)] += 1
                all_feet_contact_steps += int(support_count == num_feet)
                no_feet_contact_steps += int(support_count == 0)
                foot_contact_steps += foot_contacts.float()
                lf_contact, rf_contact, lb_contact, rb_contact = [bool(v) for v in foot_contacts]
                diagonal_a_match = lf_contact == rb_contact
                diagonal_b_match = rf_contact == lb_contact
                diagonal_pair_match_steps += int(diagonal_a_match and diagonal_b_match)
                diagonal_pair_opposed_steps += int(diagonal_a_match and diagonal_b_match and lf_contact != rf_contact)
                front_pair_match_steps += int(lf_contact == rf_contact)
                rear_pair_match_steps += int(lb_contact == rb_contact)
                left_pair_match_steps += int(lf_contact == lb_contact)
                right_pair_match_steps += int(rf_contact == rb_contact)
                if previous_foot_contacts is not None:
                    foot_transition_counts += (foot_contacts != previous_foot_contacts).float()
                previous_foot_contacts = foot_contacts.clone()

                foot_pos = robot.data.body_pos_w.torch.detach()[0, foot_body_ids]
                foot_vel = robot.data.body_lin_vel_w.torch.detach()[0, foot_body_ids]
                foot_z_min = torch.minimum(foot_z_min, foot_pos[:, 2])
                foot_z_max = torch.maximum(foot_z_max, foot_pos[:, 2])
                if previous_foot_pos is not None:
                    foot_xy_travel += torch.linalg.norm(foot_pos[:, :2] - previous_foot_pos[:, :2], dim=-1)
                previous_foot_pos = foot_pos.clone()

                try:
                    step_torque = (
                        robot.data.applied_torque.torch.detach()[0, robot_joint_ids_ros].cpu().numpy()
                    )
                except Exception:
                    step_torque = None
                if num_record_envs == 1:
                    gait_recorder.record(
                        foot_contacts.cpu().numpy(),
                        foot_pos.cpu().numpy(),
                        root_pos[0].cpu().numpy(),
                        step_torque,
                        root_quat[0].cpu().numpy(),
                    )
                else:
                    all_contacts = contact_tensors["foot_contacts"][:num_record_envs].cpu().numpy()
                    all_foot_pos = (
                        robot.data.body_pos_w.torch.detach()[:num_record_envs][:, foot_body_ids]
                        .cpu()
                        .numpy()
                    )
                    all_root_pos = root_pos[:num_record_envs].cpu().numpy()
                    all_root_quat = root_quat[:num_record_envs].cpu().numpy()
                    try:
                        all_torque = (
                            robot.data.applied_torque.torch.detach()[:num_record_envs][
                                :, robot_joint_ids_ros
                            ]
                            .cpu()
                            .numpy()
                        )
                    except Exception:
                        all_torque = None
                    for env_index, recorder in enumerate(gait_recorders):
                        recorder.record(
                            all_contacts[env_index],
                            all_foot_pos[env_index],
                            all_root_pos[env_index],
                            None if all_torque is None else all_torque[env_index],
                            all_root_quat[env_index],
                        )

                if record_forces:
                    force_vec_rows.append(
                        contact_tensors["foot_force_vecs"][:num_record_envs].cpu().numpy()
                    )
                    foot_vel_rows.append(
                        robot.data.body_lin_vel_w.torch.detach()[:num_record_envs][:, foot_body_ids]
                        .cpu()
                        .numpy()
                    )
                    normal_hist_rows.append(
                        contact_tensors["foot_normal_hist"][:num_record_envs].cpu().numpy()
                    )

                foot_planar_speed = torch.linalg.norm(foot_vel[:, :2], dim=-1)
                stance_planar_speed_sum += torch.where(foot_contacts, foot_planar_speed, torch.zeros_like(foot_planar_speed))
                stance_planar_speed_count += foot_contacts.float()
                swing_contacts = ~foot_contacts
                swing_planar_speed_sum += torch.where(swing_contacts, foot_planar_speed, torch.zeros_like(foot_planar_speed))
                swing_planar_speed_count += swing_contacts.float()
            if step % max(args_cli.steps // 100, 1) == 0 or step == args_cli.steps - 1:
                foot_pos_for_row = robot.data.body_pos_w.torch.detach()[0, foot_body_ids]
                rows.append(
                    {
                        "step": step,
                        "time_s": step * env.unwrapped.step_dt,
                        "x": float(root_pos[0, 0] - initial_pos[0, 0]),
                        "y": float(root_pos[0, 1] - initial_pos[0, 1]),
                        "z": float(root_pos[0, 2]),
                        "quat_x": float(root_quat[0, 0]),
                        "quat_y": float(root_quat[0, 1]),
                        "quat_z": float(root_quat[0, 2]),
                        "quat_w": float(root_quat[0, 3]),
                        "roll_rad": float(root_rpy[0, 0]),
                        "pitch_rad": float(root_rpy[0, 1]),
                        "yaw_rad": float(root_rpy[0, 2]),
                        "tilt_rad": float(tilt[0]),
                        "tilt_deg": float(torch.rad2deg(tilt[0])),
                        "projected_gravity_x": float(projected_gravity[0, 0]),
                        "projected_gravity_y": float(projected_gravity[0, 1]),
                        "projected_gravity_z": float(projected_gravity[0, 2]),
                        "root_vx_b": float(root_lin_vel_b[0, 0]),
                        "root_vy_b": float(root_lin_vel_b[0, 1]),
                        "root_wz_b": float(root_ang_vel_b[0, 2]),
                        "cmd_x": float(command[0, 0]),
                        "cmd_y": float(command[0, 1]),
                        "cmd_yaw": float(command[0, 2]),
                        "reward": float(rewards[0]),
                        "done": int(dones[0]),
                        "timeout": int(timeouts[0]),
                        "non_timeout_done": int(non_timeout_dones[0]),
                        "foot_contact_fraction": contact_state.get("foot_contact_fraction_mean"),
                        "chassis_contact_any": contact_state.get("chassis_contact_any"),
                        "max_foot_force_n": contact_state.get("max_foot_force_n"),
                        "max_chassis_force_n": contact_state.get("max_chassis_force_n"),
                        "support_count": support_count if contact_tensors is not None else None,
                        **(
                            {
                                f"{name}_contact": int(foot_contacts[index])
                                for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
                            }
                            if contact_tensors is not None
                            else {}
                        ),
                        **{
                            f"{name}_z": float(foot_pos_for_row[index, 2])
                            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
                        },
                        "raw_action_min": float(actions[0].min()),
                        "raw_action_max": float(actions[0].max()),
                        "clipped_raw_action_min": float(received_actions[0].min()),
                        "clipped_raw_action_max": float(received_actions[0].max()),
                        "processed_target_min": float(processed_actions[0].min()),
                        "processed_target_max": float(processed_actions[0].max()),
                        **{
                            f"{leg}_raw_action_abs_mean": float(torch.mean(torch.abs(actions[0, ids])))
                            for leg, ids in action_leg_ids.items()
                        },
                        **{
                            f"{leg}_joint_vel_abs_mean": float(
                                torch.mean(torch.abs(joint_vel_ros[0, robot_leg_ids[leg]]))
                            )
                            for leg in LEG_JOINTS
                        },
                    }
                )

        final_pos = robot.data.root_pos_w.torch.detach().clone()
        final_quat = robot.data.root_quat_w.torch.detach().clone()
        final_rpy = _quat_xyzw_to_rpy(final_quat)
        final_projected_gravity = robot.data.projected_gravity_b.torch.detach().clone()
        final_tilt = _tilt_from_projected_gravity(final_projected_gravity)
        final_lin_vel = robot.data.root_lin_vel_b.torch.detach().clone()
        final_ang_vel = robot.data.root_ang_vel_b.torch.detach().clone()
        command = env.unwrapped.command_manager.get_command("base_velocity").detach().clone()
        elapsed = max(args_cli.steps * env.unwrapped.step_dt, 1e-6)
        displacement = final_pos - initial_pos
        mean_reward_per_step = reward_sum / args_cli.steps
        mean_forward_velocity = displacement[:, 0] / elapsed
        mean_lateral_velocity = displacement[:, 1] / elapsed
        mean_vx_error = torch.abs(mean_forward_velocity - command[:, 0])
        mean_lateral_abs = torch.abs(mean_lateral_velocity)
        mean_yaw_rate_abs = torch.abs(final_ang_vel[:, 2])
        final_tilt_deg = float(torch.rad2deg(final_tilt).mean())
        max_tilt_deg = float(torch.rad2deg(torch.tensor(max_tilt_rad)))
        contact_summary = _per_body_contact_state(env, args_cli.contact_threshold)
        contact_sensor_present = bool(contact_summary.get("present"))
        foot_contact_duty = foot_contact_fraction_sum / args_cli.steps if contact_sensor_present else 0.0
        foot_contact_duty_by_name = {
            name: float((foot_contact_steps[index] / max(args_cli.steps, 1)).detach().cpu())
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        foot_transition_count_by_name = {
            name: int(foot_transition_counts[index].detach().cpu())
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        foot_z_range_by_name = {
            name: float((foot_z_max[index] - foot_z_min[index]).detach().cpu())
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        foot_xy_travel_by_name = {
            name: float(foot_xy_travel[index].detach().cpu()) for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        leg_raw_action_abs_mean = {
            leg: leg_raw_action_abs_sum[leg] / max(leg_metric_steps, 1) for leg in LEG_JOINTS
        }
        leg_raw_action_rms = {
            leg: (leg_raw_action_sq_sum[leg] / max(leg_metric_steps, 1)) ** 0.5 for leg in LEG_JOINTS
        }
        leg_clipped_action_abs_mean = {
            leg: leg_clipped_action_abs_sum[leg] / max(leg_metric_steps, 1) for leg in LEG_JOINTS
        }
        leg_target_range = {
            leg: leg_target_max[leg] - leg_target_min[leg] for leg in LEG_JOINTS
        }
        leg_joint_pos_range = {
            leg: leg_joint_pos_max[leg] - leg_joint_pos_min[leg] for leg in LEG_JOINTS
        }
        leg_joint_vel_abs_mean = {
            leg: leg_joint_vel_abs_sum[leg] / max(leg_metric_steps, 1) for leg in LEG_JOINTS
        }
        joint_raw_action_abs_mean = {
            name: float((joint_raw_action_abs_sum[index] / max(leg_metric_steps, 1)).detach().cpu())
            for index, name in enumerate(MINI_PUPPER_ROS_JOINT_ORDER)
        }
        joint_clipped_action_abs_mean = {
            name: float((joint_clipped_action_abs_sum[index] / max(leg_metric_steps, 1)).detach().cpu())
            for index, name in enumerate(MINI_PUPPER_ROS_JOINT_ORDER)
        }
        joint_raw_action_clip_fraction = {
            name: float((joint_raw_clip_counts[index] / max(leg_metric_steps, 1)).detach().cpu())
            for index, name in enumerate(MINI_PUPPER_ROS_JOINT_ORDER)
        }
        joint_raw_action_range = {
            name: [
                float(joint_raw_action_min[index].detach().cpu()),
                float(joint_raw_action_max[index].detach().cpu()),
            ]
            for index, name in enumerate(MINI_PUPPER_ROS_JOINT_ORDER)
        }
        front_raw_action = (leg_raw_action_abs_mean["lf"] + leg_raw_action_abs_mean["rf"]) / 2.0
        rear_raw_action = (leg_raw_action_abs_mean["lb"] + leg_raw_action_abs_mean["rb"]) / 2.0
        front_clipped_action = (leg_clipped_action_abs_mean["lf"] + leg_clipped_action_abs_mean["rf"]) / 2.0
        rear_clipped_action = (leg_clipped_action_abs_mean["lb"] + leg_clipped_action_abs_mean["rb"]) / 2.0
        front_joint_range = (leg_joint_pos_range["lf"] + leg_joint_pos_range["rf"]) / 2.0
        rear_joint_range = (leg_joint_pos_range["lb"] + leg_joint_pos_range["rb"]) / 2.0
        front_foot_z_range = (foot_z_range_by_name["lffoot"] + foot_z_range_by_name["rffoot"]) / 2.0
        rear_foot_z_range = (foot_z_range_by_name["lbfoot"] + foot_z_range_by_name["rbfoot"]) / 2.0
        stance_planar_speed_by_name = {
            name: float(
                (
                    stance_planar_speed_sum[index]
                    / torch.clamp(stance_planar_speed_count[index], min=1.0)
                ).detach().cpu()
            )
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        swing_planar_speed_by_name = {
            name: float(
                (
                    swing_planar_speed_sum[index]
                    / torch.clamp(swing_planar_speed_count[index], min=1.0)
                ).detach().cpu()
            )
            for index, name in enumerate(MINI_PUPPER_FOOT_BODY_NAMES)
        }
        mean_support_count = support_count_sum / max(args_cli.steps, 1)
        support_count_min = 0.0 if support_count_min == float("inf") else support_count_min
        support_count_max = 0.0 if support_count_max == float("-inf") else support_count_max
        front_transition_count = foot_transition_count_by_name["lffoot"] + foot_transition_count_by_name["rffoot"]
        rear_transition_count = foot_transition_count_by_name["lbfoot"] + foot_transition_count_by_name["rbfoot"]
        fall_detected = (
            float(done_count.mean()) > 0.0
            or min_height < args_cli.fall_height
            or max_tilt_deg > args_cli.fall_tilt_deg
            or final_tilt_deg > args_cli.unhealthy_final_tilt_deg
        )
        leg_supported_locomotion_detected = (
            not fall_detected
            and contact_sensor_present
            and chassis_contact_steps == 0
            and float(mean_forward_velocity.mean()) > 0.05
            and sum(foot_transition_count_by_name.values()) >= 4
            and max(foot_z_range_by_name.values()) > 0.003
            and all(value > 0.10 for value in foot_contact_duty_by_name.values())
            and mean_support_count >= 1.0
        )

        summary = {
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "scripted_agent": args_cli.scripted_agent,
            "task": args_cli.task,
            "steps": args_cli.steps,
            "step_dt": float(env.unwrapped.step_dt),
            "elapsed_s": float(elapsed),
            "command_x": args_cli.command_x,
            "num_envs": args_cli.num_envs,
            "mean_reward_per_step": float(mean_reward_per_step.mean()),
            "mean_forward_displacement_m": float(displacement[:, 0].mean()),
            "mean_lateral_displacement_m": float(displacement[:, 1].mean()),
            "mean_forward_velocity_mps": float(mean_forward_velocity.mean()),
            "mean_lateral_velocity_mps": float(mean_lateral_velocity.mean()),
            "mean_abs_forward_velocity_error_mps": float(mean_vx_error.mean()),
            "mean_abs_lateral_velocity_mps": float(mean_lateral_abs.mean()),
            "final_mean_abs_yaw_rate_rps": float(mean_yaw_rate_abs.mean()),
            "initial_mean_body_height_m": float(initial_pos[:, 2].mean()),
            "final_mean_body_height_m": float(final_pos[:, 2].mean()),
            "min_body_height_m": min_height,
            "max_body_height_m": max_height,
            "final_mean_roll_rad": float(final_rpy[:, 0].mean()),
            "final_mean_pitch_rad": float(final_rpy[:, 1].mean()),
            "final_mean_yaw_rad": float(final_rpy[:, 2].mean()),
            "final_mean_tilt_rad": float(final_tilt.mean()),
            "final_mean_tilt_deg": final_tilt_deg,
            "max_tilt_rad": max_tilt_rad,
            "max_tilt_deg": max_tilt_deg,
            "fall_detected": bool(fall_detected),
            "fall_height_threshold_m": args_cli.fall_height,
            "fall_tilt_threshold_deg": args_cli.fall_tilt_deg,
            "unhealthy_final_tilt_threshold_deg": args_cli.unhealthy_final_tilt_deg,
            "contact_threshold_n": args_cli.contact_threshold,
            "contact_sensor_present": contact_sensor_present,
            "foot_contact_duty": foot_contact_duty,
            "foot_contact_duty_by_name": foot_contact_duty_by_name,
            "foot_transition_count_by_name": foot_transition_count_by_name,
            "foot_z_range_by_name": foot_z_range_by_name,
            "foot_xy_travel_by_name": foot_xy_travel_by_name,
            "leg_raw_action_abs_mean": leg_raw_action_abs_mean,
            "leg_raw_action_rms": leg_raw_action_rms,
            "leg_clipped_action_abs_mean": leg_clipped_action_abs_mean,
            "joint_raw_action_abs_mean": joint_raw_action_abs_mean,
            "joint_clipped_action_abs_mean": joint_clipped_action_abs_mean,
            "joint_raw_action_clip_fraction": joint_raw_action_clip_fraction,
            "joint_raw_action_range": joint_raw_action_range,
            "leg_processed_target_range": leg_target_range,
            "leg_joint_pos_range": leg_joint_pos_range,
            "leg_joint_vel_abs_mean": leg_joint_vel_abs_mean,
            "rear_front_raw_action_abs_ratio": rear_raw_action / max(front_raw_action, 1.0e-6),
            "rear_front_clipped_action_abs_ratio": rear_clipped_action / max(front_clipped_action, 1.0e-6),
            "rear_front_joint_pos_range_ratio": rear_joint_range / max(front_joint_range, 1.0e-6),
            "rear_front_foot_z_range_ratio": rear_foot_z_range / max(front_foot_z_range, 1.0e-6),
            "stance_planar_speed_by_name": stance_planar_speed_by_name,
            "swing_planar_speed_by_name": swing_planar_speed_by_name,
            "mean_support_count": mean_support_count,
            "min_support_count": support_count_min,
            "max_support_count": support_count_max,
            "support_count_hist": support_count_hist,
            "all_feet_contact_fraction": all_feet_contact_steps / max(args_cli.steps, 1),
            "no_feet_contact_fraction": no_feet_contact_steps / max(args_cli.steps, 1),
            "diagonal_pair_match_fraction": diagonal_pair_match_steps / max(args_cli.steps, 1),
            "diagonal_pair_opposed_fraction": diagonal_pair_opposed_steps / max(args_cli.steps, 1),
            "front_pair_match_fraction": front_pair_match_steps / max(args_cli.steps, 1),
            "rear_pair_match_fraction": rear_pair_match_steps / max(args_cli.steps, 1),
            "left_pair_match_fraction": left_pair_match_steps / max(args_cli.steps, 1),
            "right_pair_match_fraction": right_pair_match_steps / max(args_cli.steps, 1),
            "front_transition_count": front_transition_count,
            "rear_transition_count": rear_transition_count,
            "rear_front_transition_ratio": rear_transition_count / max(front_transition_count, 1),
            "left_transition_count": foot_transition_count_by_name["lffoot"] + foot_transition_count_by_name["lbfoot"],
            "right_transition_count": foot_transition_count_by_name["rffoot"] + foot_transition_count_by_name["rbfoot"],
            "leg_supported_locomotion_detected": bool(leg_supported_locomotion_detected),
            "chassis_contact_steps": chassis_contact_steps,
            "max_foot_force_n": max_foot_force,
            "max_chassis_force_n": max_chassis_force,
            "final_contact": contact_summary,
            "raw_action_min": raw_min,
            "raw_action_max": raw_max,
            "raw_action_clip_fraction": raw_clip_count / max(raw_action_count, 1),
            "clipped_raw_action_min": clipped_raw_min,
            "clipped_raw_action_max": clipped_raw_max,
            "processed_target_min": proc_min,
            "processed_target_max": proc_max,
            "done_count_mean": float(done_count.mean()),
            "timeout_count_mean": float(timeout_count.mean()),
            "video_dir": str(video_dir) if args_cli.video else None,
            "follow_robot_camera": bool(args_cli.follow_robot_camera),
            "camera_view": args_cli.camera_view,
            "initial_projected_gravity_mean": _tensor_row(initial_yaw_like.mean(dim=0, keepdim=True), 0),
            "final_projected_gravity_mean": _tensor_row(final_projected_gravity.mean(dim=0, keepdim=True), 0),
            "final_root_lin_vel_b_mean": _tensor_row(final_lin_vel.mean(dim=0, keepdim=True), 0),
            "final_root_ang_vel_b_mean": _tensor_row(final_ang_vel.mean(dim=0, keepdim=True), 0),
        }

        if num_record_envs == 1:
            summary.update(gait_recorder.summary())
            summary["gait_traces"] = gait_recorder.write_traces(output_dir, run_stem)
        else:
            per_env = [recorder.summary() for recorder in gait_recorders]
            summary.update(aggregate_summaries(per_env))
            summary["gait_per_env"] = per_env
            summary["gait_traces"] = write_traces_multi(output_dir, run_stem, gait_recorders)
        summary["record_envs"] = num_record_envs
        if record_forces and force_vec_rows:
            forces_path = output_dir / f"{run_stem}_foot_forces.npz"
            force_arrays = {
                # (T, E, F, 3) -> (E, T, F, 3) to match the gait-trace layout
                "foot_forces_w": np.stack(force_vec_rows).transpose(1, 0, 2, 3),
                "foot_vel_w": np.stack(foot_vel_rows).transpose(1, 0, 2, 3),
                # (T, E, H, F) -> (E, T, H, F): normal force at EVERY physics
                # substep (H == decimation), the anti-aliased contact record
                "foot_normal_hist": np.stack(normal_hist_rows).transpose(1, 0, 2, 3),
                "foot_names": np.array(MINI_PUPPER_FOOT_BODY_NAMES),
                "step_dt": float(env.unwrapped.step_dt),
                "physics_dt": float(env.unwrapped.physics_dt),
            }
            np.savez_compressed(forces_path, **force_arrays)
            summary["foot_forces"] = str(forces_path)
        summary["gait_plots"] = gait_recorder.write_plots(
            output_dir, run_stem, f"{run_stem} cmd_x={args_cli.command_x}"
        )

        if rows:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        if args_cli.command_yaw is not None:
            summary["command_yaw_range"] = [float(args_cli.command_yaw[0]),
                                            float(args_cli.command_yaw[1])]
            try:
                cmd = env.unwrapped.command_manager.get_command("base_velocity")
                summary["command_z_per_env"] = [float(v) for v in cmd[:, 2].cpu().tolist()]
            except Exception as exc:  # noqa: BLE001 — readback is best-effort, absence is loud
                summary["command_z_per_env"] = f"READBACK_FAILED: {exc}"
        json_path = output_dir / f"{run_stem}_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        env.close()

        print(f"CHECKPOINT {checkpoint}" if checkpoint is not None else f"SCRIPTED_AGENT {args_cli.scripted_agent}")
        print(f"EVAL_SUMMARY {json_path}")
        print(f"EVAL_CSV {csv_path}")
        print(f"MEAN_REWARD_PER_STEP {summary['mean_reward_per_step']:.6f}")
        print(f"MEAN_FORWARD_DISPLACEMENT_M {summary['mean_forward_displacement_m']:.6f}")
        print(f"MEAN_FORWARD_VELOCITY_MPS {summary['mean_forward_velocity_mps']:.6f}")
        print(f"MEAN_ABS_FORWARD_VELOCITY_ERROR_MPS {summary['mean_abs_forward_velocity_error_mps']:.6f}")
        print(f"MEAN_ABS_LATERAL_VELOCITY_MPS {summary['mean_abs_lateral_velocity_mps']:.6f}")
        print(f"FINAL_MEAN_BODY_HEIGHT_M {summary['final_mean_body_height_m']:.6f}")
        print(f"MIN_BODY_HEIGHT_M {summary['min_body_height_m']:.6f}")
        print(f"FINAL_MEAN_TILT_DEG {summary['final_mean_tilt_deg']:.6f}")
        print(f"MAX_TILT_DEG {summary['max_tilt_deg']:.6f}")
        print(f"FALL_DETECTED {int(summary['fall_detected'])}")
        print(f"FOOT_CONTACT_DUTY {summary['foot_contact_duty']:.6f}")
        print(f"MEAN_SUPPORT_COUNT {summary['mean_support_count']:.6f}")
        print(f"FOOT_TRANSITION_COUNTS {summary['foot_transition_count_by_name']}")
        print(f"FOOT_Z_RANGE_M {summary['foot_z_range_by_name']}")
        print(f"STANCE_PLANAR_SPEED_MPS {summary['stance_planar_speed_by_name']}")
        print(f"SUPPORT_COUNT_HIST {summary['support_count_hist']}")
        print(f"DIAGONAL_PAIR_MATCH_FRACTION {summary['diagonal_pair_match_fraction']:.6f}")
        print(f"DIAGONAL_PAIR_OPPOSED_FRACTION {summary['diagonal_pair_opposed_fraction']:.6f}")
        print(f"FRONT_REAR_TRANSITION_COUNTS {summary['front_transition_count']},{summary['rear_transition_count']}")
        print(f"REAR_FRONT_TRANSITION_RATIO {summary['rear_front_transition_ratio']:.6f}")
        print(f"LEG_RAW_ACTION_ABS_MEAN {summary['leg_raw_action_abs_mean']}")
        print(f"LEG_JOINT_POS_RANGE {summary['leg_joint_pos_range']}")
        print(f"REAR_FRONT_RAW_ACTION_ABS_RATIO {summary['rear_front_raw_action_abs_ratio']:.6f}")
        print(f"REAR_FRONT_CLIPPED_ACTION_ABS_RATIO {summary['rear_front_clipped_action_abs_ratio']:.6f}")
        print(f"REAR_FRONT_JOINT_POS_RANGE_RATIO {summary['rear_front_joint_pos_range_ratio']:.6f}")
        print(f"REAR_FRONT_FOOT_Z_RANGE_RATIO {summary['rear_front_foot_z_range_ratio']:.6f}")
        print(f"JOINT_RAW_ACTION_CLIP_FRACTION {summary['joint_raw_action_clip_fraction']}")
        print(f"LEG_SUPPORTED_LOCOMOTION_DETECTED {int(summary['leg_supported_locomotion_detected'])}")
        print(f"CHASSIS_CONTACT_STEPS {summary['chassis_contact_steps']}")
        print(f"MAX_CHASSIS_FORCE_N {summary['max_chassis_force_n']:.6f}")
        print(f"RAW_ACTION_MIN_MAX {raw_min:.6f},{raw_max:.6f}")
        print(f"RAW_ACTION_CLIP_FRACTION {summary['raw_action_clip_fraction']:.6f}")
        print(f"CLIPPED_RAW_ACTION_MIN_MAX {clipped_raw_min:.6f},{clipped_raw_max:.6f}")
        print(f"PROCESSED_TARGET_MIN_MAX {proc_min:.6f},{proc_max:.6f}")
        if args_cli.video:
            print(f"VIDEO_DIR {video_dir}")
        if summary.get("gait_metrics_present"):
            # A multi-env summary is an aggregate over AGGREGATE_KEYS and does
            # not carry every single-rollout key, so read defensively.
            for label, key, fmt in (
                ("SWING_CLEARANCE_MEDIAN_M", "swing_clearance_median_m", "{:.6f}"),
                ("SWING_CLEARANCE_MIN_FOOT_M", "swing_clearance_min_foot_m", "{:.6f}"),
                ("FOOT_PATH_EXCESS", "foot_path_excess", "{:.4f}"),
                ("STANCE_PATH_FRACTION", "stance_path_fraction", "{:.4f}"),
                ("DUTY_FACTOR_MEAN", "duty_factor_mean", "{:.4f}"),
                ("CADENCE_HZ", "cadence_hz", "{:.4f}"),
                ("STRIDE_MEDIAN_M", "stride_median_m", "{:.6f}"),
                ("SWING_COUNT_SPREAD", "swing_count_spread", "{:.4f}"),
                ("SWING_DURATION_CV", "swing_duration_cv", "{:.4f}"),
                ("PER_FOOT_LIFT_SPREAD_M", "per_foot_lift_spread_m", "{:.6f}"),
                ("PER_FOOT_DUTY_SPREAD", "per_foot_duty_spread", "{:.4f}"),
                ("BASE_HEIGHT_STD_M", "base_height_std_m", "{:.6f}"),
                ("YAW_DRIFT_DEG", "yaw_drift_deg", "{:+.3f}"),
                ("PITCH_MEDIAN_DEG", "pitch_median_deg", "{:+.3f}"),
                ("JOINT_TORQUE_ABS_MEAN_NM", "joint_torque_abs_mean_nm", "{:.4f}"),
                ("JOINT_TORQUE_ABS_MAX_NM", "joint_torque_abs_max_nm", "{:.4f}"),
            ):
                if key in summary and isinstance(summary[key], (int, float)):
                    print(f"{label} {fmt.format(summary[key])}")
        print("EVAL_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
