"""Reward, observation, termination and curriculum terms used by the two tasks.

Everything else the tasks use comes from Isaac Lab's stock locomotion mdp.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase


# ------------------------------------------------------------------ helpers


def _quat_xyzw_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sensor_contact(env, sensor_names: list[str], threshold: float) -> torch.Tensor:
    contacts = []
    for sensor_name in sensor_names:
        sensor = env.scene.sensors[sensor_name]
        net_forces = sensor.data.net_forces_w_history.torch
        force_norm = torch.linalg.norm(net_forces, dim=-1)
        contacts.append(torch.amax(force_norm, dim=(1, 2)) > threshold)
    return torch.stack(contacts, dim=1)


def _body_ids(env, body_names: list[str]) -> list[int]:
    robot = env.scene["robot"]
    return [robot.body_names.index(name) for name in body_names]


def _phase(env, frequency: float) -> torch.Tensor:
    time = env.episode_length_buf.to(dtype=torch.float32, device=env.device) * env.step_dt
    return torch.remainder(2.0 * math.pi * frequency * time, 2.0 * math.pi)


def _phase_pair_stance(phase: torch.Tensor) -> torch.Tensor:
    return torch.cos(phase) >= 0.0


# ------------------------------------------------------------- observations


def gait_phase_clock(env, frequency: float = 2.0) -> torch.Tensor:
    """[sin, cos] of the gait phase. The CPG action term uses the same clock,
    so this is the phase the residual policy is acting against."""

    phase = _phase(env, frequency)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


# ------------------------------------------------------------- terminations


def any_contact(env, sensor_names: list[str], threshold: float = 1.0) -> torch.Tensor:
    """True for envs where any named contact sensor exceeds ``threshold``."""

    first_forces = env.scene.sensors[sensor_names[0]].data.net_forces_w_history.torch
    contact = torch.zeros(first_forces.shape[0], dtype=torch.bool, device=first_forces.device)
    for sensor_name in sensor_names:
        sensor = env.scene.sensors[sensor_name]
        net_forces = sensor.data.net_forces_w_history.torch
        force_norm = torch.linalg.norm(net_forces, dim=-1)
        contact |= torch.amax(force_norm, dim=(1, 2)) > threshold
    return contact


# ------------------------------------------------- rewards shared by both tasks


def yaw_l2(env, asset_name: str = "robot") -> torch.Tensor:
    """Penalize heading drift from the spawn heading (world frame)."""

    asset = env.scene[asset_name]
    yaw = _quat_xyzw_to_yaw(asset.data.root_quat_w.torch)
    return torch.square(yaw)


def lateral_position_l2(env, asset_name: str = "robot") -> torch.Tensor:
    """Penalize lateral displacement from the env origin."""

    asset = env.scene[asset_name]
    lateral = asset.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(lateral)


def lateral_velocity_l2(env, asset_name: str = "robot") -> torch.Tensor:
    """Penalize lateral body-frame velocity."""

    asset = env.scene[asset_name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def stance_foot_slip_l2(
    env,
    sensor_names: list[str],
    foot_body_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize planar foot speed while a foot reports contact."""

    asset = env.scene["robot"]
    contacts = _sensor_contact(env, sensor_names, threshold).float()
    foot_ids = _body_ids(env, foot_body_names)
    foot_planar_speed = torch.linalg.norm(asset.data.body_lin_vel_w.torch[:, foot_ids, :2], dim=-1)
    return torch.sum(contacts * torch.square(foot_planar_speed), dim=1)


# ---------------------------------------- rewards of the CPG-residual task


def rear_contact_transitions(
    env,
    rear_sensor_names: list[str],
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward rear-foot touchdowns while a forward command is active."""

    transition_sum = None
    for sensor_name in rear_sensor_names:
        sensor = env.scene.sensors[sensor_name]
        first_contact = sensor.compute_first_contact(env.step_dt).torch[:, 0].float()
        transition_sum = first_contact if transition_sum is None else transition_sum + first_contact
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > 0.01
    return transition_sum * moving.float()


def contact_transition_imbalance_l2(
    env,
    front_sensor_names: list[str],
    rear_sensor_names: list[str],
) -> torch.Tensor:
    """Penalize front/rear touchdown-count imbalance."""

    front_transition = None
    for sensor_name in front_sensor_names:
        sensor = env.scene.sensors[sensor_name]
        transition = sensor.compute_first_contact(env.step_dt).torch[:, 0].float()
        front_transition = transition if front_transition is None else front_transition + transition

    rear_transition = None
    for sensor_name in rear_sensor_names:
        sensor = env.scene.sensors[sensor_name]
        transition = sensor.compute_first_contact(env.step_dt).torch[:, 0].float()
        rear_transition = transition if rear_transition is None else rear_transition + transition

    return torch.square(front_transition - rear_transition)


def rear_air_fraction(
    env,
    rear_sensor_names: list[str],
    command_name: str = "base_velocity",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward rear feet spending time in swing while moving."""

    contacts = _sensor_contact(env, rear_sensor_names, threshold)
    rear_air = (~contacts).float().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > 0.01
    return rear_air * moving.float()


def rear_relative_swing_clearance(
    env,
    all_foot_body_names: list[str],
    rear_sensor_names: list[str],
    rear_foot_body_names: list[str],
    command_name: str = "base_velocity",
    threshold: float = 1.0,
    target_lift: float = 0.010,
) -> torch.Tensor:
    """Reward rear swing-foot lift relative to the lowest foot in the frame."""

    asset = env.scene["robot"]
    all_ids = _body_ids(env, all_foot_body_names)
    rear_ids = _body_ids(env, rear_foot_body_names)
    all_z = asset.data.body_pos_w.torch[:, all_ids, 2]
    rear_z = asset.data.body_pos_w.torch[:, rear_ids, 2]
    baseline = torch.amin(all_z, dim=1, keepdim=True)
    rear_lift = torch.clamp(rear_z - baseline, min=0.0, max=target_lift) / target_lift
    rear_contacts = _sensor_contact(env, rear_sensor_names, threshold)
    swing = (~rear_contacts).float()
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > 0.01
    return torch.sum(swing * rear_lift, dim=1) * moving.float()


def front_rear_relative_lift_imbalance_l2(
    env,
    all_foot_body_names: list[str],
    front_foot_body_names: list[str],
    rear_foot_body_names: list[str],
    target_lift: float = 0.010,
) -> torch.Tensor:
    """Penalize larger front-than-rear relative foot lift."""

    asset = env.scene["robot"]
    all_ids = _body_ids(env, all_foot_body_names)
    front_ids = _body_ids(env, front_foot_body_names)
    rear_ids = _body_ids(env, rear_foot_body_names)
    all_z = asset.data.body_pos_w.torch[:, all_ids, 2]
    baseline = torch.amin(all_z, dim=1, keepdim=True)
    front_lift = torch.clamp(asset.data.body_pos_w.torch[:, front_ids, 2] - baseline, min=0.0).mean(dim=1)
    rear_lift = torch.clamp(asset.data.body_pos_w.torch[:, rear_ids, 2] - baseline, min=0.0).mean(dim=1)
    imbalance = torch.clamp(front_lift - rear_lift, min=0.0) / target_lift
    return torch.square(imbalance)


def diagonal_trot_contact_match(
    env,
    foot_sensor_names: list[str],
    command_name: str = "base_velocity",
    frequency: float = 2.0,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching the LF/RB then RF/LB diagonal trot contact schedule."""

    contacts = _sensor_contact(env, foot_sensor_names, threshold)
    phase = _phase(env, frequency)
    pair_a_stance = _phase_pair_stance(phase)
    expected = torch.stack(
        (
            pair_a_stance,
            ~pair_a_stance,
            ~pair_a_stance,
            pair_a_stance,
        ),
        dim=1,
    )
    match = (contacts == expected).float().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > 0.01
    return match * moving.float()


def diagonal_trot_swing_clearance(
    env,
    all_foot_body_names: list[str],
    command_name: str = "base_velocity",
    frequency: float = 2.0,
    target_lift: float = 0.010,
) -> torch.Tensor:
    """Reward the scheduled swing feet of the diagonal trot for relative lift."""

    asset = env.scene["robot"]
    foot_ids = _body_ids(env, all_foot_body_names)
    foot_z = asset.data.body_pos_w.torch[:, foot_ids, 2]
    baseline = torch.amin(foot_z, dim=1, keepdim=True)
    relative_lift = torch.clamp(foot_z - baseline, min=0.0, max=target_lift) / target_lift
    phase = _phase(env, frequency)
    pair_a_stance = _phase_pair_stance(phase)
    swing_expected = torch.stack(
        (
            ~pair_a_stance,
            pair_a_stance,
            pair_a_stance,
            ~pair_a_stance,
        ),
        dim=1,
    ).float()
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :2], dim=1) > 0.01
    return torch.sum(swing_expected * relative_lift, dim=1) * moving.float()


# ------------------------------------------- rewards of the end-to-end task


class SwingProgressTouchdown(ManagerTermBase):
    """Reward a lifted foot for moving forward relative to the body, but only
    if it has recently been on the ground.

    Without the recency requirement the term has a degenerate optimum: a foot
    waved fore and aft in the air passes the motion test on every forward
    stroke, is permanently the highest foot, and never bears weight. A real
    swing lands every cycle and is unaffected; a parked leg stops earning
    after one ``touchdown_window_s``.

    "Down" is geometric -- within ``planted_tolerance`` of the lowest foot --
    rather than the contact flag, which over-reports on this asset and
    cannot be trusted for a reward.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._foot_ids = _body_ids(env, cfg.params["foot_body_names"])
        num_feet = len(self._foot_ids)
        # Steps since each foot was last near the stance plane. Starts at zero:
        # every foot begins the episode standing.
        self._since_down = torch.zeros(
            (env.num_envs, num_feet), dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._since_down.zero_()
        else:
            self._since_down[env_ids] = 0.0

    def __call__(
        self,
        env,
        foot_body_names: list[str],
        command_name: str = "base_velocity",
        lift_threshold: float = 0.008,
        target_lift: float = 0.020,
        reference_speed: float = 0.15,
        max_rewarded_feet: float = 1.0,
        command_deadband: float = 0.02,
        touchdown_window_s: float = 0.40,
        planted_tolerance: float = 0.005,
    ) -> torch.Tensor:
        asset = env.scene["robot"]

        foot_pos = asset.data.body_pos_w.torch[:, self._foot_ids, :]
        lift = foot_pos[:, :, 2] - torch.amin(foot_pos[:, :, 2], dim=1, keepdim=True)

        # Update the recency buffer before it is used, so a foot that is down
        # this step earns this step.
        down_now = lift < planted_tolerance
        self._since_down = torch.where(
            down_now, torch.zeros_like(self._since_down), self._since_down + env.step_dt
        )
        recently_down = (self._since_down <= touchdown_window_s).float()

        span = max(target_lift - lift_threshold, 1e-6)
        lift_credit = torch.clamp((lift - lift_threshold) / span, min=0.0, max=1.0)

        foot_vel = asset.data.body_lin_vel_w.torch[:, self._foot_ids, :2]
        root_vel = asset.data.root_lin_vel_w.torch[:, None, :2]
        relative_vel = foot_vel - root_vel

        yaw = _quat_xyzw_to_yaw(asset.data.root_quat_w.torch)
        forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)[:, None, :]
        forward_speed = torch.sum(relative_vel * forward, dim=-1)
        progress = torch.clamp(forward_speed / reference_speed, min=0.0, max=1.0)

        reward = torch.clamp(
            torch.sum(lift_credit * progress * recently_down, dim=1), max=max_rewarded_feet
        )
        command = env.command_manager.get_command(command_name)
        moving = torch.linalg.norm(command[:, :2], dim=1) > command_deadband
        return reward * moving.float()


# ---------------------------------------------------------------- curriculum


def ramp_observation_noise(
    env,
    env_ids,
    term_names: list[str],
    start_amps: list[float],
    end_amps: list[float],
    hold_iters: int,
    ramp_iters: int,
    steps_per_iter: int = 24,
) -> float:
    """Linearly ramp the uniform observation-noise amplitudes over training.

    Hold ``start_amps`` for ``hold_iters`` learning iterations, then move
    linearly to ``end_amps`` over ``ramp_iters``. The observation manager
    stores noise cfgs by reference and reads n_min/n_max on every call, so
    mutating them in place takes effect on the same step. The iteration is
    ``env.common_step_counter // steps_per_iter`` (the runner steps the env
    exactly ``num_steps_per_env`` times per iteration; the counter restarts
    at 0 when training resumes from a checkpoint, which is what the
    warm-start needs). Returns the ramp fraction, logged as
    ``Curriculum/<term name>``.

    Never start a ramp from ``noise=None`` -- the manager nulls the cfg at
    init when corruption is off. The training config declares start-amplitude
    noise cfgs explicitly.
    """
    it = env.common_step_counter // steps_per_iter
    if it <= hold_iters:
        frac = 0.0
    else:
        frac = min(1.0, (it - hold_iters) / max(ramp_iters, 1))
    manager = env.observation_manager
    names = manager._group_obs_term_names["policy"]
    cfgs = manager._group_obs_term_cfgs["policy"]
    for tname, a0, a1 in zip(term_names, start_amps, end_amps):
        amp = a0 + frac * (a1 - a0)
        nc = cfgs[names.index(tname)].noise
        nc.n_min = -amp
        nc.n_max = amp
    return frac
