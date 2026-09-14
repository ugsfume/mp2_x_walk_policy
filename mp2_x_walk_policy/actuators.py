"""Servo models for the Mini Pupper V2.

Isaac Lab ships torque-speed saturation (``DCMotor``) and a command delay
buffer (``DelayedPDActuator``) as separate classes, and the latter drops the
torque-speed curve. The real servo needs both: the measured command-to-motion
transport delay is ~70 ms, and the drive saturates. ``DelayedDCMotor`` combines
them; ``BacklashDCMotor`` adds the lost motion measured at the toe.

Delay is counted in PHYSICS steps, matching ``DelayBuffer`` semantics --
``compute()`` runs once per ``sim.dt`` inside the decimation loop. With
``sim.dt = 0.002`` and ``decimation = 10``, 70 ms is 35 physics steps
(3.5 control steps). Do not specify it in control steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class DelayedDCMotor(DCMotor):
    """DC motor with a delayed command buffer.

    Mirrors :class:`isaaclab.actuators.DelayedPDActuator` -- the setpoints are
    pushed through a circular buffer before the actuator model runs -- but
    keeps :class:`DCMotor` as the base so ``_clip_effort`` still applies the
    linear four-quadrant torque-speed curve.
    """

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # A fresh delay per environment per reset. The measured 70 ms is a
        # round trip that mixes actuator lag with the sensor read-back path,
        # and the split between the two is not identifiable, so the value is
        # randomised over a band rather than fixed.
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buffer in (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        ):
            buffer.set_time_lag(time_lags, env_ids)
            buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for :class:`DelayedDCMotor`."""

    class_type: type = DelayedDCMotor

    min_delay: int = 0
    """Minimum command delay, in PHYSICS steps (not control steps)."""

    max_delay: int = 0
    """Maximum command delay, in PHYSICS steps (not control steps)."""


class BacklashDCMotor(DelayedDCMotor):
    """DelayedDCMotor plus drive-side lost motion (backlash).

    On the robot the toe moves 3-4 mm before the servo resists, about
    0.05-0.06 rad at the joint. ``backlash_rad`` is the FULL band in joint
    radians. The commanded position is filtered through the play operator
    ``y = clip(y_prev, u - b/2, u + b/2)`` before the (optionally delayed) PD,
    so the output shaft only follows once the band is crossed. With
    min/max_delay at 0 this is a backlash-only actuator.
    """

    cfg: "BacklashDCMotorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._play = None

    def reset(self, env_ids):
        super().reset(env_ids)
        if self._play is not None:
            self._play[env_ids] = 0.0

    def compute(self, control_action, joint_pos, joint_vel):
        band = float(self.cfg.backlash_rad)
        if band > 0.0 and control_action.joint_positions is not None:
            u = control_action.joint_positions
            if self._play is None:
                self._play = u.clone()
            half = 0.5 * band
            self._play = torch.clamp(self._play, min=u - half, max=u + half)
            control_action.joint_positions = self._play.clone()
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class BacklashDCMotorCfg(DelayedDCMotorCfg):
    """Configuration for :class:`BacklashDCMotor`."""

    class_type: type = BacklashDCMotor

    backlash_rad: float = 0.0
    """Full lost-motion band, joint radians. 0 disables the play operator."""
