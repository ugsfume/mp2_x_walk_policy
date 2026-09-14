"""Action term for the CPG-residual task (phase 1).

The policy output is a residual added to a deterministic diagonal-trot joint
trajectory: LF/RB and RF/LB move in anti-phase at a fixed frequency, the hip
swings fore-aft on a cosine, the knee flexes during the positive half of the
sine (swing). Amplitudes scale with the commanded speed. The residual scale
(0.12 rad) bounds how far the policy can move away from the prior.

The end-to-end task (phase 2) does not use this term; it uses the stock
``JointPositionActionCfg`` with scale 0.25 and no prior.

This module imports ``isaaclab.envs`` and must only be imported after the
simulation app has started; the config (``actions_cfg.py``) refers to it by
name for that reason.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from .actions_cfg import CPGResidualJointPositionActionCfg


class CPGResidualJointPositionAction(JointPositionAction):
    """Apply policy residuals on top of a deterministic diagonal joint-position gait."""

    cfg: CPGResidualJointPositionActionCfg

    _LEG_JOINTS = {
        "lf": ("base_lf1", "lf1_lf2", "lf2_lf3"),
        "rf": ("base_rf1", "rf1_rf2", "rf2_rf3"),
        "lb": ("base_lb1", "lb1_lb2", "lb2_lb3"),
        "rb": ("base_rb1", "rb1_rb2", "rb2_rb3"),
    }
    _PHASE_OFFSETS = {
        "lf": 0.0,
        "rb": 0.0,
        "rf": math.pi,
        "lb": math.pi,
    }
    # Stand-pose FK sensitivity of [foot_x, foot_z] wrt [hip_pitch, knee_pitch],
    # used only by the ``linear_foot`` mode (not the shipped configuration).
    _FOOT_XZ_TO_HIP_KNEE = (
        (-14.23814723, 14.59717780),
        (-1.02320070, -24.52513760),
    )

    def __init__(self, cfg: CPGResidualJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._leg_joint_action_ids = {
            leg: [self._joint_names.index(name) for name in joint_names]
            for leg, joint_names in self._LEG_JOINTS.items()
        }

    def _gait_offsets(self) -> torch.Tensor:
        time = self._env.episode_length_buf.to(dtype=torch.float32, device=self.device) * self._env.step_dt
        offsets = torch.zeros_like(self._raw_actions)
        command = self._env.command_manager.get_command("base_velocity")
        command_scale = torch.clamp(torch.abs(command[:, 0]) / 0.10, min=0.5, max=1.5)

        for leg, ids in self._leg_joint_action_ids.items():
            phase = 2.0 * math.pi * self.cfg.frequency * time + self._PHASE_OFFSETS[leg]
            fore_aft = torch.cos(phase)
            if self.cfg.knee_lift_profile == "sin_positive":
                swing = torch.clamp(torch.sin(phase), min=0.0)
            elif self.cfg.knee_lift_profile == "cos_swing":
                swing = torch.clamp(-fore_aft, min=0.0)
            else:
                raise ValueError(f"Unsupported knee_lift_profile: {self.cfg.knee_lift_profile}")
            side_sign = 1.0 if leg in ("lf", "lb") else -1.0
            is_rear = leg in ("lb", "rb")
            hip_scale = self.cfg.rear_hip_amplitude_scale if is_rear else self.cfg.front_hip_amplitude_scale
            knee_scale = self.cfg.rear_knee_lift_scale if is_rear else self.cfg.front_knee_lift_scale

            offsets[:, ids[0]] = side_sign * self.cfg.abduction_amplitude * fore_aft
            if self.cfg.cpg_mode == "joint_sinusoid":
                offsets[:, ids[1]] = self.cfg.hip_amplitude * hip_scale * command_scale * fore_aft
                offsets[:, ids[2]] = -self.cfg.knee_lift_amplitude * knee_scale * command_scale * swing
            elif self.cfg.cpg_mode == "linear_foot":
                foot_x = -self.cfg.foot_stride_amplitude * hip_scale * command_scale * fore_aft
                foot_z = self.cfg.foot_lift_height * knee_scale * command_scale * swing
                offsets[:, ids[1]] = (
                    self._FOOT_XZ_TO_HIP_KNEE[0][0] * foot_x
                    + self._FOOT_XZ_TO_HIP_KNEE[0][1] * foot_z
                )
                offsets[:, ids[2]] = (
                    self._FOOT_XZ_TO_HIP_KNEE[1][0] * foot_x
                    + self._FOOT_XZ_TO_HIP_KNEE[1][1] * foot_z
                )
            else:
                raise ValueError(f"Unsupported cpg_mode: {self.cfg.cpg_mode}")
            if is_rear and self.cfg.rear_extra_swing_knee_lift_amplitude != 0.0:
                phase_aligned_swing = torch.clamp(-fore_aft, min=0.0)
                offsets[:, ids[2]] -= (
                    self.cfg.rear_extra_swing_knee_lift_amplitude * command_scale * phase_aligned_swing
                )
        return offsets

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset + self._gait_offsets()
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
