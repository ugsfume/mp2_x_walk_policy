"""Config for the CPG-residual action term. The implementing class is
referenced by name so that importing this config does not import the
simulator (see actions.py)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import CPGResidualJointPositionAction


@configclass
class CPGResidualJointPositionActionCfg(ActionTermCfg):
    """Joint-position residual action around a deterministic diagonal gait prior."""

    class_type: type["CPGResidualJointPositionAction"] | str = (
        "mp2_x_walk_policy.actions:CPGResidualJointPositionAction"
    )

    joint_names: list[str] = MISSING
    """Joint names or regex expressions the action maps to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the policy residual."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action."""

    preserve_order: bool = False
    """Whether to preserve the order of joint names in the action output."""

    use_default_offset: bool = True
    """Whether to use the articulation default joint pose as offset."""

    frequency: float = 2.0
    """Nominal gait frequency in Hz."""

    hip_amplitude: float = 0.16
    """Sinusoidal hip offset amplitude in radians."""

    front_hip_amplitude_scale: float = 1.0
    """Multiplier for front-leg hip offsets."""

    rear_hip_amplitude_scale: float = 1.0
    """Multiplier for rear-leg hip offsets."""

    knee_lift_amplitude: float = 0.20
    """Additional knee flexion during swing in radians."""

    front_knee_lift_scale: float = 1.0
    """Multiplier for front-leg swing knee lift."""

    rear_knee_lift_scale: float = 1.0
    """Multiplier for rear-leg swing knee lift."""

    rear_extra_swing_knee_lift_amplitude: float = 0.0
    """Additional rear-leg knee flexion during the phase-aligned swing half-cycle."""

    cpg_mode: str = "joint_sinusoid"
    """CPG target mode: ``joint_sinusoid`` or ``linear_foot``."""

    knee_lift_profile: str = "sin_positive"
    """Swing lift profile: ``sin_positive`` or ``cos_swing``."""

    foot_stride_amplitude: float = 0.010
    """Approximate sagittal foot stride amplitude in metres for ``linear_foot`` mode."""

    foot_lift_height: float = 0.010
    """Approximate swing foot lift in metres for ``linear_foot`` mode."""

    abduction_amplitude: float = 0.0
    """Optional ab/adduction amplitude in radians."""
