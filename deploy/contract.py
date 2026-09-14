"""The policy contract: what goes into the network and what comes out.

Reference implementation in numpy of the observation assembly and action
decoding both policies were trained with, plus a small runner that drives an
exported ONNX policy at 50 Hz. Everything here is exactly what the training
environment does (mp2_x_walk_policy/tasks/*, actions.py); the robot side
must reproduce it or the policy will not behave as it did in simulation.

Two contracts, told apart by the network's input width:

    CPG_RESIDUAL_47   phase 1   obs 47 = base 45 + [sin, cos] gait clock
                                target = STAND + cpg(tick, vx) + 0.12 * clip(a)
    END_TO_END_45     phase 2   obs 45
                                target = STAND + 0.25 * clip(a)

Observation layout (both; ROS joint order throughout):

    [0:3]    base angular velocity, body frame, rad/s
    [3:6]    gravity direction in the body frame, unit vector (z ~ -1 upright)
    [6:9]    velocity command (vx, 0, 0), m/s
    [9:21]   joint position minus STAND_POSE, rad
    [21:33]  joint velocity, rad/s
    [33:45]  previous CLIPPED action (zeros at reset)
    [45:47]  gait clock, CPG contract only

Hardware-specific and NOT implemented here (see docs/deployment.md): servo
tick <-> radian conversion and the knee convention, the IMU mounting frame,
and how joint velocity is estimated from position readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTROL_HZ = 50.0
ACTION_SIZE = 12

JOINT_ORDER = (
    "base_lf1", "lf1_lf2", "lf2_lf3",
    "base_rf1", "rf1_rf2", "rf2_rf3",
    "base_lb1", "lb1_lb2", "lb2_lb3",
    "base_rb1", "rb1_rb2", "rb2_rb3",
)
STAND_POSE = np.tile(np.asarray((0.0, 0.994, -1.767), dtype=np.float32), 4)

# Gait prior of the CPG contract (must match the training action term).
CPG_FREQUENCY_HZ = 2.0
CPG_PERIOD_TICKS = 25  # 50 Hz / 2 Hz
CPG_HIP_AMPLITUDE = np.float32(0.16)
CPG_KNEE_LIFT = np.float32(0.20)
CPG_LEG_PHASE_OFFSETS = np.asarray((0.0, math.pi, math.pi, 0.0), dtype=np.float32)  # lf, rf, lb, rb


@dataclass(frozen=True)
class Contract:
    name: str
    observation_size: int
    action_scale: np.float32
    uses_cpg_prior: bool


CPG_RESIDUAL_47 = Contract("cpg-residual-47", 47, np.float32(0.12), True)
END_TO_END_45 = Contract("end-to-end-45", 45, np.float32(0.25), False)
CONTRACTS = {c.observation_size: c for c in (CPG_RESIDUAL_47, END_TO_END_45)}


def contract_for_observation_size(size: int) -> Contract:
    try:
        return CONTRACTS[int(size)]
    except KeyError:
        raise ValueError(f"no contract for observation size {size}; known: {sorted(CONTRACTS)}") from None


# ------------------------------------------------------------------ gait clock


def gait_phase(tick: int) -> np.float32:
    """Phase of the 2 Hz clock at 50 Hz tick ``tick`` (tick 0 at reset)."""
    if tick < 0:
        raise ValueError("tick must be non-negative")
    return np.float32(2.0 * math.pi * (int(tick) % CPG_PERIOD_TICKS) / CPG_PERIOD_TICKS)


def gait_phase_clock(tick: int) -> np.ndarray:
    phase = gait_phase(tick)
    return np.asarray((np.sin(phase), np.cos(phase)), dtype=np.float32)


def command_scale(vx: float) -> np.float32:
    """The prior's amplitude scales with |vx| / 0.10, clipped to [0.5, 1.5]."""
    return np.float32(np.clip(np.float32(abs(float(vx)) / 0.10), np.float32(0.5), np.float32(1.5)))


def cpg_offsets(velocity_command: np.ndarray, tick: int) -> np.ndarray:
    """Joint offsets of the diagonal-trot prior, ROS order. Hip: cosine;
    knee: negative half-wave sine (flexes during swing); abduction: 0."""
    phase = gait_phase(tick)
    scale = command_scale(velocity_command[0])
    offsets = np.zeros(ACTION_SIZE, dtype=np.float32)
    for leg, phase_offset in enumerate(CPG_LEG_PHASE_OFFSETS):
        leg_phase = np.float32(phase + phase_offset)
        offsets[leg * 3 + 1] = np.float32(CPG_HIP_AMPLITUDE * scale * np.cos(leg_phase))
        offsets[leg * 3 + 2] = np.float32(-CPG_KNEE_LIFT * scale * np.maximum(np.sin(leg_phase), np.float32(0.0)))
    return offsets


# ------------------------------------------------------------ observation


def assemble_observation(
    contract: Contract,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    velocity_command: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    previous_action: np.ndarray,
    tick: int,
) -> np.ndarray:
    gravity = np.asarray(projected_gravity, dtype=np.float32)
    if not math.isclose(float(np.linalg.norm(gravity)), 1.0, abs_tol=1e-4):
        raise ValueError(f"projected_gravity must be a unit vector, norm {np.linalg.norm(gravity):.6f}")
    obs = np.empty(contract.observation_size, dtype=np.float32)
    obs[0:3] = base_ang_vel
    obs[3:6] = gravity
    obs[6:9] = velocity_command
    obs[9:21] = np.asarray(joint_pos, dtype=np.float32) - STAND_POSE
    obs[21:33] = joint_vel
    obs[33:45] = previous_action
    if contract.uses_cpg_prior:
        obs[45:47] = gait_phase_clock(tick)
    return obs


def imu_to_observation(
    accel_sensor: np.ndarray,
    gyro_sensor_dps: np.ndarray,
    sensor_to_body: np.ndarray,
    gyro_bias_body_dps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(projected_gravity, base_ang_vel) from raw IMU readings.

    ``sensor_to_body`` is the 3x3 rotation of your IMU mounting. Gravity is the
    negated, normalised body-frame accelerometer vector (upright: z ~ -1).
    The gyro bias is estimated while the robot stands still before a run and
    subtracted in deg/s before converting to rad/s.
    """
    acc_b = np.asarray(sensor_to_body, dtype=np.float32) @ np.asarray(accel_sensor, dtype=np.float32)
    gravity = (-acc_b / np.linalg.norm(acc_b)).astype(np.float32)
    gyro_b = np.asarray(sensor_to_body, dtype=np.float32) @ np.asarray(gyro_sensor_dps, dtype=np.float32)
    ang_vel = ((gyro_b - np.asarray(gyro_bias_body_dps, dtype=np.float32)) * np.float32(math.pi / 180.0)).astype(np.float32)
    return gravity, ang_vel


# ----------------------------------------------------------------- action


def decode_action(
    contract: Contract, raw_action: np.ndarray, velocity_command: np.ndarray, tick: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(clipped action, cpg offsets, joint targets in rad, ROS order)."""
    clipped = np.clip(np.asarray(raw_action, dtype=np.float32), np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    cpg = cpg_offsets(velocity_command, tick) if contract.uses_cpg_prior else np.zeros(ACTION_SIZE, dtype=np.float32)
    target = (STAND_POSE + cpg + contract.action_scale * clipped).astype(np.float32)
    if not np.all(np.isfinite(target)):
        raise ValueError("joint target contains non-finite values")
    return clipped, cpg, target


# ---------------------------------------------------------------- runner


class OnnxPolicy:
    """Run an exported policy.onnx with onnxruntime (CPU)."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.observation_size = int(inp.shape[-1])
        self.contract = contract_for_observation_size(self.observation_size)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return self.session.run(None, {self.input_name: obs})[0].reshape(-1).astype(np.float32)


class PolicyRunner:
    """Stateful 50 Hz stepper: tick counter and previous-action memory.

    Call ``reset()`` when the robot is standing still in STAND_POSE, then
    ``step(...)`` once per control period with the current sensor sample.
    """

    def __init__(self, policy: OnnxPolicy):
        self.policy = policy
        self.contract = policy.contract
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def step(self, base_ang_vel, projected_gravity, velocity_command, joint_pos, joint_vel) -> dict:
        obs = assemble_observation(
            self.contract, base_ang_vel, projected_gravity, velocity_command,
            joint_pos, joint_vel, self.previous_action, self.tick,
        )
        raw = self.policy(obs)
        clipped, cpg, target = decode_action(self.contract, raw, np.asarray(velocity_command, dtype=np.float32), self.tick)
        result = {"tick": self.tick, "observation": obs, "raw_action": raw, "clipped_action": clipped,
                  "cpg_offset": cpg, "joint_target": target}
        self.previous_action = clipped
        self.tick += 1
        return result
