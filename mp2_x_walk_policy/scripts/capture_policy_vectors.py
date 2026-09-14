#!/usr/bin/env python3
"""Record a policy's observations and raw actions along a closed-loop rollout.

One environment on a Play task, one block per forward command (0.05, 0.10,
0.15 m/s), ``--steps-per-command`` steps each. The live policy and the
TorchScript export are run side by side and must agree to 1e-6. Output is a
fixture bundle (deploy/fixtures.py): ``<output>.npz`` + ``<output>.json`` with
``observations``, ``expected_raw_actions``, ``command_x``, ``step_indices``.

Two uses:
  - deployment parity: the bundle is what deploy/check_onnx.py replays through
    onnxruntime (see deploy/models/*/trajectory_fixtures.json);
  - open-loop replay: the 0.10 m/s block is the action stream that
    scripts/openloop_replay.sh plays back in simulation, and that
    analysis/stream_from_capture.py turns into joint targets for the robot.

    python -m mp2_x_walk_policy.scripts.capture_policy_vectors --task MP2-Walk-E2E-Play-v0 \\
        --checkpoint <model.pt> --export <dir>/policy.pt --label e2e --output runs/capture/e2e --steps-per-command 400 --viz none

Pass ``--output`` without an extension.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

from mp2_x_walk_policy import register_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from deploy.fixtures import sha256_file, write_bundle  # noqa: E402

register_tasks()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="MP2-Walk-E2E-Play-v0")
parser.add_argument("--checkpoint", action="append", required=True)
parser.add_argument("--export", action="append", required=True, help="TorchScript policy.pt of the same checkpoint")
parser.add_argument("--label", action="append", required=True)
parser.add_argument("--output", type=Path, required=True, help="bundle stem (no extension)")
parser.add_argument("--steps-per-command", type=int, default=50)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if not (len(args_cli.checkpoint) == len(args_cli.export) == len(args_cli.label)):
    raise ValueError("--checkpoint, --export, and --label counts must match")
if args_cli.output.suffix:
    raise ValueError("--output is a bundle stem: pass it without an extension")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402

COMMANDS = (0.05, 0.10, 0.15)


def main() -> None:
    checkpoints = [Path(value).expanduser().resolve() for value in args_cli.checkpoint]
    exports = [Path(value).expanduser().resolve() for value in args_cli.export]
    for path in checkpoints + exports:
        if not path.is_file():
            raise FileNotFoundError(path)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.10)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    observations, raw_actions, model_indices, command_values, step_indices = [], [], [], [], []
    jit_max_errors = {}

    with launch_simulation(env_cfg, args_cli):
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        for model_index, (checkpoint, export, label) in enumerate(zip(checkpoints, exports, args_cli.label)):
            runner.load(str(checkpoint), map_location=agent_cfg.device)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            jit = torch.jit.load(str(export), map_location="cpu").eval()
            model_max_error = 0.0

            for command_x in COMMANDS:
                obs, _ = env.reset()
                policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                for step in range(args_cli.steps_per_command):
                    command = env.unwrapped.command_manager.get_command("base_velocity")
                    command[:, 0] = command_x
                    command[:, 1:] = 0.0
                    obs = env.get_observations().to(env.unwrapped.device)

                    with torch.inference_mode():
                        actions = policy(obs)
                        jit_actions = jit(obs["policy"].detach().cpu())
                    error = torch.max(torch.abs(actions.detach().cpu() - jit_actions)).item()
                    model_max_error = max(model_max_error, error)

                    observations.append(obs["policy"][0].detach().cpu().numpy().astype(np.float32))
                    raw_actions.append(actions[0].detach().cpu().numpy().astype(np.float32))
                    model_indices.append(model_index)
                    command_values.append(command_x)
                    step_indices.append(step)

                    obs, _, dones, _ = env.step(actions.to(env.device))
                    policy.reset(dones)

            if model_max_error > 1.0e-6:
                raise RuntimeError(f"{label}: checkpoint vs TorchScript error {model_max_error}")
            jit_max_errors[label] = model_max_error

        env.close()

    manifest = write_bundle(
        args_cli.output,
        {
            "observations": np.stack(observations).astype(np.float32),
            "expected_raw_actions": np.stack(raw_actions).astype(np.float32),
            "model_indices": np.asarray(model_indices, dtype=np.int64),
            "command_x": np.asarray(command_values, dtype=np.float32),
            "step_indices": np.asarray(step_indices, dtype=np.int64),
        },
        metadata={
            "purpose": "closed-loop rollout in simulation: observations -> expected_raw_actions, per command block",
            "task": args_cli.task,
            "labels": args_cli.label,
            "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
            "export_sha256": [sha256_file(path) for path in exports],
            "checkpoint_torchscript_max_error": jit_max_errors,
            "steps_per_command": args_cli.steps_per_command,
            "commands_m_s": list(COMMANDS),
        },
    )
    print(json.dumps({"status": "pass", "manifest": str(manifest), "rows": len(observations),
                      "checkpoint_torchscript_max_error": jit_max_errors}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
