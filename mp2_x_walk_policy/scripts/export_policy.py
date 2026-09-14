"""Export a checkpoint to TorchScript (policy.pt) and ONNX (policy.onnx + .data).

    python -m mp2_x_walk_policy.scripts.export_policy --task MP2-Walk-E2E-Play-v0 \
        --checkpoint phase2/checkpoints/e2e_anchored_s42_1050.pt --output_dir runs/export --viz none

Loads the checkpoint through rsl_rl on the given Play task (only its
observation width matters), exports both formats, then runs a few steps and
checks the TorchScript output against the live policy. The ONNX file is what
deploy/contract.py drives; check it with deploy/check_onnx.py.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import sys

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

from mp2_x_walk_policy import register_tasks

register_tasks()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="MP2-Walk-E2E-Play-v0")
parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument("--output_dir", default=None, help="Directory for exported policy files.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=64)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import launch_simulation, resolve_task_config


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else checkpoint.parent / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    installed_version = metadata.version("rsl-rl-lib")
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.log_dir = str(checkpoint.parent)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), map_location=agent_cfg.device)
        runner.export_policy_to_jit(path=str(output_dir), filename="policy.pt")
        runner.export_policy_to_onnx(path=str(output_dir), filename="policy.onnx")

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations().to(env.unwrapped.device)
        reward_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
        action_min = float("inf")
        action_max = float("-inf")
        last_actions = None
        with torch.inference_mode():
            for _ in range(args_cli.eval_steps):
                last_actions = policy(obs)
                action_min = min(action_min, float(last_actions.min()))
                action_max = max(action_max, float(last_actions.max()))
                obs, rewards, dones, _ = env.step(last_actions.to(env.device))
                obs = obs.to(env.unwrapped.device)
                reward_sum += rewards.to(env.unwrapped.device)
                policy.reset(dones)

        with torch.inference_mode():
            actions = policy(obs)

        jit_path = output_dir / "policy.pt"
        onnx_path = output_dir / "policy.onnx"
        jit_policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
        with torch.inference_mode():
            jit_actions = jit_policy(obs["policy"].detach().cpu())

        print(f"CHECKPOINT {checkpoint}")
        print(f"EXPORT_DIR {output_dir}")
        print(f"JIT_EXPORT {jit_path} {jit_path.stat().st_size} bytes")
        print(f"ONNX_EXPORT {onnx_path} {onnx_path.stat().st_size} bytes")
        print(f"OBS_POLICY_SHAPE {tuple(obs['policy'].shape)}")
        print(f"ACTIONS_SHAPE {tuple(actions.shape)}")
        print(f"JIT_ACTIONS_SHAPE {tuple(jit_actions.shape)}")
        print(f"EVAL_STEPS {args_cli.eval_steps}")
        print(f"EVAL_REWARD_MEAN_PER_STEP {float((reward_sum / max(args_cli.eval_steps, 1)).mean()):.6f}")
        print(f"EVAL_ACTION_MIN_MAX {action_min:.6f},{action_max:.6f}")
        obs_terms = "base_ang_vel,projected_gravity,velocity_commands,joint_pos,joint_vel,actions"
        if obs["policy"].shape[-1] == 47:
            obs_terms += ",gait_phase_clock"
        print(f"OBS_TERMS {obs_terms}")
        print("ACTION_ORDER base_lf1,lf1_lf2,lf2_lf3,base_rf1,rf1_rf2,rf2_rf3,base_lb1,lb1_lb2,lb2_lb3,base_rb1,rb1_rb2,rb2_rb3")
        print("ROS_TO_ISAAC_INDEX 0,4,8,1,5,9,3,7,11,2,6,10")
        print("ISAAC_TO_ROS_INDEX 0,3,9,6,1,4,10,7,2,5,11,8")

        if obs["policy"].shape[-1] not in (45, 47):
            raise RuntimeError(f"Expected 45 or 47 policy observations, got {obs['policy'].shape}")
        if actions.shape[-1] != 12:
            raise RuntimeError(f"Expected 12 actions, got {actions.shape}")
        if jit_actions.shape[-1] != 12:
            raise RuntimeError(f"Expected 12 JIT actions, got {jit_actions.shape}")
        if version.parse(installed_version) < version.parse("5.0.1"):
            raise RuntimeError(f"Unexpected rsl-rl-lib version: {installed_version}")

        env.close()
        print("EXPORT_INTERFACE_VERIFIED")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
