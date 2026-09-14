#!/usr/bin/env python3
"""Collect (observation, action) pairs from the phase-1 teacher for the clone.

The teacher (phase 1, TorchScript export, deterministic mean) is rolled
closed-loop on the clean randomised-plant evaluation env at a few pinned
forward commands. Per step:

    obs45  = the 47-dim policy observation without the two gait-clock dims,
             with the previous-action slice [33:45] rewritten to the STUDENT's
             own previous action (the student's contract encodes actions
             differently from the teacher's residuals, and it must train on
             what it will be fed at deployment; zeros at reset)
    act45  = clip((applied joint target - stand pose) / 0.25, -1, 1)

The joint target is read from the action term after the step, so the CPG
phase, the residual scale and the joint order are captured exactly as
executed. The fraction of clipped labels is recorded per joint (expect a few
percent at commands <= 0.10 m/s).

    python -m distill.bc_collect --teacher runs/export/policy.pt --output runs/distill/pairs --viz none

~1 minute on a GPU for the default 512 envs x 400 steps x 3 commands.
Unseeded: two collections give statistically equivalent, not identical, data.
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="MP2-Walk-CpgResidual-PlayRand-v0")
parser.add_argument("--teacher", required=True, help="TorchScript policy.pt of the phase-1 checkpoint")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--commands", default="0.05,0.075,0.10")
parser.add_argument("--output", required=True, help="output stem; writes <output>.npz and <output>.npz.meta.json")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--action-noise", type=float, default=0.0,
                    help="execute teacher + N(0, sigma) in residual units but label with the clean "
                         "teacher target (DART-style state coverage). 0 = plain behaviour cloning.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402
from importlib import metadata  # noqa: E402
from mp2_x_walk_policy import register_tasks  # noqa: E402

register_tasks()

E2E_SCALE = 0.25
RESIDUAL_SCALE = 0.12


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    commands = [float(c) for c in args_cli.commands.split(",")]

    obs_list, act_list = [], []
    clip_hits = np.zeros(12, dtype=np.int64)
    total_rows = 0

    with launch_simulation(env_cfg, args_cli):
        env = RslRlVecEnvWrapper(
            gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions
        )
        device = env.unwrapped.device
        teacher = torch.jit.load(args_cli.teacher, map_location=device).eval()
        term = env.unwrapped.action_manager._terms["joint_pos"]
        robot = env.unwrapped.scene["robot"]
        joint_ids = term._joint_ids
        if joint_ids is None or isinstance(joint_ids, slice):
            default_pose = robot.data.default_joint_pos
        else:
            default_pose = robot.data.default_joint_pos[:, joint_ids]

        for command_x in commands:
            obs, _ = env.reset()
            prev_a45 = torch.zeros(env.num_envs, 12, device=device)
            for _step in range(args_cli.steps):
                command = env.unwrapped.command_manager.get_command("base_velocity")
                command[:, 0] = command_x
                command[:, 1:] = 0.0
                obs = env.get_observations().to(device)
                obs_t = obs["policy"].clone()
                with torch.inference_mode():
                    a47 = teacher(obs["policy"])
                a47_exec = a47
                if args_cli.action_noise > 0:
                    a47_exec = a47 + torch.randn_like(a47) * args_cli.action_noise
                env.step(a47_exec)
                targets = term.processed_actions.detach()
                if args_cli.action_noise > 0:
                    # Label with the clean teacher target: remove the executed
                    # noise's contribution (clip-aware, residual scale 0.12).
                    clean_res = torch.clamp(a47, -1.0, 1.0)
                    exec_res = torch.clamp(a47_exec, -1.0, 1.0)
                    targets = targets + RESIDUAL_SCALE * (clean_res - exec_res)
                a45 = (targets - default_pose) / E2E_SCALE
                clipped = torch.clamp(a45, -1.0, 1.0)
                clip_hits += (a45.abs() > 1.0).sum(dim=0).cpu().numpy().astype(np.int64)
                obs_t[:, 33:45] = prev_a45
                obs_list.append(obs_t[:, :45].detach().cpu().numpy().astype(np.float16))
                act_list.append(clipped.detach().cpu().numpy().astype(np.float16))
                total_rows += obs_t.shape[0]
                prev_a45 = clipped

        env.close()

    obs45 = np.concatenate(obs_list, axis=0)
    act45 = np.concatenate(act_list, axis=0)
    out = Path(args_cli.output)
    if out.suffix == ".npz":
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), obs45=obs45, act45=act45)
    meta = {
        "task": args_cli.task,
        "teacher": str(Path(args_cli.teacher).resolve()),
        "num_envs": args_cli.num_envs,
        "steps_per_command": args_cli.steps,
        "commands": commands,
        "rows": int(total_rows),
        "action_noise": args_cli.action_noise,
        "clip_fraction_per_joint": (clip_hits / max(total_rows, 1)).round(5).tolist(),
        "clip_fraction_overall": float(clip_hits.sum() / max(total_rows * 12, 1)),
        "dtype": "float16 storage; cast to float32 in training",
    }
    Path(str(out.with_suffix(".npz")) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("BC_COLLECT_META " + json.dumps(meta))
    print("BC_COLLECT: PASS")


if __name__ == "__main__":
    main()
    simulation_app.close()
