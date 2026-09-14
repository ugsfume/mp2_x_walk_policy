"""Gym ids.

Training ids bind the agent config that trains them; every evaluation id
binds the stock PPO runner config, so evaluation, export and capture never
construct the anchored algorithm (and never need the clone).
"""

import gymnasium as gym

_ENV = "isaaclab.envs:ManagerBasedRLEnv"
_TASKS = "mp2_x_walk_policy.tasks"
_AGENTS = "mp2_x_walk_policy.agents"

_IDS = (
    # phase 1: CPG-residual
    ("MP2-Walk-CpgResidual-v0", "cpg_residual:CpgResidualEnvCfg", "PPORunnerCfg"),
    ("MP2-Walk-CpgResidual-Play-v0", "cpg_residual:CpgResidualPlayEnvCfg", "PPORunnerCfg"),
    ("MP2-Walk-CpgResidual-PlayRand-v0", "cpg_residual:CpgResidualPlayRandEnvCfg", "PPORunnerCfg"),
    # phase 2: end-to-end, anchored to the clone
    ("MP2-Walk-E2E-v0", "e2e_anchored:E2EAnchoredEnvCfg", "AnchoredPPORunnerCfg"),
    ("MP2-Walk-E2E-Play-v0", "e2e_anchored:E2EAnchoredPlayEnvCfg", "PPORunnerCfg"),
    ("MP2-Walk-E2E-PlayRand-v0", "e2e_anchored:E2EAnchoredPlayRandEnvCfg", "PPORunnerCfg"),
    ("MP2-Walk-E2E-Noisy-v0", "e2e_anchored:E2EAnchoredNoisyEnvCfg", "PPORunnerCfg"),
)

for _id, _env_cfg, _agent_cfg in _IDS:
    gym.register(
        id=_id,
        entry_point=_ENV,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_TASKS}.{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:{_agent_cfg}",
        },
    )

TASK_IDS = [entry[0] for entry in _IDS]
