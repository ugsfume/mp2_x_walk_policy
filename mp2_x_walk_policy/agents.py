"""rsl_rl runner configs: stock PPO for phase 1 and evaluation, anchored PPO for phase 2."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

EXPERIMENT_NAME = "mp2_x_walk"

# Shipped clone, relative to the repo root (see algorithms.AnchorPPO for how
# the path is resolved). Override both from the command line to anchor to a
# clone you distilled yourself:
#   agent.algorithm.anchor_path=/abs/path/bc_clone.pt agent.algorithm.anchor_sha256=<sha256>
DEFAULT_ANCHOR_PATH = "phase2/checkpoints/bc_clone.pt"
DEFAULT_ANCHOR_SHA256 = "92cb5922abd4e942a741a196c8c06f10a47e9e6151af9c94d29d03d6ad6cc74b"


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 and every evaluation id."""

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    clip_actions = 1.0
    experiment_name = EXPERIMENT_NAME
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnchoredPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus a loss-side MSE between the actor mean and a frozen clone's
    mean, evaluated on the clean ``anchor`` observation group.

    Fields must be declared here: rsl_rl forwards ``cfg["algorithm"]`` as
    ``__init__`` kwargs to the algorithm class.
    """

    class_name: str = "mp2_x_walk_policy.algorithms:AnchorPPO"
    anchor_path: str = DEFAULT_ANCHOR_PATH
    anchor_sha256: str = DEFAULT_ANCHOR_SHA256
    # beta0 sized so that, at 30% of the distance between the clone and an
    # unanchored policy, the anchor gradient on the actor equals the PPO
    # surrogate gradient: 0.7946 / 0.6615. Held constant for the whole run.
    anchor_beta0: float = 1.2011598444702776
    anchor_hold_iters: int = 1300
    anchor_decay_iters: int = 0
    anchor_beta_floor: float = 0.0
    anchor_obs_group: str = "anchor"


@configclass
class AnchoredPPORunnerCfg(PPORunnerCfg):
    """Phase 2: 1300 iterations warm-started from the clone, constant anchor,
    checkpoint every 25 iterations."""

    max_iterations = 1300
    save_interval = 25
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    algorithm = AnchoredPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
