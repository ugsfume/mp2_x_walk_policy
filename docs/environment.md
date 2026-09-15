# Environment

What the shipped checkpoints and numbers were produced on. Other versions may
work; these are the ones that did.

| | |
|---|---|
| OS | Ubuntu 22.04, NVIDIA driver 595, RTX 5080 (16 GB) |
| Python | 3.12 (conda/micromamba env) |
| Isaac Sim | 6.0.1.0, pip wheels |
| Isaac Lab | `github.com/isaac-sim/IsaacLab` @ `ffff603eafc6b74264a5261cc0183d6a65390d78` (`v3.0.0-beta2.patch1`), unmodified |
| rsl_rl | `rsl-rl-lib==5.0.1` (installed by Isaac Lab's `--install rsl_rl`), unmodified |
| torch | 2.10.0+cu128 |
| extras | `onnxruntime` for `deploy/check_onnx.py`; `moviepy` (installed with Isaac Lab) for `--video` |

No patches to Isaac Lab or rsl_rl. The task package hooks in through Isaac
Lab's `--external_callback` and the anchored PPO through rsl_rl's
`class_name` plug point.

## Install

```bash
# 1. Python 3.12 environment
micromamba create -p ./isaaclab-env -c conda-forge python=3.12 pip
micromamba activate ./isaaclab-env

# 2. Isaac Sim (pip). Accept the EULA once; keep the variable set.
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES

# 3. Isaac Lab at the pinned commit, with rsl_rl
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && git checkout ffff603eafc6b74264a5261cc0183d6a65390d78
./isaaclab.sh --install rsl_rl
pip install onnxruntime

# 4. This repo
git clone <this repo> ~/mp2_x_walk_policy
cd ~/mp2_x_walk_policy && pip install -e .
export ISAACLAB_DIR=~/IsaacLab
scripts/smoke.sh
```

Isaac Sim's own metadata asks for a newer torch than Isaac Lab pins; follow
Isaac Lab. First launch of the simulator compiles shaders and takes a few
minutes; later launches take ~20 s.

## Conventions the scripts assume

- The Isaac Lab python environment is active, and `ISAACLAB_DIR` points at
  the checkout. `scripts/_env.sh` sets everything else (`PYTHONPATH`,
  `OMNI_KIT_ACCEPT_EULA`, the log root).
- Training goes through `./isaaclab.sh train ... --external_callback
  mp2_x_walk_policy.register_tasks`; checkpoints land in
  `$ISAACLAB_DIR/logs/rsl_rl/mp2_x_walk/<timestamp>_<run>/`, with the fully
  resolved config in `params/{env,agent}.yaml`.
- `--viz none` is headless in this Isaac Lab version (`--headless` is
  deprecated). Rendering video (`scripts/render.sh`) needs an RTX-capable
  GPU and no `--viz none`.
- Every script writes its console output next to its results under `runs/`
  (git-ignored). The Isaac Lab CLI can exit 0 after a configuration error,
  so the scripts check the log for "Learning iteration" / "Parsing
  configuration from" rather than trusting the exit code.
- Command-line overrides of the config work as Hydra-style tokens after the
  fixed arguments, e.g. `agent.algorithm.anchor_beta0=0.6` or
  `env.scene.num_envs=256`, and are recorded in the run's `params/`. This
  needs `register_tasks()` to return `None`; returning an empty list would
  silently drop them.

## Reproducibility

Training is deterministic given the seed, the env count and the start
checkpoint on one machine: two launches of the same task at seed 42 for
10 iterations give bit-identical weights (`tools/compare_checkpoints.py`).
Across GPUs or driver versions expect the same distribution of outcomes
over seeds, not the same bytes. Evaluation (`scripts/eval.sh`) is not
seeded; repeat it when a verdict lands near a bar.

What the seed does not pin is the outcome. The seed decides which gait
variant PPO settles into, and the variants differ: across the phase-2
seeds we have trained (the README table plus a sweep from a freshly
distilled clone) speed spans roughly 0.07–0.085 m/s, f0 1.73–2.04 Hz, and
the left/right balance of the gait runs from symmetric to visibly
one-sided. The clock keeps phase 1 much tighter (all seeds 2.04 Hz,
0.098–0.108 m/s). Two further draws sit on top: clone collection is
unseeded, so every `phase2/run_distill.sh` gives a different clone; and
every evaluation episode is a different start — an end-to-end policy can
settle into a left- or right-heavy limit cycle per episode, so one
rendered clip is one draw, not the policy. The shipped phase-2 checkpoint
is seed 42 chosen out of four. Reproducing the *result* therefore means
the same sweep over seeds and the same selection (`phase2/run_select.sh`,
median-of-3 evaluations), not one run; a single seed that walks worse
than the shipped checkpoint is expected, not a failed reproduction.
