# Quadruped Locomotion RL (compliance implementation explanation)

Compliance implementation (in task-space) is in the `feat/compliance_task_space` branch.

**Task:** compliant locomotion for Unitree Go2.

## Compliance Architecture

### Overview

The compliance system models deformations as second-order mass-spring-damper (MSD) system.
External forces applied to the robot's bodies, and the resulting deformations in Cartesian space are computed by integrating the MSD dynamics. The policy is rewarded for tracking these deformed states.

- `ComplianceManager` (in `src/compliance/compliance_manager.py`) implements the core compliance logic: it reads external forces from monitored bodies and updates the MSD model to produce deformation vectors. Compliance parameters are defined in `compliance_manager_cfg.py` — compliant bodies, per-body Cartesian stiffness scales, timestep (`dt`), base stiffness, and base inertia.


- The task configuration is in `soft_walk_env_cfg.py` (`UnitreeGo2WalkSoftEnvCfg`) for a walking task.

The compliant stance task (stable stance under external forces) is in `compliant_stance_env_cfg.py` (`UnitreeGo2StanceEnvCfg`).

### Deformations

**Deformations are calculated at each step by solving the dynamic equation** 
`m*q'' + d*q' + k*q = tau`
(implementation - in `/src/compliance/utils`)

To avoid exploding values, deformations are clamped to `[-max_deformation, max_deformation]` defined in the ComplianceManagerCfg config. 


### Commands 

**Stiffness values are generated as commands** — `StiffnessCommand` (in `src/modules/commands/stiffness_command.py`) samples a base stiffness `kp`.

### Observations

The policy (actor) has ***stiffness_commands*** **in observations**. Critic's observations have deformations.

### Rewards

- `track_compliant_pos` — exponential reward for tracking compliant base position: `exp(-||x_sim - x_ref||^2 / std^2)` (function: `track_compliant_base_pos_exp`)
- `track_compliant_vel` — L2 penalty for deviating from compliant velocity reference.

(Error is calculated in Cartesian space.)



### Events

A special kind of event is created for compliance learning:

`compliance_push` — step-based sinusoidal forces, applied every step (mode="step"). They act as continuous external perturbations that the MSD system responds to, producing the deformation targets the policy must track.


## Launch

Use the following command to launch compliant policy training (supposed you have installed Isaac Sim 5.1):

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

```
python3 scripts/train.py --task=go2_compliant_stance --num_envs=4096 --max_iterations=5000 --headless
```

To visualize in IsaacSim:

```
python3 scripts/play.py --task=go2_compliant_stance --num_envs=4
```

*One can use the following guiude to install the relevant version of IsaacSim: 
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html



-------------------------------------------------------------------------------------



# Quadruped Locomotion RL

This repository includes reinforcement learning locomotion experiments for the Unitree Go2 robot and the deployment infrastructure required to test them in Mujoco simulator and transfer them to real hardware.

Currently supported tasks:

| Task ID | Environment Config | Description |
|---------|-------------------|-------------|
| `go2_walk_flat` | `flat_walk_env_cfg.py` | Walking on flat terrain |
| `go2_soft_walk` | `compliant_walk_env_cfg.py` | Soft compliant walking |
| `go2_compliant_stance` | `compliant_stance_env_cfg.py` | Compliant stance under external forces |
| `go2_compliant_stance_fixed_stiffness` | `compliant_stance_fixed_stiffness_env_cfg.py` | Compliant stance with fixed stiffness |
| `go2_default_stance` | `stance_env_cfg.py` | Default standing pose |

This repo uses Isaac Sim `5.1.0.0` and Isaac Lab `v2.3.1`.

## Project Structure

```
quadruped-locomotion-rl/
├── scripts/                    # Training & evaluation scripts
│   ├── train.py                # Main training script
│   ├── play.py                 # Policy visualization in Isaac Sim
│   └── cli_args.py             # CLI argument helpers
├── src/                        # Main source package
│   ├── algorithms/             # RL algorithm configs
│   │   └── rsl_rl_ppo_cfg.py
│   ├── compliance/             # Compliance system
│   │   ├── compliant_manager.py
│   │   ├── compliance_manager_cfg.py
│   │   └── utils/              # MSD dynamics, frame transforms
│   └── modules/
│       ├── tasks/              # Environment configurations
│       │   ├── flat_walk_env_cfg.py
│       │   ├── compliant_walk_env_cfg.py
│       │   ├── soft_walk_env_cfg.py
│       │   ├── compliant_stance_env_cfg.py
│       │   ├── compliant_stance_fixed_stiffness_env_cfg.py
│       │   └── stance_env_cfg.py
│       ├── commands/           # Command generators
│       │   └── stiffness_command.py
│       ├── terrains.py
│       ├── rewards.py
│       ├── events.py
│       └── curriculums.py
├── deploy/                     # Deployment code
│   ├── configs/                # Robot/task configs
│   ├── common/                 # Shared utilities
│   └── mujoco/                 # MuJoCo deployment
└── logs/                       # Training outputs
```

## Installation

### Training

Requires an NVIDIA GPU. Tested with:

| Package | Version |
|---------|---------|
| Isaac Sim | `5.1.0.0` |
| Isaac Lab | `v2.3.1` |
| rsl-rl-lib | `3.0.1` |
| Python | `3.11` |
| CUDA drivers | `>= 525` |

Run the setup script — it handles everything (conda env, Isaac Sim, Isaac Lab, RSL-RL, unitree_rl_lab, this project):

```bash
bash setup_train.sh
conda activate env_isaaclab
```

### Deployment (MuJoCo simulation)

Fully Dockerized — no manual dependency setup needed. All dependencies (MuJoCo, PyTorch, Unitree SDK) are installed automatically inside the container.

```bash
cd deploy/docker
docker compose build                                     # first time only (~15 min)
docker compose run --rm quadruped-policy go2_soft_walk   # run a task
```

Source code is bind-mounted, so code changes take effect immediately without rebuilding.

## Training

```bash
conda activate env_isaaclab

# Compliant stance
python scripts/train.py --task=go2_compliant_stance --num_envs=4096 --max_iterations=5000 --headless

# Soft walking
python scripts/train.py --task=go2_soft_walk --num_envs=4096 --max_iterations=5000 --headless
```

Visualize a trained policy in Isaac Sim:

```bash
python scripts/play.py --task=go2_compliant_stance --num_envs=4
```


## Deployment (MuJoCo simulation)

```bash
cd deploy/docker

# Build image (first time only)
docker compose build

# Run a task (auto-resolves latest policy checkpoint)
docker compose run --rm quadruped-policy go2_soft_walk
docker compose run --rm quadruped-policy go2_compliant_stance

# Override duration or starting stiffness
DURATION=60 docker compose run --rm quadruped-policy go2_soft_walk
CMD_ARGS="stiffness_commands=500.0" docker compose run --rm quadruped-policy go2_soft_walk

# Change stiffness at runtime (in a second terminal)
python deploy/stiffness_client.py --stiffness 500.0
```


## Real Hardware

```bash
python deploy/deploy.py \
    --policy logs/rsl_rl/unitree_go2_walk_soft/2026-05-06_00-20-12/exported/policy.pt \
    --config deploy/configs/soft_walk.yaml \
    --interface eth0 --domain 0
```                               