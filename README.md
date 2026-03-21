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

This repo uses Isaac Sim 5.1 and Isaac Lab 2.3.0.

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

### Training Environment (Isaac Sim + Isaac Lab)

Follow the official Isaac Lab installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

**Training dependencies:**

| Package | Version | Notes |
|---------|---------|-------|
| Isaac Sim | 5.1 | NVIDIA Omniverse simulator ([install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)) |
| Isaac Lab | 2.3.0 | RL framework (installed via Isaac Sim guide above) |
| RSL-RL | 3.0.1+ | PPO implementation (`pip install rsl-rl`) |
| unitree_rl_lab | - | IsaacLab extension for Unitree robots (sibling repo: `../unitree_robotics/unitree_rl_lab/`) |
| CUDA | 12.x | GPU acceleration |

After installing Isaac Sim and Isaac Lab, install this project:

```bash
cd quadruped-locomotion-rl
pip install -e .
```

### Deployment Environment (MuJoCo)

For deployment/testing see [quadruped-locomotion-deploy](https://github.com/illusoryTwin/quadruped-locomotion-deploy).

Alternatively, set up locally:

```bash
conda create -n go2_deploy python=3.10 -y
conda activate go2_deploy
pip install -r requirements.txt

export UNITREE_MUJOCO_PATH=~/unitree_robotics/unitree_mujoco
mkdir -p $(dirname $UNITREE_MUJOCO_PATH)
git clone https://github.com/unitreerobotics/unitree_mujoco $UNITREE_MUJOCO_PATH
git clone https://github.com/unitreerobotics/unitree_sdk2_python $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python
pip install -e $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python
```

## Training

Launch training with Isaac Lab:

```bash
# Activate Isaac Sim environment
conda activate isaacsim

# Train flat terrain walking
python3 scripts/train.py --task=go2_walk_flat --num_envs=4096 --max_iterations=5000
```

Visualize trained policy in Isaac Sim:

```bash
python3 scripts/play.py --task=go2_walk_flat --num_envs=16
```


## Deployment (MuJoCo)

### Terminal 1 - Launch MuJoCo simulator

```bash
cd ~/Workspace/Projects/quadruped-locomotion-rl
conda activate go2_deploy
python -m deploy.mujoco.launch_sim
```

### Terminal 2 - Run policy

```bash
cd ~/Workspace/Projects/quadruped-locomotion-rl
conda activate go2_deploy

# Run with default config
python -m deploy.mujoco.run_policy --config deploy/configs/go2_flat.yaml

# With velocity overrides
python -m deploy.mujoco.run_policy --config deploy/configs/go2_flat.yaml --vx 1.0 --wz 0.5

# With different checkpoint
python -m deploy.mujoco.run_policy --config deploy/configs/go2_flat.yaml \
    --checkpoint logs/rsl_rl/unitree_go2_walk/2025-12-29_15-43-52/model_1500.pt
```

