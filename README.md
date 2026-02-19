# Quadruped Locomotion RL (compliance implementation explanation)

Compliance implementation (in task-space) is in the `feat/compliance_task_space` branch.

**Task:** compliant locomotion for Unitree Go2.

## Compliance Architecture

### Overview

The compliance system models deformations as second-order mass-spring-damper (MSD) system.
External forces applied to the robot's bodies, and the resulting deformations in Cartesian space are computed by integrating the MSD dynamics. The policy is rewarded for tracking these deformed states.

`CompliantRLEnv` (in `src/modules/envs/compliant_rl_env.py`) extends `ManagerBasedRLEnv` — it overrides `step()` to call `_compute_compliance_targets()` after physics simulation to calculate new state under compliance.


`ComplianceManager` (in `src/compliance/compliance_manager.py`) implements the core compliance logic: it reads external forces from monitored bodies and updates the MSD model to produce deformation vectors. Compliance parameters are defined in `compliance_manager_cfg.py` — monitored bodies, per-joint stiffness scales, timestep (`dt`), base stiffness, and base inertia.


The task configuration is in `flat_walk_soft_env_cfg.py` (`UnitreeGo2WalkSoftEnvCfg`)

### Deformations

**Deformations are calculated at each step by solving the dynamic equation** 
`m*q'' + d*q' + k*q = tau`
(implementation - in `/src/compliance/utils`)

To avoid exploding values, deformations are clamped to `[-max_deformation, max_deformation]` defined in the ComplianceManagerCfg config. 


### Commands 

**Stiffness values are generated as commands** — `StiffnessCommand` (in `src/   modules/commands/stiffness_command.py`) samples a base stiffness `kp` from a range based on curriculum:                       

| Iteration | kp range    |
|-----------|-------------|
| 0         | 3000 – 4000 |
| 1000      | 1000 – 3000 |
| 2000      | 200 – 1000  |
| 3000      | 50 – 200    |                          

### Observations

**in observations** we have stiffness commands (in actor). Critic's observations have deformations.

### Rewards

- `track_compliant_targets` reward for tracking deformed states  

It calculates the error in Cartesian space:

$$reward = \exp\!\left(-\frac{\|\Delta p - x_{def}\|^2}{\sigma^2}\right)$$

where `delta_p = J @ q_error`, `q_error = q_actual - q_target`k  and `x_def` is deformation between current joint-space state and the deformed joint-space state. deformed joint space state is calculated via J @ x_deformed.

### Events

A special kind of event is created for compliance learning:

`compliance_push` — step-based sinusoidal forces, applied every step (mode="step"). They act as continuous external perturbations that the MSD system responds to, producing the deformation targets the policy must track.


## Launch

Use the following command to launch compliant policy training (supposed you have installed Isaac Sim 5.1):

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

```
python3 src/scripts/train.py --task=go2_compliant_locomotion --num_envs=4096 --max_iterations=5000 --headless
```

To visualize in isaac sim:

```
python3 src/scripts/play.py --task=go2_compliant_locomotion --num_envs=4
```

*One can use the following guiude to install the relevant version of IsaacSim: 
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html



-------------------------------------------------------------------------------------



# Quadruped Locomotion RL

This repository includes reinforcement learning locomotion experiments for the Unitree Go2 robot and the deployment infrastructure required to test them in Mujoco simulator and transfer them to real hardware.

Currently supported tasks include:

- walking on flat terrain
- walking on rough terrain
- climbing upstairs 

These tasks serve as a basis for future experiments, including:

- Mixture-of-Experts–based policy architectures
- Soft compliant policies 

This repo uses Isaac Sim 5.1 and Isaac Lab 2.3.0.

## Project Structure

```
quadruped-locomotion-rl/
├── scripts/                # Training & evaluation scripts
│   ├── train.py            # Main training script
│   └── play.py             # Policy visualization in Isaac Sim
├── tasks/                  # Environment configurations
│   ├── flat_walk_env_cfg.py
│   ├── rough_walk_env_cfg.py
│   └── stairs_climbing_env_cfg.py
├── modules/                # Custom components
│   ├── terrains.py
│   ├── rewards.py
│   └── curriculums.py
├── deploy/                 # Deployment code
│   ├── configs/            # Robot/task configs
│   ├── common/             # Shared utilities
│   └── mujoco/             # MuJoCo deployment
└── logs/                   # Training outputs
```

## Installation

### Training Environment (Isaac Sim + Isaac Lab)

Follow the official installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

**Dependencies:**
| Package | Version | Notes |
|---------|---------|-------|
| Isaac Sim | 5.1 | Base simulator |
| Isaac Lab | 2.3.0 | RL framework (installed via guide above) |
| RSL-RL | 3.0.1+ | `pip install rsl-rl` |
| isaaclab_tasks | - | Comes with Isaac Lab |
| isaaclab_assets | - | Comes with Isaac Lab |

### Deployment Environment (MuJoCo)

```bash
# Create conda environment
conda create -n go2_deploy python=3.10 -y
conda activate go2_deploy

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install mujoco pygame numpy pyyaml

# Set path for external repos (customize to your preference)
export UNITREE_MUJOCO_PATH=~/unitree_robotics/unitree_mujoco

# Clone and install external repos
mkdir -p $(dirname $UNITREE_MUJOCO_PATH)
git clone https://github.com/unitreerobotics/unitree_mujoco $UNITREE_MUJOCO_PATH
git clone https://github.com/unitreerobotics/unitree_sdk2_python $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python
pip install -e $(dirname $UNITREE_MUJOCO_PATH)/unitree_sdk2_python

# Add to shell config for persistence
echo "export UNITREE_MUJOCO_PATH=$UNITREE_MUJOCO_PATH" >> ~/.bashrc
```

## Training

Launch training with Isaac Lab:

```bash
# Activate Isaac Sim environment
conda activate isaacsim

# Train flat terrain walking
python3 src/scripts/train.py --task=go2_walk_flat --num_envs=4096 --max_iterations=5000
```

Visualize trained policy in Isaac Sim:

```bash
python3 src/scripts/play.py --task=go2_walk_flat --num_envs=16
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

