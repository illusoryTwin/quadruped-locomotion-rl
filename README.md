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
../../IsaacLab/isaaclab.sh -p scripts/train.py --task=go2_walk_flat --num_envs=4096 --max_iterations=5000

# Train rough terrain walking
../../IsaacLab/isaaclab.sh -p scripts/train.py --task=go2_walk_rough --num_envs=4096 --max_iterations=5000

# Train stairs climbing
../../IsaacLab/isaaclab.sh -p scripts/train.py --task=go2_stairs_climbing --num_envs=4096 --max_iterations=5000
```

Visualize trained policy in Isaac Sim:

```bash
../../IsaacLab/isaaclab.sh -p scripts/play.py --task=go2_walk_flat --num_envs=16
```
## Compliant Policies

To launch a policy which is supposed to be compliant, aka soft, use the following command. 

```bash
# Activate Isaac Sim environment
conda activate isaacsim

# Train flat terrain walking
../../IsaacLab/isaaclab.sh -p scripts/train.py --task=go2_compliant_locomotion --num_envs=4096 --max_iterations=5000
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

