# quadruped-locomotion-deploy

Deployment and testing for trained quadruped locomotion policies on the Unitree Go2 robot. Supports MuJoCo simulation and real hardware via DDS.

Lightweight runtime — does **not** require Isaac Sim or Isaac Lab.

## Quick Start (Docker)

```bash
cd docker
docker compose build
docker compose run quadruped-deploy
```

### Inside the container

**Terminal 1** — MuJoCo simulator:

```bash
cd /workspace/quadruped-locomotion-rl
python -m deploy.mujoco.launch_sim
```

**Terminal 2** — Policy (open a second shell into the running container):

```bash
docker exec -it quadruped-deploy bash
cd /workspace/quadruped-locomotion-deploy
python run.py
```

### One-shot policy runner

```bash
# Compliant stance (default)
docker compose run quadruped-policy

# Custom task
DEPLOY_CONFIG=soft_pos_xy_tracking \
POLICY_PATH=/workspace/quadruped-locomotion-rl/logs/rsl_rl/.../exported/policy.pt \
CMD_ARGS="position_commands=0.5,0.0" \
docker compose run quadruped-policy
```

Source code is bind-mounted — changes apply immediately without rebuilds.

Required sibling repos (relative to this repo's parent directory):
- `../quadruped-locomotion-rl/`
- `../unitree_robotics/unitree_mujoco/`
- `../unitree_robotics/unitree_sdk2_python/`

## Project Structure

```
quadruped-locomotion-deploy/
├── core/
│   ├── policy_controller.py   # Policy inference + observation history
│   ├── command_manager.py     # Velocity command interface
│   └── config.py              # YAML config loader
├── runners/
│   ├── dds_runner.py          # Main control loop
│   └── dds_interface.py       # Unitree SDK2 DDS communication
├── utils/
│   └── joint_mapper.py        # URDF <-> policy joint ordering
├── policies/                  # Pre-trained policy weights + configs
├── docker/
│   ├── Dockerfile             # Deploy container (CUDA + MuJoCo + SDK2)
│   ├── docker-compose.yaml    # Container orchestration
│   ├── entrypoint.sh          # Interactive entrypoint
│   └── entrypoint-policy.sh   # One-shot policy runner entrypoint
└── run.py                     # Main entry point
```

## Training

Policies are trained in [quadruped-locomotion-rl](https://github.com/illusoryTwin/quadruped-locomotion-rl) and exported as JIT models with config YAML files. Copy exported artifacts to `policies/` for deployment.
