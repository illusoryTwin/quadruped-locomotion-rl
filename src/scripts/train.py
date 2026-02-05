# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

import isaaclab_tasks  # noqa: F401
import envs  # noqa: F401  # registers go2_walk and go2_compliant_locomotion environments
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _export_deployment_metadata(env, env_cfg, agent_cfg, export_dir: str):
    """Export deployment metadata alongside the JIT policy.

    This creates YAML files with observation/action specifications
    that deployment code can use to properly configure the pipeline.
    """
    import yaml
    from datetime import datetime

    os.makedirs(export_dir, exist_ok=True)

    # Get the unwrapped Isaac Lab environment (go through RslRlVecEnvWrapper)
    unwrapped_env = env.unwrapped

    # Get observation/action dimensions from the wrapper (RslRlVecEnvWrapper exposes these)
    obs_dim = gym.spaces.flatdim(env.observation_space["policy"])
    action_dim = env.num_actions

    # Get robot joint information from the scene
    robot = unwrapped_env.scene.articulations.get("robot")
    if robot is not None:
        joint_names = list(robot.joint_names)
        num_joints = len(joint_names)

        # Get default joint positions from robot config
        default_joint_pos = {}
        if hasattr(robot.cfg, 'init_state') and hasattr(robot.cfg.init_state, 'joint_pos'):
            init_joint_pos = robot.cfg.init_state.joint_pos
            if isinstance(init_joint_pos, dict):
                default_joint_pos = {k: float(v) for k, v in init_joint_pos.items()}
    else:
        joint_names = [f"joint_{i}" for i in range(12)]
        num_joints = 12
        default_joint_pos = {}

    # Get control parameters
    sim_dt = float(unwrapped_env.physics_dt) if hasattr(unwrapped_env, 'physics_dt') else 0.005
    decimation = getattr(env_cfg, 'decimation', 4)
    control_dt = sim_dt * decimation

    # Get action scale from action manager
    action_scale_value = 0.5  # default
    if hasattr(unwrapped_env, 'action_manager') and hasattr(unwrapped_env.action_manager, '_terms'):
        if 'joint_pos' in unwrapped_env.action_manager._terms:
            action_term = unwrapped_env.action_manager._terms['joint_pos']
            if hasattr(action_term, '_scale'):
                action_scale_value = float(action_term._scale)

    # Check if normalizer is embedded in policy
    has_normalizer = agent_cfg.empirical_normalization if hasattr(agent_cfg, 'empirical_normalization') else False

    # Build manifest
    manifest = {
        "artifact_version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "task_name": args_cli.task,
        "model": {
            "policy_path": "policy.pt",
            "input_dim": int(obs_dim),
            "output_dim": int(action_dim),
            "has_normalizer": has_normalizer,  # True = normalizer embedded in JIT
        },
        "robot": {
            "joint_names": joint_names,
            "num_joints": num_joints,
        },
        "control": {
            "control_dt": float(control_dt),
            "control_frequency": float(1.0 / control_dt),
            "action_scale": float(action_scale_value),
            "default_joint_pos": default_joint_pos,
        },
        "observation": {
            "obs_dim": int(obs_dim),
        },
    }

    # Save manifest
    manifest_path = os.path.join(export_dir, "manifest.yaml")
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Exported deployment manifest to: {manifest_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) # , clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # if agent_cfg.class_name == "OnPolicyRunner":
    #     runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # elif agent_cfg.class_name == "DistillationRunner":
    #     runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # else:
    #     raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Helper function to export JIT policy
    def export_jit_policy(runner, env, env_cfg, agent_cfg, log_dir, suffix=""):
        """Export JIT policy with normalizer and metadata."""
        # Switch to eval mode for export (important for batch norm, dropout, etc.)
        runner.eval_mode()

        # Get the policy network
        try:
            policy_nn = runner.alg.policy  # RSL-RL 2.3+
        except AttributeError:
            policy_nn = runner.alg.actor_critic  # RSL-RL 2.2 and below

        # Get the normalizer - try different locations depending on RSL_RL version
        normalizer = None

        # RSL_RL 2.x: runner.obs_normalizer
        if hasattr(runner, 'obs_normalizer'):
            normalizer = runner.obs_normalizer
        # RSL_RL 3.x: actor_critic.actor_obs_normalizer
        elif hasattr(policy_nn, 'actor_obs_normalizer'):
            normalizer = policy_nn.actor_obs_normalizer
        # Fallback: check alg
        elif hasattr(runner, 'alg') and hasattr(runner.alg, 'obs_normalizer'):
            normalizer = runner.alg.obs_normalizer

        # Check if it's a real normalizer or just Identity/None
        if normalizer is not None:
            normalizer_type = type(normalizer).__name__
            if normalizer_type == "Identity":
                print(f"[INFO] Normalizer is Identity (normalization disabled)")
                normalizer = None
            else:
                print(f"[INFO] Using normalizer: {normalizer_type}")
                if hasattr(normalizer, '_mean'):
                    print(f"[INFO] Normalizer mean range: [{normalizer._mean.min():.3f}, {normalizer._mean.max():.3f}]")
                    print(f"[INFO] Normalizer std range: [{normalizer._std.min():.3f}, {normalizer._std.max():.3f}]")
        else:
            print("[INFO] No normalizer found - policy will be exported without normalization")

        # Export to JIT
        export_dir = os.path.join(log_dir, "exported")
        filename = f"policy{suffix}.pt" if suffix else "policy.pt"
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename=filename)
        print(f"[INFO] Exported JIT policy to: {os.path.join(export_dir, filename)}")

        # Also save metadata for deployment
        _export_deployment_metadata(env, env_cfg, agent_cfg, export_dir)

        # Switch back to train mode
        runner.train_mode()

    # Patch runner's save method to also export JIT
    original_save = runner.save
    def save_with_jit_export(it):
        original_save(it)
        # Export JIT alongside checkpoint (overwrites previous)
        export_jit_policy(runner, env, env_cfg, agent_cfg, log_dir)
        print(f"[INFO] JIT policy updated at iteration {it}")
    runner.save = save_with_jit_export

    # Run training with proper cleanup on interrupt
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        print(f"Training time before interrupt: {round(time.time() - start_time, 2)} seconds")
    finally:
        # Always export JIT on exit (even if interrupted)
        print("[INFO] Exporting final JIT policy...")
        export_jit_policy(runner, env, env_cfg, agent_cfg, log_dir)

    # close the simulator
    env.close()


def run():
    """Entry point for console script."""
    main()
    simulation_app.close()


if __name__ == "__main__":
    run()

