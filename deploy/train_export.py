# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL and export policy after training."""

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
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL and export policy.")
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
parser.add_argument(
    "--no-export", action="store_true", default=False, help="Skip exporting policy after training."
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

import copy
import gymnasium as gym
import logging
import os
import re
import time
import torch
import numpy as np
from datetime import datetime
from collections.abc import Sequence
from typing import Any

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import tasks  # noqa: F401  # registers go2_walk environment
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def resolve_matching_names_values(
    data: dict[str, Any], list_of_strings: Sequence[str], preserve_order: bool = False
) -> tuple[list[int], list[str], list[Any]]:
    """Resolve matching names and values from a dictionary using regex patterns."""
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")

    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(data))]

    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if re.fullmatch(re_key, potential_match_string):
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                keys_match_found[key_index].append(potential_match_string)

    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1

        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]

        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder

    return index_list, names_list, values_list


# Observation name remapping for config
obs_name_remap = {
    "joint_pos": "dof_pos",
    "joint_vel": "dof_vel",
}


def generate_rl_controller_config(env) -> dict:
    """Generate RL controller configuration from environment.

    Args:
        env: The wrapped RSL-RL environment.

    Returns:
        Dictionary containing the RL controller configuration.
    """
    cfg = {
        "joint_order": [],
        "active_joint_order": [],
        "commands": {
            "ranges": {},
            "scales": {},
            "names": [],
        },
        "control": {
            "control_type": {},
            "stiffness": {},
            "damping": {},
            "torque_limits": {},
            "action_scale": 1.0,
            "clip_actions": float('inf'),
            "use_lpf": False,
            "decimation": None,
        },
        "sim": {
            "dt": 0.0,
        },
        "init_state": {
            "default_joint_angles": {},
            "default_pos": [],
        },
        "observation": {
            "num_obs_hist": {},
            "hist_by_term": False,
            "order": [],
            "scale": {},
            "dims": {},
            "clip": {},
            "noise": {
                "add_noise": False,
            },
        },
    }

    # Get the unwrapped environment
    env_unwrapped = env.unwrapped

    # Get observation info
    obs_names = list(env_unwrapped.observation_manager._group_obs_term_names.get("policy", []))
    term_cfgs = list(env_unwrapped.observation_manager._group_obs_term_cfgs.get("policy", []))
    term_dims = list(env_unwrapped.observation_manager._group_obs_term_dim.get("policy", []))

    # Process commands
    for cmd_term in env_unwrapped.command_manager.active_terms:
        cmd_cfg = getattr(env_unwrapped.command_manager.cfg, cmd_term)
        cmd_scale = 1.0

        # Find scale from observation terms
        for obs_term in term_cfgs:
            if obs_term.params.get("command_name", "") == cmd_term:
                cmd_scale = float(obs_term.scale) if obs_term.scale is not None else 1.0
                break

        if hasattr(cmd_cfg, "ranges"):
            if hasattr(cmd_cfg.ranges, "to_dict"):
                cmd_ranges = cmd_cfg.ranges.to_dict()
            elif isinstance(cmd_cfg.ranges, dict):
                cmd_ranges = cmd_cfg.ranges
            else:
                continue

            for cmd, rng in cmd_ranges.items():
                if cmd == "heading":
                    continue
                if rng is not None:
                    cfg["commands"]["ranges"][cmd] = list(rng)
                    cfg["commands"]["scales"][cmd] = cmd_scale
                    cfg["commands"]["names"].append(cmd)

    # Process observations
    cfg["observation"]["hist_by_term"] = (
        True if env_unwrapped.observation_manager.cfg.policy.history_length is None else False
    )

    for name, term_cfg, dims in zip(obs_names, term_cfgs, term_dims):
        history_length = max(getattr(term_cfg, 'history_length', 1) or 1, 1)
        obs_dim = dims[0] if isinstance(dims, (list, tuple)) else dims
        mapped_name = obs_name_remap.get(name, name)

        if "command" not in name.lower() and "velocity_commands" not in name.lower():
            cfg["observation"]["dims"][mapped_name] = int(obs_dim) // history_length
            cfg["observation"]["scale"][mapped_name] = (
                float(term_cfg.scale) if term_cfg.scale is not None else 1.0
            )
            cfg["observation"]["clip"][mapped_name] = (
                list(term_cfg.clip) if term_cfg.clip is not None else [-float('inf'), float('inf')]
            )
            cfg["observation"]["num_obs_hist"][mapped_name] = history_length
            cfg["observation"]["order"].append(mapped_name)
        else:
            if "commands" in cfg["observation"]["num_obs_hist"]:
                assert (
                    history_length == cfg["observation"]["num_obs_hist"]["commands"]
                ), "only equal history length for commands supported"
            else:
                cfg["observation"]["num_obs_hist"]["commands"] = history_length

    if len(cfg["commands"]["names"]) != 0:
        cfg["observation"]["order"].append("commands")
        cfg["observation"]["dims"]["commands"] = len(cfg["commands"]["names"])

    # Process actions/joints
    joint_order = []
    if "joint_pos" in env_unwrapped.action_manager._terms:
        p_joints = list(env_unwrapped.action_manager._terms["joint_pos"]._joint_names)
        joint_order += p_joints
        cfg["control"]["control_type"].update({j: "P" for j in p_joints})
    if "joint_vel" in env_unwrapped.action_manager._terms:
        v_joints = list(env_unwrapped.action_manager._terms["joint_vel"]._joint_names)
        joint_order += v_joints
        cfg["control"]["control_type"].update({j: "V" for j in v_joints})

    cfg["active_joint_order"] = list(joint_order)

    # Default joint positions
    robot = env_unwrapped.scene.articulations.get("robot")
    if robot is not None:
        default_position_dict = copy.copy(robot.cfg.init_state.joint_pos)
        try:
            indices, names, values = resolve_matching_names_values(
                data=default_position_dict,
                list_of_strings=robot.joint_names,
            )
            cfg["init_state"]["default_joint_angles"].update(
                {name: value for name, value in zip(names, values)}
            )
        except Exception as e:
            logger.warning(f"Could not resolve joint default positions: {e}")

        cfg["init_state"]["default_pos"] = list(robot.cfg.init_state.pos)

        # Get actuator configs
        for actuator_group_name in robot.cfg.actuators.keys():
            actuator_config = robot.cfg.actuators[actuator_group_name]

            # Stiffness
            stiffness_dict = copy.copy(actuator_config.stiffness)
            if isinstance(stiffness_dict, dict):
                try:
                    indices, names, values = resolve_matching_names_values(
                        data=stiffness_dict,
                        list_of_strings=robot.joint_names,
                    )
                    cfg["joint_order"] = list(joint_order) + [
                        name for name in names if name not in joint_order
                    ]
                    cfg["control"]["stiffness"].update(
                        {name: value for name, value in zip(names, values)}
                    )
                except Exception as e:
                    logger.warning(f"Could not resolve stiffness: {e}")

            # Damping
            damping_dict = copy.copy(actuator_config.damping)
            if isinstance(damping_dict, dict):
                try:
                    indices, names, values = resolve_matching_names_values(
                        data=damping_dict,
                        list_of_strings=robot.joint_names,
                    )
                    cfg["control"]["damping"].update(
                        {name: value for name, value in zip(names, values)}
                    )
                except Exception as e:
                    logger.warning(f"Could not resolve damping: {e}")

            # Torque limits
            torque_limit_dict = copy.copy(getattr(actuator_config, 'effort_limit', None))
            if torque_limit_dict is not None and isinstance(torque_limit_dict, dict):
                try:
                    indices, names, values = resolve_matching_names_values(
                        data=torque_limit_dict,
                        list_of_strings=robot.joint_names,
                    )
                    cfg["control"]["torque_limits"].update(
                        {name: value for name, value in zip(names, values)}
                    )
                except Exception as e:
                    logger.warning(f"Could not resolve torque limits: {e}")

    # Action scale
    if "joint_pos" in env_unwrapped.action_manager._terms:
        cfg["control"]["action_scale"] = env_unwrapped.action_manager._terms["joint_pos"]._scale

    # Decimation and simulation dt
    cfg["control"]["decimation"] = env_unwrapped.cfg.decimation
    cfg["sim"]["dt"] = env_unwrapped.sim.get_physics_dt()

    return cfg


def export_policy(env, runner, export_dir: str):
    """Export policy to JIT and ONNX formats and save config.yaml.

    Args:
        env: The wrapped RSL-RL environment.
        runner: The RSL-RL runner containing the trained policy.
        export_dir: Directory to save exported files.
    """
    print(f"[INFO] Exporting policy to: {export_dir}")

    # Get the policy neural network
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Get normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = runner.obs_normalizer if hasattr(runner, 'obs_normalizer') else None

    # Export to JIT
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    print(f"[INFO] Exported JIT policy: {os.path.join(export_dir, 'policy.pt')}")

    # Export to ONNX
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported ONNX policy: {os.path.join(export_dir, 'policy.onnx')}")

    # Generate and save config.yaml
    try:
        ctrl_cfg = generate_rl_controller_config(env)
        dump_yaml(os.path.join(export_dir, "config.yaml"), ctrl_cfg)
        print(f"[INFO] Saved config: {os.path.join(export_dir, 'config.yaml')}")
    except Exception as e:
        logger.warning(f"Could not generate config.yaml: {e}")
        print(f"[WARNING] Could not generate config.yaml: {e}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent and export policy."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Export policy after training (unless --no-export flag is set)
    if not args_cli.no_export:
        export_dir = os.path.join(log_dir, "exported")
        export_policy(env, runner, export_dir)
    else:
        print("[INFO] Skipping policy export (--no-export flag set)")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
