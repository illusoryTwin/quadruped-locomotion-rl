#!/usr/bin/env python3
"""Debug script to print observations during play"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks
import envs  # Register custom environments

from envs.flat_walk_env_cfg import UnitreeGo2WalkEnvCfg

# Create config
cfg = UnitreeGo2WalkEnvCfg()
cfg.scene.num_envs = 1

env = gym.make("go2_walk_flat", cfg=cfg)

obs, info = env.reset()
print("\n" + "="*60)
print("OBSERVATION DEBUG")
print("="*60)
print(f"Observation shape: {obs['policy'].shape}")
print()

o = obs['policy'][0].cpu().numpy()
print(f"base_ang_vel [0:3]: {o[0:3]}")
print(f"projected_gravity [3:6]: {o[3:6]}")
print(f"velocity_commands [6:9]: {o[6:9]}")
print(f"joint_pos_rel [9:21]: {o[9:21]}")
print(f"joint_vel [21:33]: {o[21:33]}")
print(f"last_action [33:45]: {o[33:45]}")
print(f"height_scan [45:50]: {o[45:50]}...")
print(f"height_scan mean: {o[45:232].mean():.4f}")
print()
print(f"Obs min: {o.min():.4f}, max: {o.max():.4f}, mean: {o.mean():.4f}")
print("="*60)

# Print joint order
robot = env.unwrapped.scene["robot"]
print("\nJoint names (order):")
for i, name in enumerate(robot.joint_names):
    print(f"  {i}: {name}")

env.close()
simulation_app.close()
