#!/usr/bin/env python3
"""
Export trained policy with normalizer for deployment.

This script loads a checkpoint and exports the policy as a JIT model
with observation normalization baked in.

Usage:
    conda activate isaaclab
    cd ~/Workspace/Projects/quadruped-locomotion-rl
    python deploy/export_policy.py --checkpoint logs/rsl_rl/unitree_go2_walk/2025-12-29_15-43-52/model_1500.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def export_policy(checkpoint_path: str, output_dir: str = None):
    """Export policy with normalizer as JIT model."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = checkpoint_path.parent / "exported"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get model state dict
    if "model_state_dict" in checkpoint:
        model_dict = checkpoint["model_state_dict"]
    else:
        model_dict = checkpoint

    # Get actor dimensions
    actor_keys = [k for k in model_dict.keys() if k.startswith("actor.")]
    first_layer = model_dict["actor.0.weight"]
    input_dim = first_layer.shape[1]

    last_layer_idx = max(int(k.split(".")[1]) for k in actor_keys if ".weight" in k)
    output_dim = model_dict[f"actor.{last_layer_idx}.weight"].shape[0]

    # Get hidden dims
    hidden_dims = []
    idx = 0
    while f"actor.{idx}.weight" in model_dict:
        if idx < last_layer_idx:
            hidden_dims.append(model_dict[f"actor.{idx}.weight"].shape[0])
        idx += 2

    print(f"[INFO] Actor: input={input_dim}, hidden={hidden_dims}, output={output_dim}")

    # Check for normalizer in checkpoint
    has_normalizer = "actor_obs_normalizer.running_mean" in model_dict

    if has_normalizer:
        mean = model_dict["actor_obs_normalizer.running_mean"]
        var = model_dict["actor_obs_normalizer.running_var"]
        print(f"[INFO] Found normalizer in checkpoint")
    else:
        print(f"[WARNING] No normalizer in checkpoint - will try to load from RSL-RL")

        # Try to load via RSL-RL
        try:
            from rsl_rl.modules import ActorCritic
            from rsl_rl.runners import OnPolicyRunner

            # Load the agent config
            agent_cfg_path = checkpoint_path.parent / "params" / "agent.yaml"
            if agent_cfg_path.exists():
                import yaml
                with open(agent_cfg_path) as f:
                    agent_cfg = yaml.safe_load(f)

                print(f"[INFO] Loaded agent config from {agent_cfg_path}")

                # Check if empirical normalization was used
                if agent_cfg.get("empirical_normalization", False):
                    print(f"[WARNING] Training used empirical_normalization but it's not in checkpoint")
                    print(f"[WARNING] The normalizer statistics were not saved during training")
                    print(f"[WARNING] You need to re-run training with a patched RSL-RL that saves normalizer")

        except ImportError:
            print(f"[WARNING] RSL-RL not available in this environment")

    # Build the actor network
    class Actor(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dims):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ELU())
                prev_dim = h
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class PolicyWithNormalizer(nn.Module):
        def __init__(self, actor, mean, var, epsilon=1e-8):
            super().__init__()
            self.actor = actor
            self.register_buffer("mean", mean)
            self.register_buffer("var", var)
            self.epsilon = epsilon

        def forward(self, obs):
            obs_normalized = (obs - self.mean) / torch.sqrt(self.var + self.epsilon)
            return self.actor(obs_normalized)

    # Create actor and load weights
    actor = Actor(input_dim, output_dim, hidden_dims)
    actor_state = {}
    for k in actor_keys:
        new_k = k.replace("actor.", "net.")
        actor_state[new_k] = model_dict[k]
    actor.load_state_dict(actor_state)
    actor.eval()

    if has_normalizer:
        # Export with normalizer
        policy = PolicyWithNormalizer(actor, mean, var)
        policy.eval()

        # Export as JIT
        example_input = torch.zeros(1, input_dim)
        traced = torch.jit.trace(policy, example_input)

        output_path = output_dir / "policy.pt"
        traced.save(str(output_path))
        print(f"[INFO] Exported policy with normalizer to: {output_path}")
    else:
        # Export actor only (without normalizer)
        example_input = torch.zeros(1, input_dim)
        traced = torch.jit.trace(actor, example_input)

        output_path = output_dir / "policy_no_normalizer.pt"
        traced.save(str(output_path))
        print(f"[WARNING] Exported policy WITHOUT normalizer to: {output_path}")
        print(f"[WARNING] This policy will likely not work correctly!")
        print(f"")
        print(f"To fix this, you have two options:")
        print(f"1. Run play.py in Isaac Lab to export with normalizer:")
        print(f"   conda activate isaaclab")
        print(f"   cd ~/Workspace/IsaacLab")
        print(f"   ./isaaclab.sh -p {Path(__file__).parent.parent}/src/scripts/play.py \\")
        print(f"       --task Isaac-Velocity-Flat-Unitree-Go2-v0 \\")
        print(f"       --checkpoint {checkpoint_path}")
        print(f"")
        print(f"2. Re-train with a fixed RSL-RL that saves the normalizer")


def main():
    parser = argparse.ArgumentParser(description="Export policy for deployment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
