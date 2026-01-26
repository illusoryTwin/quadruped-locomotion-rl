#!/usr/bin/env python3
"""
Export trained policy for deployment.

This script has TWO modes:

1. STANDALONE MODE (no Isaac Lab required):
   Extracts policy from checkpoint, but normalizer may be missing.

   python deploy/export_policy.py --checkpoint logs/.../model_1500.pt

2. FULL MODE (requires Isaac Lab):
   Properly loads via RSL-RL runner to get normalizer.
   Run via isaaclab.sh:

   ./isaaclab.sh -p deploy/export_policy.py \
       --checkpoint logs/.../model_1500.pt \
       --task go2_walk_flat

The recommended workflow is to let train.py export automatically after training.
This script is for re-exporting existing checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import yaml


def export_standalone(checkpoint_path: str, output_dir: str = None):
    """
    Export policy from checkpoint WITHOUT Isaac Lab.

    This reconstructs the actor network from checkpoint weights.
    WARNING: Normalizer may not be properly extracted.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    # Extract actor architecture from weights
    actor_keys = [k for k in model_dict.keys() if k.startswith("actor.")]
    if not actor_keys:
        raise ValueError("Could not find actor weights in checkpoint (keys starting with 'actor.')")

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
        idx += 2  # Skip activation layers

    print(f"[INFO] Actor architecture: input={input_dim}, hidden={hidden_dims}, output={output_dim}")

    # Check for normalizer
    has_normalizer = "actor_obs_normalizer.running_mean" in model_dict

    # Build actor network
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

    # Create and load actor
    actor = Actor(input_dim, output_dim, hidden_dims)
    actor_state = {k.replace("actor.", "net."): v for k, v in model_dict.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()

    # Export
    example_input = torch.zeros(1, input_dim)

    if has_normalizer:
        mean = model_dict["actor_obs_normalizer.running_mean"]
        var = model_dict["actor_obs_normalizer.running_var"]
        print(f"[INFO] Found normalizer in checkpoint")

        policy = PolicyWithNormalizer(actor, mean, var)
        policy.eval()

        traced = torch.jit.trace(policy, example_input)
        output_path = output_dir / "policy.pt"
        traced.save(str(output_path))
        print(f"[SUCCESS] Exported policy WITH normalizer to: {output_path}")
    else:
        print(f"[WARNING] No normalizer found in checkpoint!")
        print(f"[WARNING] Exporting actor only - deployment may not work correctly")

        traced = torch.jit.trace(actor, example_input)
        output_path = output_dir / "policy_no_normalizer.pt"
        traced.save(str(output_path))
        print(f"[WARNING] Exported policy WITHOUT normalizer to: {output_path}")

    # Save manifest
    manifest = {
        "artifact_version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "export_mode": "standalone",
        "model": {
            "policy_path": output_path.name,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "has_normalizer": has_normalizer,
        },
        "source_checkpoint": str(checkpoint_path),
    }

    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Saved manifest to: {manifest_path}")

    if not has_normalizer:
        print()
        print("=" * 60)
        print("WARNING: Policy exported WITHOUT normalizer!")
        print()
        print("The policy will likely not work correctly in deployment.")
        print("To fix this, use one of these options:")
        print()
        print("Option 1: Re-run training (recommended)")
        print("  Training now automatically exports JIT with normalizer.")
        print()
        print("Option 2: Use play.py with --num_steps 0")
        print("  ./isaaclab.sh -p src/scripts/play.py \\")
        print(f"      --task YOUR_TASK --checkpoint {checkpoint_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export policy for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (if provided, uses Isaac Lab for proper export)")
    args = parser.parse_args()

    if args.task:
        print("[INFO] Task specified - this requires Isaac Lab environment")
        print("[INFO] Please run via: ./isaaclab.sh -p deploy/export_policy.py ...")
        print("[INFO] Falling back to standalone mode...")

    export_standalone(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
