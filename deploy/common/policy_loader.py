"""
Policy loading utilities for RSL-RL trained models.

Supports two formats:
1. JIT-exported policy (exported/policy.pt) - preferred, includes normalizer
2. Raw checkpoint (model_*.pt) - requires normalizer to be stored in checkpoint
"""

import os
import torch
import torch.nn as nn
from typing import Optional, List


class MLP(nn.Module):
    """Simple MLP matching RSL-RL actor architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmpiricalNormalization:
    """Empirical observation normalization."""

    def __init__(self, mean: torch.Tensor, var: torch.Tensor, epsilon: float = 1e-8):
        self.mean = mean
        self.var = var
        self.epsilon = epsilon

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / torch.sqrt(self.var + self.epsilon)


class JitPolicyWrapper:
    """Wrapper for JIT-exported policy (includes normalizer)."""

    def __init__(self, jit_path: str, device: str = "cpu"):
        self.device = device
        print(f"[INFO] Loading JIT policy: {jit_path}")
        self.policy = torch.jit.load(jit_path, map_location=device)
        self.policy.eval()

        # Try to infer dimensions from the model
        # JIT models don't expose this easily, so we'll use config values
        self.input_dim = None
        self.output_dim = None

    def __call__(self, obs: torch.Tensor, debug: bool = False) -> torch.Tensor:
        with torch.inference_mode():
            return self.policy(obs)


class RawPolicyWrapper:
    """Wrapper for raw RSL-RL checkpoint (may need external normalizer)."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device

        # Load checkpoint
        print(f"[INFO] Loading raw checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            model_dict = checkpoint["model_state_dict"]
        else:
            model_dict = checkpoint

        # Find actor keys
        actor_keys = [k for k in model_dict.keys() if k.startswith("actor.")]
        if not actor_keys:
            raise ValueError("Could not find actor weights in checkpoint")

        # Get input/output dimensions
        first_layer_key = "actor.0.weight"
        if first_layer_key not in model_dict:
            raise ValueError(f"Could not find {first_layer_key} in checkpoint")

        self.input_dim = model_dict[first_layer_key].shape[1]

        # Get output dim from last layer
        last_layer_idx = max(int(k.split(".")[1]) for k in actor_keys if ".weight" in k)
        last_layer_key = f"actor.{last_layer_idx}.weight"
        self.output_dim = model_dict[last_layer_key].shape[0]

        print(f"[INFO] Actor input dim: {self.input_dim}, output dim: {self.output_dim}")

        # Determine hidden dimensions
        hidden_dims = []
        idx = 0
        while f"actor.{idx}.weight" in model_dict:
            if idx < last_layer_idx:
                hidden_dims.append(model_dict[f"actor.{idx}.weight"].shape[0])
            idx += 2  # Skip activation layers

        print(f"[INFO] Hidden dims: {hidden_dims}")

        # Build actor network
        self.actor = MLP(self.input_dim, self.output_dim, hidden_dims).to(device)

        # Load actor weights
        actor_state_dict = {}
        for key in actor_keys:
            new_key = key.replace("actor.", "net.")
            actor_state_dict[new_key] = model_dict[key]
        self.actor.load_state_dict(actor_state_dict)
        self.actor.eval()

        # Load normalizer if available
        self.normalizer: Optional[EmpiricalNormalization] = None
        if "actor_obs_normalizer.running_mean" in model_dict:
            mean = model_dict["actor_obs_normalizer.running_mean"]
            var = model_dict["actor_obs_normalizer.running_var"]
            self.normalizer = EmpiricalNormalization(mean, var)
            print("[INFO] Loaded observation normalizer from checkpoint")
        else:
            print("[INFO] No normalizer in checkpoint, using estimated normalization")
            self.normalizer = self._create_estimated_normalizer(self.input_dim, device)

    def _create_estimated_normalizer(self, obs_dim: int, device: str) -> EmpiricalNormalization:
        """Create estimated normalizer based on typical observation ranges.

        Observation structure (232 dims total):
        - base_ang_vel: 3 dims, range ~[-3, 3] rad/s
        - projected_gravity: 3 dims, range [-1, 1]
        - velocity_commands: 3 dims, range ~[-1.5, 1.5]
        - joint_pos_rel: 12 dims, range ~[-0.5, 0.5] rad
        - joint_vel: 12 dims, range ~[-10, 10] rad/s
        - last_action: 12 dims, range [-1, 1]
        - height_scan: 187 dims, range ~[-1, 1] (relative heights)
        """
        # Estimated standard deviations for each observation component
        std_values = []

        # base_ang_vel (3): std ~1.0
        std_values.extend([1.0] * 3)
        # projected_gravity (3): std ~0.5
        std_values.extend([0.5] * 3)
        # velocity_commands (3): std ~0.8
        std_values.extend([0.8] * 3)
        # joint_pos_rel (12): std ~0.3
        std_values.extend([0.3] * 12)
        # joint_vel (12): std ~3.0
        std_values.extend([3.0] * 12)
        # last_action (12): std ~0.5
        std_values.extend([0.5] * 12)

        # height_scan (remaining dims): std ~0.3
        height_scan_dim = obs_dim - 45
        if height_scan_dim > 0:
            std_values.extend([0.3] * height_scan_dim)

        mean = torch.zeros(obs_dim, device=device)
        var = torch.tensor(std_values, device=device) ** 2  # variance = std^2

        return EmpiricalNormalization(mean, var)

    def __call__(self, obs: torch.Tensor, debug: bool = False) -> torch.Tensor:
        with torch.inference_mode():
            if debug:
                print(f"  [POLICY] Input obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")

            if self.normalizer is not None:
                obs_normalized = self.normalizer.normalize(obs)
                if debug:
                    print(f"  [POLICY] Normalized obs range: [{obs_normalized.min():.3f}, {obs_normalized.max():.3f}]")
                obs = obs_normalized

            output = self.actor(obs)
            if debug:
                print(f"  [POLICY] Actor output range: [{output.min():.3f}, {output.max():.3f}]")
            return output


def load_policy(checkpoint_path: str, device: str = "cpu"):
    """Load a trained policy from checkpoint.

    Automatically detects and prefers JIT-exported policy if available.
    """
    # Check if this is a raw checkpoint and look for exported JIT policy
    checkpoint_dir = os.path.dirname(checkpoint_path)
    jit_path = os.path.join(checkpoint_dir, "exported", "policy.pt")

    if os.path.exists(jit_path):
        print(f"[INFO] Found exported JIT policy, using it instead of raw checkpoint")
        wrapper = JitPolicyWrapper(jit_path, device)
        # Get dimensions from raw checkpoint for compatibility
        raw_wrapper = RawPolicyWrapper(checkpoint_path, device)
        wrapper.input_dim = raw_wrapper.input_dim
        wrapper.output_dim = raw_wrapper.output_dim
        return wrapper
    else:
        print(f"[INFO] No exported JIT policy found at {jit_path}")
        print(f"[INFO] Using raw checkpoint (normalizer may be missing)")
        return RawPolicyWrapper(checkpoint_path, device)
