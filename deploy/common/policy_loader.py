"""
Policy loading utilities for RSL-RL trained models.
"""

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


class PolicyWrapper:
    """Wrapper for RSL-RL trained policy."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device

        # Load checkpoint
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
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
            print("[INFO] Loaded observation normalizer")

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.normalizer is not None:
                obs = self.normalizer.normalize(obs)
            return self.actor(obs)


def load_policy(checkpoint_path: str, device: str = "cpu") -> PolicyWrapper:
    """Load a trained policy from checkpoint."""
    return PolicyWrapper(checkpoint_path, device)
