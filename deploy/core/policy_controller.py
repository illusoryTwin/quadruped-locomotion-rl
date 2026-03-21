import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config


class PolicyController:
    def __init__(self, policy_path: str, policy_config: "Config", device="cpu"):
        self.device = device
        self.policy_config = policy_config

        # Load the policy
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # Per-term observation history buffers (critical for correct observation order)
        self.obs_history = {term.name: None for term in policy_config.obs_terms}

        # Last action for observation building
        self.last_action = torch.zeros(12, dtype=torch.float32, device=device)

        # Commands from config
        self.cmd_names = policy_config.cmd_names
        self.cmd_scales = policy_config.cmd_scales
        self.commands = {name: 0.0 for name in self.cmd_names}

    def set_commands(self, **kwargs):
        """Set velocity commands for the robot."""
        for name, value in kwargs.items():
            if name in self.commands:
                self.commands[name] = value

    def _build_observation(self, obs_dict: dict) -> torch.Tensor:
        """Build full observation with per-term history buffers.

        Isaac Lab with flatten_history_dim=True produces:
        [term1_t0..t9, term2_t0..t9, ...]

        NOT: [all_terms_t0, all_terms_t1, ...]
        """
        obs_parts = []

        for term in self.policy_config.obs_terms:
            name = term.name
            dim = term.dim
            hist_len = term.hist_len

            # Get current value for this observation term
            if name == "dof_pos":
                val = obs_dict["dof_pos"] - self.policy_config.default_joint_pos
            elif name == "actions":
                val = self.last_action
            elif name == "commands":
                val = torch.tensor(
                    [self.commands[n] * self.cmd_scales[n] for n in self.cmd_names],
                    dtype=torch.float32,
                    device=self.device
                )
            elif name == "stiffness_cmd":
                # At deploy time, use a fixed stiffness value (midpoint of training range)
                val = torch.tensor(
                    [self.policy_config.deploy_stiffness],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                val = obs_dict.get(name)
                if val is None:
                    raise ValueError(f"Missing observation term: {name}")

            # Ensure tensor is on correct device
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32, device=self.device)
            val = val.to(self.device)

            # Update per-term history buffer
            if self.obs_history[name] is None:
                # Initialize buffer by repeating current value
                self.obs_history[name] = val.repeat(hist_len)
            else:
                # Shift buffer and append new value
                self.obs_history[name] = torch.cat([
                    self.obs_history[name][dim:],
                    val
                ])

            obs_parts.append(self.obs_history[name])

        return torch.cat(obs_parts)

    def get_action(self, obs_dict: dict) -> torch.Tensor:
        """Get action from policy given current observation."""
        obs = self._build_observation(obs_dict)

        with torch.no_grad():
            action = self.policy(obs.unsqueeze(0)).squeeze()

        # Store action for next observation
        self.last_action = action.clone()

        # Convert to target joint positions
        target_pos = action * self.policy_config.action_scale + self.policy_config.default_joint_pos

        return target_pos
