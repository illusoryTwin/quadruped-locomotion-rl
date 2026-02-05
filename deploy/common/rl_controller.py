"""
Modular RL controller for quadruped locomotion.

Combines:
- Policy inference
- Observation processing with history
- Action post-processing
- Command generation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union

from .observation import Observation
from .action import Action
from .commander import Commander
from .policy_loader import load_policy


class RLController:
    """Reinforcement learning locomotion controller."""

    def __init__(
        self,
        config: Dict,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            config: Full configuration dict with keys:
                - checkpoint: Path to policy checkpoint
                - joint_order: List of joint names
                - observation: Observation config
                - control: Control/action config
                - commands: Command config
                - init_state: Initial state config (default_joint_pos)
            device: Torch device for inference
        """
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Joint configuration
        self.joint_order: List[str] = self.cfg["joint_order"]
        self.num_joints = len(self.joint_order)

        # Default joint positions
        default_angles = self.cfg.get("init_state", {}).get("default_joint_angles", {})
        if isinstance(default_angles, dict):
            self.default_joint_pos = torch.tensor(
                [default_angles.get(name, 0.0) for name in self.joint_order],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Assume it's a list
            self.default_joint_pos = torch.tensor(
                default_angles,
                dtype=torch.float32,
                device=self.device,
            )

        # Load policy
        checkpoint_path = self.cfg["checkpoint"]
        self.policy = load_policy(checkpoint_path, str(self.device))

        # Initialize components
        self.observation_processor = Observation(
            config=self.cfg["observation"],
            device=self.device,
        )

        self.action_processor = Action(
            config=self.cfg["control"],
            joint_order=self.joint_order,
            default_joint_pos=self.default_joint_pos,
            device=self.device,
        )

        self.commander = Commander(
            config=self.cfg.get("commands", {}),
            device=self.device,
        )

        # State
        self.last_action = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)

        self.reset()

    def __call__(
        self,
        obs_dict: Dict[str, Union[torch.Tensor, np.ndarray, List]],
    ) -> np.ndarray:
        """Run policy inference and return target joint positions.

        Args:
            obs_dict: Dictionary of observations matching config order.
                Required keys depend on observation.order config.
                Common keys: base_ang_vel, projected_gravity, dof_pos, dof_vel

        Returns:
            Target joint positions as numpy array
        """
        # Convert observations to tensors
        joint_pos = None
        joint_vel = None

        for name, obs in obs_dict.items():
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)
            elif isinstance(obs, list):
                obs = torch.tensor(obs)
            obs_dict[name] = obs.to(dtype=torch.float32, device=self.device)

            if name == "dof_pos":
                joint_pos = obs_dict[name].clone()
            elif name == "dof_vel":
                joint_vel = obs_dict[name].clone()

        # Prepare observation (add commands, last action, compute relative positions)
        observation = self.prepare_observation(obs_dict)

        # Run policy inference
        with torch.no_grad():
            action = self.policy(observation.unsqueeze(0)).squeeze(0)

        # Process action
        self.last_action, target_joint_pos = self.action_processor.process_action(
            action=action,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

        # Return as numpy
        if target_joint_pos.is_cuda:
            target_joint_pos = target_joint_pos.cpu()

        return target_joint_pos.numpy()

    def prepare_observation(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Prepare observation vector for policy.

        Adds derived observations (commands, last_action) and
        converts joint positions to relative.

        Args:
            obs_dict: Raw observation dictionary

        Returns:
            Processed observation tensor ready for policy
        """
        # Add last action to observations
        obs_dict["actions"] = self.last_action.clone()

        # Add commands to observations
        obs_dict["commands"] = self.commander.get_cmd()

        # Convert joint positions to relative (offset from default)
        if "dof_pos" in obs_dict:
            obs_dict["dof_pos"] = obs_dict["dof_pos"] - self.default_joint_pos

        # Process through observation handler (scaling, clipping, history)
        observation = self.observation_processor.prepare_observations(obs_dict)

        return observation

    def set_cmd(self, name: str, value: float):
        """Set a velocity command."""
        self.commander.set_cmd(name, value)

    def set_cmds(self, cmds_dict: Dict[str, float]):
        """Set multiple velocity commands."""
        self.commander.set_cmds(cmds_dict)

    def reset(self):
        """Reset controller state."""
        self.last_action = torch.zeros(
            self.num_joints, dtype=torch.float32, device=self.device
        )
        self.observation_processor.reset()
        self.action_processor.reset()
        self.commander.reset()

    def get_gains(self):
        """Get PD gains for robot interface."""
        return self.action_processor.get_gains()

    @property
    def obs_dim(self) -> int:
        """Total observation dimension including history."""
        return self.observation_processor.output_dim
