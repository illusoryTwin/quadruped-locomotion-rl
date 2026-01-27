"""
Action post-processing for deployment.

Handles:
- Action scaling and clipping
- Adding default joint positions
- Torque limiting (optional)
- Multiple control types (P/V/T)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class Action:
    """Action post-processor for robot control."""

    def __init__(
        self,
        config: Dict,
        joint_order: List[str],
        default_joint_pos: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            config: Control configuration dict with keys:
                - action_scale: Scale factor for actions (default 0.5)
                - clip_actions: Clip bound for raw actions (default inf)
                - control_type: "P" (position), "V" (velocity), "T" (torque)
                - stiffness: Per-joint Kp gains (for torque limiting)
                - damping: Per-joint Kd gains (for torque limiting)
                - torque_limits: Per-joint torque limits
                - limit_actions_by_torque: Enable torque limiting (default False)
            joint_order: List of joint names in policy order
            default_joint_pos: Default joint positions tensor
            device: Torch device
        """
        self.cfg = config
        self.joint_order = joint_order
        self.default_joint_pos = default_joint_pos.to(device)
        self.device = device
        self.num_actions = len(joint_order)

        # Action processing params
        self.action_scale = self.cfg.get("action_scale", 0.5)
        self.action_clip = self.cfg.get("clip_actions", float("inf"))

        # Control type (can be per-joint or global)
        control_type = self.cfg.get("control_type", "P")
        if isinstance(control_type, dict):
            self.control_type = np.array(
                [control_type.get(joint, "P") for joint in joint_order]
            )
        else:
            self.control_type = np.array([control_type] * self.num_actions)

        # PD gains for torque limiting
        self.p_gains = torch.zeros(self.num_actions, device=device)
        self.d_gains = torch.zeros(self.num_actions, device=device)
        self._init_control_gains()

        # Torque limiting
        self.limit_actions_by_torque = self.cfg.get("limit_actions_by_torque", False)
        if self.limit_actions_by_torque:
            self.torque_limits = torch.tensor(
                [self.cfg["torque_limits"].get(joint, 23.7) for joint in joint_order],
                dtype=torch.float32,
                device=device,
            )

    def _init_control_gains(self):
        """Initialize PD gains from config."""
        stiffness = self.cfg.get("stiffness", {})
        damping = self.cfg.get("damping", {})

        # Default gains if not specified
        default_kp = self.cfg.get("kp", 25.0)
        default_kd = self.cfg.get("kd", 0.5)

        for i, joint in enumerate(self.joint_order):
            # Check for exact match or partial match (e.g., "hip" matches "FL_hip_joint")
            kp = default_kp
            kd = default_kd

            if isinstance(stiffness, dict):
                for k, v in stiffness.items():
                    if k in joint or k == joint:
                        kp = v
                        break
            elif isinstance(stiffness, (int, float)):
                kp = stiffness

            if isinstance(damping, dict):
                for k, v in damping.items():
                    if k in joint or k == joint:
                        kd = v
                        break
            elif isinstance(damping, (int, float)):
                kd = damping

            self.p_gains[i] = kp
            self.d_gains[i] = kd

    def process_action(
        self,
        action: torch.Tensor,
        joint_pos: Optional[torch.Tensor] = None,
        joint_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process raw policy action to joint commands.

        Args:
            action: Raw action from policy (typically in [-1, 1])
            joint_pos: Current joint positions (for torque limiting)
            joint_vel: Current joint velocities (for torque limiting)

        Returns:
            Tuple of (raw_action, target_joint_state)
        """
        # 1. Clip raw action
        action_raw = torch.clamp(action, -self.action_clip, self.action_clip)

        # 2. Scale action
        target = action_raw * self.action_scale

        # 3. Add default position for position control
        pos_mask = self.control_type == "P"
        target[pos_mask] = target[pos_mask] + self.default_joint_pos[pos_mask]

        # 4. Apply torque limiting if enabled
        if self.limit_actions_by_torque and joint_pos is not None and joint_vel is not None:
            target = self._limit_by_torque(target, joint_pos, joint_vel)

        return action_raw, target

    def _limit_by_torque(
        self,
        target: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Limit target positions/velocities by torque constraints.

        For position control: τ = Kp * (q_target - q) - Kd * dq
        Ensures |τ| <= τ_max
        """
        result = target.clone()

        # Position control torque limiting
        pos_mask = self.control_type == "P"
        if pos_mask.any():
            # Compute max allowable position delta given torque limit
            # τ = Kp * Δq - Kd * dq <= τ_max
            # Δq <= (τ_max + Kd * dq) / Kp
            max_delta = (self.torque_limits + self.d_gains * joint_vel) / self.p_gains
            min_delta = (-self.torque_limits + self.d_gains * joint_vel) / self.p_gains

            clipped_delta = torch.clamp(
                target - joint_pos,
                min_delta,
                max_delta,
            )
            result[pos_mask] = (joint_pos + clipped_delta)[pos_mask]

        # Velocity control torque limiting
        vel_mask = self.control_type == "V"
        if vel_mask.any():
            max_vel_delta = self.torque_limits / self.d_gains
            clipped_vel = torch.clamp(
                target - joint_vel,
                -max_vel_delta,
                max_vel_delta,
            )
            result[vel_mask] = (joint_vel + clipped_vel)[vel_mask]

        # Torque control - direct clipping
        torque_mask = self.control_type == "T"
        if torque_mask.any():
            result[torque_mask] = torch.clamp(
                target[torque_mask],
                -self.torque_limits[torque_mask],
                self.torque_limits[torque_mask],
            )

        return result

    def reset(self):
        """Reset action processor state (placeholder for future use)."""
        pass

    def get_gains(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get PD gains as numpy arrays for robot interface."""
        return self.p_gains.cpu().numpy(), self.d_gains.cpu().numpy()
