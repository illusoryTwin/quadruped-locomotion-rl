import torch
import numpy as np


class URDFPolicyMapper:
    """Maps between URDF joint order (MuJoCo/SDK) and policy joint order."""

    URDF_ORDER = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __init__(self, policy_joint_order: list[str]):
        # urdf_to_policy[urdf_idx] = policy_idx for that joint
        self.urdf_to_policy = [
            policy_joint_order.index(name) for name in self.URDF_ORDER
        ]
        # policy_to_urdf[policy_idx] = urdf_idx for that joint
        self.policy_to_urdf = [
            self.URDF_ORDER.index(name) for name in policy_joint_order
        ]

    def to_policy_order(self, urdf_data):
        """Convert data from URDF order to policy order."""
        # policy_data[i] = urdf_data[urdf_idx_for_policy_joint_i]
        return urdf_data[self.policy_to_urdf]

    def to_urdf_order(self, policy_data):
        """Convert data from policy order to URDF order."""
        # urdf_data[i] = policy_data[policy_idx_for_urdf_joint_i]
        return policy_data[self.urdf_to_policy]
