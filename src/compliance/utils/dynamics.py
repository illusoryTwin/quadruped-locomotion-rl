"""Dynamics utilities for torque and wrench calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from compliance.utils.frame_transforms import (
    transform_jacobian_world2body,
    transform_vector_body2world,
    transform_vector_world2body,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def compute_joint_torques(
    jacobians: torch.Tensor,
    wrench: torch.Tensor,
) -> torch.Tensor:
    """Compute joint torques from Jacobian and wrench: tau = J^T @ wrench.

    Args:
        jacobians: Jacobians [num_envs, num_bodies, 6, num_dofs]
        wrench: Wrench [num_envs, num_bodies, 6]

    Returns:
        Joint torques [num_envs, num_dofs]
    """
    return torch.einsum('ebji,ebj->ei', jacobians, wrench)


def calculate_external_torques(
    robot: "Articulation",
    body_names: list[str],
    joint_mask: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Calculate joint torques from external forces on specified bodies.

    Uses world frame Jacobian and world frame wrench: tau = J_w^T @ wrench_w

    Args:
        robot: Isaac Lab Articulation instance
        body_names: List of body names to calculate torques for
        joint_mask: Optional boolean mask [num_dofs] to zero out inactive joints
        verbose: Print debug information

    Returns:
        Joint torques [num_envs, num_dofs]
    """
    body_indices = [robot.body_names.index(name) for name in body_names]

    # Get Jacobian in world frame
    jacobians_w = robot.root_physx_view.get_jacobians()[:, body_indices, :, :]

    # Get wrenches in body frame
    forces_b = robot._external_force_b[:, body_indices, :]
    torques_b = robot._external_torque_b[:, body_indices, :]

    body_quat_w = robot.data.body_quat_w[:, body_indices, :]

    # Transform forces/torques from body frame to world frame
    forces_w = transform_vector_body2world(forces_b, body_quat_w)
    torques_w = transform_vector_body2world(torques_b, body_quat_w)

    # Apply joint mask (zero out inactive joints)
    if joint_mask is not None:
        jacobians_w = jacobians_w.clone()
        jacobians_w[:, :, :, ~joint_mask] = 0

    if verbose:
        print("\n=== WORLD FRAME ===")
        for i, name in enumerate(body_names):
            print(f"\nBody: {name}")
            print(
                f"  Wrench (world): force={forces_w[0, i].cpu().numpy()}, torque={torques_w[0, i].cpu().numpy()}"
            )
            print(f"  Jacobian (world) shape: {jacobians_w[0, i].shape}")
            print(f"  Jacobian (world):\n{jacobians_w[0, i].cpu().numpy()}")

    # Stack wrenches: [num_envs, num_bodies, 6]
    wrench_w = torch.cat([forces_w, torques_w], dim=-1)

    # Vectorized torque calculation
    total_torques = compute_joint_torques(jacobians_w, wrench_w)

    if verbose:
        print(f"\nTotal torques (world frame): {total_torques[0].cpu().numpy()}")

    return total_torques


def calculate_external_torques_b(
    robot: "Articulation",
    body_names: list[str],
    joint_mask: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Calculate joint torques from external forces on specified bodies in BODY frame.

    Uses BODY frame Jacobian and BODY frame wrench: tau = J_b^T @ wrench_b

    Args:
        robot: Isaac Lab Articulation instance
        body_names: List of body names to calculate torques for
        joint_mask: Optional boolean mask [num_dofs] to zero out inactive joints
        verbose: Print debug information

    Returns:
        Joint torques [num_envs, num_dofs]
    """
    body_indices = [robot.body_names.index(name) for name in body_names]

    # Get Jacobian in world frame: [num_envs, num_bodies, 6, num_joints]
    jacobians_w = robot.root_physx_view.get_jacobians()[:, body_indices, :, :]

    # Get body orientations for frame transformation
    body_quat_w = robot.data.body_quat_w[:, body_indices, :]

    # Transform Jacobian from world frame to body frame
    jacobians_b = transform_jacobian_world2body(jacobians_w, body_quat_w)

    # Get wrenches in body frame
    forces_b = robot._external_force_b[:, body_indices, :]
    torques_b = robot._external_torque_b[:, body_indices, :]

    # Apply joint mask (zero out inactive joints)
    if joint_mask is not None:
        jacobians_b = jacobians_b.clone()
        jacobians_b[:, :, :, ~joint_mask] = 0

    if verbose:
        print("\n=== BODY FRAME ===")
        for i, name in enumerate(body_names):
            print(f"\nBody: {name}")
            print(
                f"  Wrench (body): force={forces_b[0, i].cpu().numpy()}, torque={torques_b[0, i].cpu().numpy()}"
            )
            print(f"  Jacobian (body) shape: {jacobians_b[0, i].shape}")
            print(f"  Jacobian (body):\n{jacobians_b[0, i].cpu().numpy()}")

    # Stack wrenches: [num_envs, num_bodies, 6]
    wrench_b = torch.cat([forces_b, torques_b], dim=-1)

    # Vectorized torque calculation
    total_torques = compute_joint_torques(jacobians_b, wrench_b)

    if verbose:
        print(f"\nTotal torques (body frame): {total_torques[0].cpu().numpy()}")

    return total_torques


def create_joint_mask(
    num_joints: int,
    active_joint_indices: list[int],
    fix_base: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Create DOF mask for active joints.

    Args:
        num_joints: Number of actuated joints
        active_joint_indices: List of active joint indices
        fix_base: If False, adds 6 DOF offset for floating base
        device: Torch device

    Returns:
        Boolean mask [num_dofs] with True for active DOFs
    """
    if fix_base:
        n_dofs = num_joints
        offset = 0
    else:
        n_dofs = num_joints + 6
        offset = 6

    mask = torch.zeros(n_dofs, dtype=torch.bool, device=device)
    for idx in active_joint_indices:
        mask[idx + offset] = True
    return mask
