import torch
from isaaclab.utils.math import matrix_from_quat, quat_inv


def transform_vector_world2body(
    vec_w: torch.Tensor, body_quat_w: torch.Tensor
) -> torch.Tensor:
    """Transform a vector from world frame to body frame (vectorized).

    Args:
        vec_w: Vector in world frame [num_envs, num_bodies, 3]
        body_quat_w: Body quaternion in world frame [num_envs, num_bodies, 4]

    Returns:
        Vector in body frame [num_envs, num_bodies, 3]
    """
    # Flatten for batched quaternion conversion: [num_envs * num_bodies, 4]
    quat_flat = body_quat_w.reshape(-1, 4)

    # R_w_b: rotation from body to world, shape [num_envs * num_bodies, 3, 3]
    R_w_b = matrix_from_quat(quat_flat)

    # R_b_w = R_w_b^T (transpose last two dims)
    R_b_w = R_w_b.transpose(-1, -2)

    # Reshape vector for batched matmul: [num_envs * num_bodies, 3, 1]
    vec_flat = vec_w.reshape(-1, 3, 1)

    # Apply rotation and reshape back
    vec_b = (R_b_w @ vec_flat).reshape(vec_w.shape)

    return vec_b

def transform_vector_body2world(
    vec_b: torch.Tensor, body_quat_w: torch.Tensor
) -> torch.Tensor:
    """Transform a vector from body frame to world frame (vectorized).

    Args:
        vec_b: Vector in body frame [num_envs, num_bodies, 3]
        body_quat_w: Body quaternion in world frame [num_envs, num_bodies, 4]

    Returns:
        Vector in world frame [num_envs, num_bodies, 3]
    """
    # Flatten for batched quaternion conversion: [num_envs * num_bodies, 4]
    quat_flat = body_quat_w.reshape(-1, 4)

    # R_w_b: rotation from body to world, shape [num_envs * num_bodies, 3, 3]
    R_w_b = matrix_from_quat(quat_flat)

    # Reshape vector for batched matmul: [num_envs * num_bodies, 3, 1]
    vec_flat = vec_b.reshape(-1, 3, 1)

    # Apply rotation and reshape back
    vec_w = (R_w_b @ vec_flat).reshape(vec_b.shape)

    return vec_w


def transform_jacobian_world2body(jacobian_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
    """Transform Jacobian from world frame to body frame (vectorized).

    Args:
        jacobian_w: Jacobian in world frame [num_envs, num_bodies, 6, num_joints]
        body_quat_w: Body quaternions [num_envs, num_bodies, 4]

    Returns:
        Jacobian in body frame [num_envs, num_bodies, 6, num_joints]
    """
    num_envs, num_bodies, _, num_joints = jacobian_w.shape

    # Flatten env and body dimensions for batched quaternion conversion
    # Shape: [num_envs * num_bodies, 4]
    quat_flat = body_quat_w.reshape(-1, 4)

    # Convert quaternions to rotation matrices (batched)
    # R_w_b: rotation from body to world, shape [num_envs * num_bodies, 3, 3]
    R_w_b = matrix_from_quat(quat_flat)

    # R_b_w = R_w_b^T (transpose last two dims)
    R_b_w = R_w_b.transpose(-1, -2)

    # Reshape for broadcasting: [num_envs, num_bodies, 3, 3]
    R_b_w = R_b_w.reshape(num_envs, num_bodies, 3, 3)

    # Transform Jacobian: J_b = R_b_w @ J_w
    # jacobian_w linear part: [num_envs, num_bodies, 3, num_joints]
    # jacobian_w angular part: [num_envs, num_bodies, 3, num_joints]
    # Use einsum for batched matrix multiplication: R @ J
    jacobian_b_linear = torch.einsum('ebij,ebjk->ebik', R_b_w, jacobian_w[:, :, :3, :])
    jacobian_b_angular = torch.einsum('ebij,ebjk->ebik', R_b_w, jacobian_w[:, :, 3:, :])

    # Concatenate linear and angular parts
    jacobian_b = torch.cat([jacobian_b_linear, jacobian_b_angular], dim=2)

    return jacobian_b
