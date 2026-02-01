import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def apply_compliance_force_torque(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply random forces and torques to the compliance buffers for monitored bodies.

    This function samples random forces and torques and writes them to the environment's
    compliance buffers (_compliance_force_b and _compliance_torque_b) which are used
    by the SoftComplianceManager to compute joint deformations.
    """
    # Check if compliance buffers exist
    if not hasattr(env, '_compliance_force_b') or env._compliance_force_b is None:
        return

    # Get robot and resolve body indices
    asset = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Resolve body indices from config
    if asset_cfg.body_ids is not None:
        body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else [asset_cfg.body_ids]
    else:
        body_ids = list(range(asset.num_bodies))

    num_bodies = len(body_ids)

    # Sample random forces (X direction only) and torques (pitch only)
    size_2d = (len(env_ids), num_bodies)
    size_3d = (len(env_ids), num_bodies, 3)

    # Forces: only X direction (forward/backward)
    forces = torch.zeros(size_3d, device=asset.device)
    forces[:, :, 0] = math_utils.sample_uniform(*force_range, size_2d, asset.device)

    # Torques: only pitch (around Y axis) for consistency with X force
    torques = torch.zeros(size_3d, device=asset.device)
    torques[:, :, 1] = math_utils.sample_uniform(*torque_range, size_2d, asset.device)

    # Write to compliance buffers
    for i, body_id in enumerate(body_ids):
        env._compliance_force_b[env_ids, body_id, :] = forces[:, i, :]
        env._compliance_torque_b[env_ids, body_id, :] = torques[:, i, :]
