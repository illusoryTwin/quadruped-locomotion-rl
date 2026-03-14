import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def base_cartesian_deformation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation term: base body Cartesian deformation from MSD. Shape [num_envs, 3]."""
    if hasattr(env, 'compliance_manager') and env.compliance_manager is not None:
        msd = env.compliance_manager._msd_system
        if msd is not None:
            return msd.state['x_def'][:, 0:3]
    return torch.zeros(env.num_envs, 3, device=env.device)


def track_compliant_base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 0.3,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential height tracking reward for compliant base reference (Z only).

    z_ref = env_origin_z + target_height + x_def_z

    No forces  -> x_def_z ~ 0  -> robot stays at target_height
    Push down  -> x_def_z < 0  -> robot yields below target_height
    Push up    -> x_def_z > 0  -> robot extends above target_height

    reward = exp(-(z_sim - z_ref)^2 / std^2)
    """
    if not hasattr(env, 'compliance_manager') or env.compliance_manager is None:
        return torch.zeros(env.num_envs, device=env.device)

    msd = env.compliance_manager._msd_system
    x_def_z = msd.state['x_def'][:, 2]  # Z deformation from MSD

    robot = env.scene["robot"]
    z_ref = env.scene.env_origins[:, 2] + target_height + x_def_z
    z_err_sq = (robot.data.root_pos_w[:, 2] - z_ref).square()

    return torch.exp(-z_err_sq / std**2)


def track_base_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_position",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for tracking the commanded base position.

    reward = exp(-||pos_actual - (env_origin + pos_cmd)||^2 / std^2)
    """
    robot = env.scene["robot"]
    pos_cmd = env.command_manager.get_command(command_name)
    target_pos = env.scene.env_origins[:, :3] + pos_cmd
    pos_err = (robot.data.root_pos_w[:, :3] - target_pos).square().sum(dim=1)
    return torch.exp(-pos_err / std**2)


def track_compliant_base_pos_tanh(
    env: ManagerBasedRLEnv,
    pos_scale: float = 0.5,
) -> torch.Tensor:
    """Saturated position tracking penalty.

    reward = -tanh(||x_sim - x_ref|| / pos_scale)

    Gives gradient for small errors but saturates at -1 for large drift.
    """
    if not hasattr(env, '_compliant_ref_pos') or env._compliant_ref_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    pos_err = (robot.data.root_pos_w[:, :3] - env._compliant_ref_pos).norm(dim=1)

    return -torch.tanh(pos_err / pos_scale)


def track_compliant_base_pos_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
) -> torch.Tensor:
    """Exponential position tracking reward for compliant base reference.

    reward = exp(-||x_sim - x_ref||^2 / std^2)

    Returns 1.0 at zero error and decays toward 0 for large errors.
    """
    if not hasattr(env, '_compliant_ref_pos') or env._compliant_ref_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    pos_err_sq = (robot.data.root_pos_w[:, :3] - env._compliant_ref_pos).square().sum(dim=1)

    return torch.exp(-pos_err_sq / std**2)


def track_compliant_velocity_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for deviating from compliant velocity reference.

    vel_ref = commanded_velocity_world + MSD_deformation_velocity

    When no external forces: dx_def ~ 0 -> tracks commanded velocity.

    reward = -||v_sim - vel_ref||^2
    """
    if not hasattr(env, '_compliant_ref_vel') or env._compliant_ref_vel is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    vel_err = torch.sum((robot.data.root_lin_vel_w[:, :3] - env._compliant_ref_vel) ** 2, dim=1)

    return -vel_err


def track_compliant_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_position",
    std: float = 0.5,
    max_def: float = 0.3,
) -> torch.Tensor:
    """Velocity tracking reward with XY compliance deformation.

    Shifts the commanded target position by the MSD XY deformation,
    recomputes the body-frame velocity command, and rewards tracking it.

        target_xy  = pos_target_w + clamp(x_def_xy)
        vel_target = clip(stiffness * rotate_to_body(target_xy - current_xy))
        reward     = exp(-||vel_actual - vel_target||^2 / std^2)

    Falls back to standard velocity tracking when compliance is unavailable.
    """
    pos_cmd_term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    # Start with the original target position (world frame XY)
    target_xy = pos_cmd_term.pos_target_w.clone()

    # Add clamped XY deformation from compliance MSD
    if hasattr(env, 'compliance_manager') and env.compliance_manager is not None:
        msd = env.compliance_manager._msd_system
        x_def_xy = msd.state['x_def'][:, 0:2]
        target_xy = target_xy + max_def * torch.tanh(x_def_xy / max_def)

    # Position error in world frame
    pos_error_w = target_xy - robot.data.root_pos_w[:, :2]

    # Rotate to body frame
    heading = robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    error_body_x = cos_h * pos_error_w[:, 0] + sin_h * pos_error_w[:, 1]
    error_body_y = -sin_h * pos_error_w[:, 0] + cos_h * pos_error_w[:, 1]

    # P-controller with same gains as the position command
    stiffness = pos_cmd_term.cfg.position_control_stiffness
    vel_min, vel_max = pos_cmd_term.cfg.ranges.vel
    vel_target_x = torch.clip(stiffness * error_body_x, min=vel_min, max=vel_max)
    vel_target_y = torch.clip(stiffness * error_body_y, min=vel_min, max=vel_max)

    # Velocity tracking error
    vel_error = (robot.data.root_lin_vel_b[:, 0] - vel_target_x).square() + \
                (robot.data.root_lin_vel_b[:, 1] - vel_target_y).square()

    return torch.exp(-vel_error / std**2)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def track_compliant_base_pos_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_position",
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential height tracking reward: command_z + deformation_z.

    z_ref = env_origin_z + pos_cmd_z + x_def_z
    reward = exp(-(z_actual - z_ref)^2 / std^2)

    The rigid reference (env_origin + pos_cmd) is independent of actual_pos,
    so the reward signal is always meaningful for the policy.
    """
    if not hasattr(env, 'compliance_manager') or env.compliance_manager is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    pos_cmd = env.command_manager.get_command(command_name)  # [num_envs, 3]
    # print("pos_cmd[:, 2]", pos_cmd[:, 2])
    z_rigid = env.scene.env_origins[:, 2] + pos_cmd[:, 2]

    msd = env.compliance_manager._msd_system
    x_def_z = msd.state['x_def'][:, 2]  # Z deformation
    
    # max_def = 0.2 # 0.3 # usually trained with this value # prev value # 0.25 # 0.15
    # x_def_z = max_def * torch.tanh(x_def_z / max_def)
    
    # print("x_def_z", x_def_z)
    z_ref = z_rigid + x_def_z
    
    # print("z_rigid", z_rigid)
    # print("robot.data.root_pos_w[:, 2]", robot.data.root_pos_w[:, 2][0])
    # print("z_ref", z_ref)
    z_err_sq = (robot.data.root_pos_w[:, 2] - z_ref).square()
    return torch.exp(-z_err_sq / std**2)


def feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for keeping all feet in contact with the ground.

    Returns the fraction of feet in contact (0 to 1).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    in_contact = net_forces.norm(dim=-1) > threshold  # [num_envs, num_feet]
    return in_contact.float().mean(dim=1)

