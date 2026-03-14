import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def apply_sinusoidal_forces_xy(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 10.0,
    frequency: float = 0.5,  # Hz
    on_duration: float = 2.0,  # seconds forces are applied
    off_duration: float = 1.0,  # seconds forces are zero (pause)
    randomize_bodies: bool = False,
):
    """Apply sinusoidal forces on all 3 axes with independent phases per axis,
    with intermittent on/off duty cycle per environment.

    During the "on" mode forces follow:
        F_i = amplitude * sin(2*pi*freq*t + phase_i)
    During the "off" mode forces are zero.

    Each environment gets a random phase offset into the duty cycle so they
    don't all pause at the same time.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, forces applied to all envs).
        asset_cfg: Asset and body configuration.
        force_amplitude: Force amplitude in Newtons. Either a single float
            (same for all bodies) or a list of floats (one per body in
            asset_cfg.body_names).
        frequency: Oscillation frequency in Hz.
        on_duration: How long (seconds) forces are applied per cycle.
        off_duration: How long (seconds) forces are paused per cycle.
        randomize_bodies: If True, each step randomly selects between 1 and
            num_bodies-1 bodies to apply forces to (never all at once).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    cycle_period = on_duration + off_duration

    # Initialize phase buffers on first call
    # independent phase per env, body, axis
    if not hasattr(env, "_sin_force_phases"):
        env._sin_force_phases = torch.rand(
            (num_envs, num_bodies, 3), device=device
        ) * 2 * torch.pi
    # Random duty-cycle offset per env so pauses are desynchronized
    if not hasattr(env, "_duty_cycle_offset"):
        env._duty_cycle_offset = torch.rand(num_envs, device=device) * cycle_period

    # Re-randomize phases for environments that just reset
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        env._sin_force_phases[reset_ids] = torch.rand(
            (len(reset_ids), num_bodies, 3), device=device
        ) * 2 * torch.pi
        env._duty_cycle_offset[reset_ids] = torch.rand(
            len(reset_ids), device=device
        ) * cycle_period

    # Global simulation time
    t = env.common_step_counter * env.step_dt

    # Per-env time within the duty cycle: [num_envs]
    cycle_time = (t + env._duty_cycle_offset) % cycle_period
    # 1.0 during on-window, 0.0 during off-window: [num_envs, 1, 1]
    on_mask = (cycle_time < on_duration).float().unsqueeze(-1).unsqueeze(-1)

    # Build per-body amplitude tensor: [1, num_bodies, 1]
    if isinstance(force_amplitude, (list, tuple)):
        amp = torch.tensor(force_amplitude, device=device).view(1, num_bodies, 1)
    else:
        amp = force_amplitude

    # Random body selection mask: [num_envs, num_bodies, 1]
    if randomize_bodies and num_bodies > 1:
        # Random count per env: k in [1, num_bodies - 1]
        k = torch.randint(1, num_bodies, (num_envs,), device=device)
        # Assign random scores, rank them to pick top-k per env
        scores = torch.rand(num_envs, num_bodies, device=device)
        ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
        body_mask = (ranks < k.unsqueeze(1)).float().unsqueeze(-1)
    else:
        body_mask = 1.0

    # Compute sinusoidal forces: [num_envs, num_bodies, 3]
    forces = amp * torch.sin(
        2 * torch.pi * frequency * t + env._sin_force_phases
    )
    # Zero out Z-axis forces (only apply in XY plane)
    forces[:, :, 2] = 0.0
    # Apply duty cycle and body activation masks
    forces = forces * on_mask * body_mask
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )


def apply_sinusoidal_forces(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 10.0,
    frequency: float = 0.5,
    on_duration: float = 2.0,
    off_duration: float = 1.0,
    z_scale: float = 0.5,
    randomize_bodies: bool = False,
):
    """Apply sinusoidal forces on all 3 axes. Z amplitude is scaled by z_scale.

    Args:
        force_amplitude: XY amplitude in Newtons (per body).
        z_scale: Multiplier for Z-axis amplitude relative to XY (default 0.5).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    cycle_period = on_duration + off_duration

    if not hasattr(env, "_sin_force_xyz_phases"):
        env._sin_force_xyz_phases = torch.rand(
            (num_envs, num_bodies, 3), device=device
        ) * 2 * torch.pi
    if not hasattr(env, "_duty_cycle_xyz_offset"):
        env._duty_cycle_xyz_offset = torch.rand(num_envs, device=device) * cycle_period

    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        env._sin_force_xyz_phases[reset_ids] = torch.rand(
            (len(reset_ids), num_bodies, 3), device=device
        ) * 2 * torch.pi
        env._duty_cycle_xyz_offset[reset_ids] = torch.rand(
            len(reset_ids), device=device
        ) * cycle_period

    t = env.common_step_counter * env.step_dt

    cycle_time = (t + env._duty_cycle_xyz_offset) % cycle_period
    on_mask = (cycle_time < on_duration).float().unsqueeze(-1).unsqueeze(-1)

    if isinstance(force_amplitude, (list, tuple)):
        amp = torch.tensor(force_amplitude, device=device).view(1, num_bodies, 1)
    else:
        amp = force_amplitude

    if randomize_bodies and num_bodies > 1:
        k = torch.randint(1, num_bodies, (num_envs,), device=device)
        scores = torch.rand(num_envs, num_bodies, device=device)
        ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
        body_mask = (ranks < k.unsqueeze(1)).float().unsqueeze(-1)
    else:
        body_mask = 1.0

    forces = amp * torch.sin(
        2 * torch.pi * frequency * t + env._sin_force_xyz_phases
    )
    # Scale Z amplitude
    forces[:, :, 2] = forces[:, :, 2] * z_scale
    forces = forces * on_mask * body_mask
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )


def apply_constant_force_z(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_z: float = -70.0,
):
    """Apply a constant downward force on the Z axis to specified bodies.

    Args:
        env: The environment instance.
        env_ids: Environment indices.
        asset_cfg: Asset and body configuration.
        force_z: Constant force in Newtons along Z axis.
            Negative = downward (default -70.0 N).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 2] = force_z
    torques = torch.zeros_like(forces)
    # print("forces", forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )


def apply_sinusoidal_forces_z(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 10.0,
    frequency: float = 0.5,
    on_duration: float = 2.0,
    off_duration: float = 1.0,
):
    """Apply sinusoidal forces on Z axis only, with duty cycle."""
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    cycle_period = on_duration + off_duration

    # Initialize phase buffer (single phase per env per body, Z only)
    if not hasattr(env, "_sin_force_z_phases"):
        env._sin_force_z_phases = torch.rand(
            (num_envs, num_bodies), device=device
        ) * 2 * torch.pi
    if not hasattr(env, "_duty_cycle_z_offset"):
        env._duty_cycle_z_offset = torch.rand(num_envs, device=device) * cycle_period

    t = env.common_step_counter * env.step_dt

    cycle_time = (t + env._duty_cycle_z_offset) % cycle_period
    on_mask = (cycle_time < on_duration).float().unsqueeze(-1)  # [num_envs, 1]

    if isinstance(force_amplitude, (list, tuple)):
        amp = torch.tensor(force_amplitude, device=device).view(1, num_bodies)
    else:
        amp = force_amplitude

    # Sinusoidal force magnitude: [num_envs, num_bodies]
    fz = amp * torch.sin(
        2 * torch.pi * frequency * t + env._sin_force_z_phases
    ) * on_mask

    # Build [num_envs, num_bodies, 3] with only Z component
    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 2] = fz
    torques = torch.zeros_like(forces)
    # print(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )
