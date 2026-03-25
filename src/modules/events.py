git diff import torch
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
    
    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )

    # Store for external logging
    env._compliance_push_fz = fz


def apply_sinusoidal_forces_xy_push(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 10.0,
    frequency: float = 0.5,
    on_duration: float = 2.0,
    off_duration: float = 1.0,
):
    """Apply sinusoidal forces on X and Y axes only, with duty cycle.

    Each environment gets independent random phases for X and Y, plus a
    random offset into the duty cycle so on/off windows are desynchronized.

    Args:
        env: The environment instance.
        env_ids: Environment indices.
        asset_cfg: Asset and body configuration.
        force_amplitude: Force amplitude in Newtons (same for X and Y).
        frequency: Oscillation frequency in Hz.
        on_duration: Seconds forces are applied per cycle.
        off_duration: Seconds forces are zero per cycle.
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

    # Initialize phase buffers: independent phase per env, per body, for X and Y
    if not hasattr(env, "_sin_force_xy_phases"):
        env._sin_force_xy_phases = torch.rand(
            (num_envs, num_bodies, 2), device=device
        ) * 2 * torch.pi
    if not hasattr(env, "_duty_cycle_xy_offset"):
        env._duty_cycle_xy_offset = torch.rand(num_envs, device=device) * cycle_period

    t = env.common_step_counter * env.step_dt

    # Duty cycle mask: [num_envs, 1]
    cycle_time = (t + env._duty_cycle_xy_offset) % cycle_period
    on_mask = (cycle_time < on_duration).float().unsqueeze(-1)

    if isinstance(force_amplitude, (list, tuple)):
        amp = torch.tensor(force_amplitude, device=device).view(1, num_bodies)
    else:
        amp = force_amplitude

    # Sinusoidal force for X and Y: [num_envs, num_bodies, 2]
    fxy = amp.unsqueeze(-1) if isinstance(amp, torch.Tensor) and amp.dim() == 2 else amp
    fxy = torch.stack([
        amp * torch.sin(2 * torch.pi * frequency * t + env._sin_force_xy_phases[:, :, 0]),
        amp * torch.sin(2 * torch.pi * frequency * t + env._sin_force_xy_phases[:, :, 1]),
    ], dim=-1)  # [num_envs, num_bodies, 2]

    # Apply duty cycle
    fxy = fxy * on_mask.unsqueeze(-1)

    # Build [num_envs, num_bodies, 3] with only X and Y components
    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 0] = fxy[:, :, 0]
    forces[:, :, 1] = fxy[:, :, 1]
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )

    # Store for external logging
    env._compliance_push_fxy = fxy


def apply_sinusoidal_forces_z_new(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 10.0,
    frequency_range: tuple[float, float] = (0.1, 0.5),
    on_duration: float = 2.0,
    off_duration: float = 1.0,
    resample_time: float = 10.0,
):
    """Apply sinusoidal forces on Z axis only.

    Compared to apply_sinusoidal_forces_z:
      - Per-env episode time (resets on env reset) instead of global sim time.
      - Per-env random frequency drawn from frequency_range instead of a single fixed value.
      - Phase and frequency are resampled every resample_time seconds within each episode.
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

    # Initialize buffers on first call
    if not hasattr(env, "_sin_force_z_new_phases"):
        env._sin_force_z_new_phases = torch.rand(
            (num_envs, num_bodies), device=device
        ) * 2 * torch.pi
    if not hasattr(env, "_sin_force_z_new_freqs"):
        env._sin_force_z_new_freqs = torch.empty(
            (num_envs, num_bodies), device=device
        ).uniform_(frequency_range[0], frequency_range[1])
    if not hasattr(env, "_duty_cycle_z_new_offset"):
        env._duty_cycle_z_new_offset = torch.rand(num_envs, device=device) * cycle_period
    if not hasattr(env, "_sin_force_z_new_resample_step"):
        env._sin_force_z_new_resample_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Per-env episode time: resets to 0 when env resets
    episode_steps = env.episode_length_buf  # [num_envs] int tensor
    t = (episode_steps * env.step_dt).unsqueeze(-1)  # [num_envs, 1]

    # Resample phase & frequency for envs that crossed the resample boundary
    resample_interval_steps = int(resample_time / env.step_dt)
    needs_resample = (episode_steps - env._sin_force_z_new_resample_step) >= resample_interval_steps
    resample_ids = needs_resample.nonzero(as_tuple=False).squeeze(-1)
    if resample_ids.numel() > 0:
        env._sin_force_z_new_phases[resample_ids] = (
            torch.rand((resample_ids.numel(), num_bodies), device=device) * 2 * torch.pi
        )
        env._sin_force_z_new_freqs[resample_ids] = torch.empty(
            (resample_ids.numel(), num_bodies), device=device
        ).uniform_(frequency_range[0], frequency_range[1])
        env._sin_force_z_new_resample_step[resample_ids] = episode_steps[resample_ids]

    # Reset resample counter for envs that just started a new episode (episode_steps == 0)
    just_reset = (episode_steps == 0)
    reset_ids = just_reset.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._sin_force_z_new_resample_step[reset_ids] = 0

    # Duty cycle
    cycle_time = (t.squeeze(-1) + env._duty_cycle_z_new_offset) % cycle_period
    on_mask = (cycle_time < on_duration).float().unsqueeze(-1)  # [num_envs, 1]

    if isinstance(force_amplitude, (list, tuple)):
        amp = torch.tensor(force_amplitude, device=device).view(1, num_bodies)
    else:
        amp = force_amplitude

    # Sinusoidal force magnitude: [num_envs, num_bodies]
    fz = amp * torch.sin(
        2 * torch.pi * env._sin_force_z_new_freqs * t + env._sin_force_z_new_phases
    ) * on_mask

    # Build [num_envs, num_bodies, 3] with only Z component
    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 2] = fz
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )

    # Store for external logging
    env._compliance_push_fz = fz


def log_env0_compliance(
    env,
    env_ids: torch.Tensor,
    log_path: str = "env0_compliance_log.csv",
):
    """Log applied force and MSD deformation for env[0] to a CSV file."""
    import csv

    if not hasattr(env, "_env0_log_writer"):
        f = open(log_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["step", "sim_time", "force_z", "x_def_x", "x_def_y", "x_def_z"])
        env._env0_log_file = f
        env._env0_log_writer = writer

    # Applied force
    fz = 0.0
    if hasattr(env, "_compliance_push_fz"):
        fz = env._compliance_push_fz[0, 0].item()

    # MSD deformation
    x_def = [0.0, 0.0, 0.0]
    if hasattr(env, "compliance_manager") and env.compliance_manager is not None:
        msd = env.compliance_manager._msd_system
        if msd is not None:
            x_def = msd.state["x_def"][0, 0:3].cpu().tolist()

    t = env.common_step_counter * env.step_dt
    env._env0_log_writer.writerow([
        env.common_step_counter,
        f"{t:.4f}",
        f"{fz:.4f}",
        f"{x_def[0]:.6f}",
        f"{x_def[1]:.6f}",
        f"{x_def[2]:.6f}",
    ])

    if env.common_step_counter % 500 == 0:
        env._env0_log_file.flush()
