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


def apply_random_constant_force_z(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 70.0,
    hold_time_range: tuple[float, float] = (2.0, 6.0),
):
    """Apply a random constant Z force per env, held for a random duration then resampled.

    Each environment independently samples a force from [-force_amplitude, +force_amplitude]
    and a hold duration from hold_time_range. When the hold expires, both are resampled.

    Args:
        env: The environment instance.
        env_ids: Environment indices.
        asset_cfg: Asset and body configuration.
        force_amplitude: Max force magnitude in Newtons. Force is sampled
            uniformly from [-amplitude, +amplitude].
        hold_time_range: (min_seconds, max_seconds) for how long each
            force is held before resampling.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    if isinstance(force_amplitude, (list, tuple)):
        amp = force_amplitude[0]
    else:
        amp = force_amplitude

    # Initialize buffers on first call
    if not hasattr(env, "_rand_force_z_values"):
        env._rand_force_z_values = (2 * torch.rand(num_envs, device=device) - 1) * amp
        env._rand_force_z_end_time = (
            torch.rand(num_envs, device=device) * (hold_time_range[1] - hold_time_range[0])
            + hold_time_range[0]
        )
        env._rand_force_z_start_time = torch.zeros(num_envs, device=device)

    t = env.common_step_counter * env.step_dt

    # Check which envs need resampling (hold expired)
    elapsed = t - env._rand_force_z_start_time
    resample_mask = elapsed >= env._rand_force_z_end_time
    n_resample = resample_mask.sum().item()

    if n_resample > 0:
        env._rand_force_z_values[resample_mask] = (
            (2 * torch.rand(n_resample, device=device) - 1) * amp
        )
        env._rand_force_z_end_time[resample_mask] = (
            torch.rand(n_resample, device=device) * (hold_time_range[1] - hold_time_range[0])
            + hold_time_range[0]
        )
        env._rand_force_z_start_time[resample_mask] = t

    # Build force tensor
    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 2] = env._rand_force_z_values.unsqueeze(-1)
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )

    # Store for external logging
    env._compliance_push_fz = env._rand_force_z_values.unsqueeze(-1)


def apply_random_constant_force_xy(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float | list[float] = 30.0,
    hold_time_range: tuple[float, float] = (2.0, 6.0),
):
    """Apply random constant X and Y forces per env, held for a random duration then resampled.

    Each environment independently samples Fx and Fy from [-force_amplitude, +force_amplitude]
    and a hold duration from hold_time_range. When the hold expires, force components
    and the hold duration are resampled together.

    Args:
        env: The environment instance.
        env_ids: Environment indices.
        asset_cfg: Asset and body configuration.
        force_amplitude: Max force magnitude in Newtons. Fx and Fy are each sampled
            uniformly from [-amplitude, +amplitude].
        hold_time_range: (min_seconds, max_seconds) for how long each
            force is held before resampling.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    if isinstance(force_amplitude, (list, tuple)):
        amp = force_amplitude[0]
    else:
        amp = force_amplitude

    # Initialize buffers on first call
    if not hasattr(env, "_rand_force_xy_values"):
        env._rand_force_xy_values = (2 * torch.rand(num_envs, 2, device=device) - 1) * amp
        env._rand_force_xy_end_time = (
            torch.rand(num_envs, device=device) * (hold_time_range[1] - hold_time_range[0])
            + hold_time_range[0]
        )
        env._rand_force_xy_start_time = torch.zeros(num_envs, device=device)

    t = env.common_step_counter * env.step_dt

    # Check which envs need resampling (hold expired)
    elapsed = t - env._rand_force_xy_start_time
    resample_mask = elapsed >= env._rand_force_xy_end_time
    n_resample = resample_mask.sum().item()

    if n_resample > 0:
        env._rand_force_xy_values[resample_mask] = (
            (2 * torch.rand(n_resample, 2, device=device) - 1) * amp
        )
        env._rand_force_xy_end_time[resample_mask] = (
            torch.rand(n_resample, device=device) * (hold_time_range[1] - hold_time_range[0])
            + hold_time_range[0]
        )
        env._rand_force_xy_start_time[resample_mask] = t

    # Build force tensor: broadcast per-env XY force across bodies
    forces = torch.zeros(num_envs, num_bodies, 3, device=device)
    forces[:, :, 0] = env._rand_force_xy_values[:, 0:1]
    forces[:, :, 1] = env._rand_force_xy_values[:, 1:2]
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )

    # Store for external logging
    env._compliance_push_fxy = env._rand_force_xy_values


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
    max_learning_iterations: int | None = None,
    num_steps_per_env: int | None = None,
    log_last_learning_iterations: int | None = None,
    extended_msd_log: bool = False,
    enabled: bool = False,
):
    """Log applied force and MSD deformation for env[0] to a CSV file.

    When ``enabled`` is False (default), the event is a no-op so training stays free of CSV I/O.

    Optional gating (RL iterations, RSL-RL style): if ``max_learning_iterations``,
    ``num_steps_per_env``, and ``log_last_learning_iterations`` are all set, rows
    are written only after env step index reaches
    ``(max_learning_iterations - log_last_learning_iterations) * num_steps_per_env``.
    Each learning iteration performs ``num_steps_per_env`` vectorized env steps; the
    environment step counter advances once per ``env.step`` call inside the rollout loop.

    If ``extended_msd_log`` is True, the wide MSD column layout is used from step 0 for
    the whole run (no gating). When gating is enabled, the same wide layout is used after
    the gate opens. Otherwise the legacy 6-column layout is used.
    """
    if not enabled:
        return
    import csv
    import math

    gated = (
        max_learning_iterations is not None
        and num_steps_per_env is not None
        and log_last_learning_iterations is not None
    )
    if gated:
        start_step = (max_learning_iterations - log_last_learning_iterations) * num_steps_per_env
        if int(env.common_step_counter) < int(start_step):
            return

    use_extended = bool(gated or extended_msd_log)

    if not hasattr(env, "_env0_log_writer"):
        f = open(log_path, "w", newline="")
        writer = csv.writer(f)
        if use_extended:
            writer.writerow(
                [
                    "step",
                    "sim_time",
                    "approx_learning_iter",
                    "kp",
                    "K_base",
                    "omega_base",
                    "msd_dt",
                    "M",
                    "u_base_x",
                    "u_base_y",
                    "u_base_z",
                    "force_z_event",
                    "x_def_x",
                    "x_def_y",
                    "x_def_z",
                    "dx_def_x",
                    "dx_def_y",
                    "dx_def_z",
                ]
            )
        else:
            writer.writerow(["step", "sim_time", "force_z", "x_def_x", "x_def_y", "x_def_z"])
        env._env0_log_file = f
        env._env0_log_writer = writer
        env._env0_log_extended = use_extended
        env._env0_log_num_steps_per_env = num_steps_per_env if (extended_msd_log or gated) else None

    fz = 0.0
    if hasattr(env, "_compliance_push_fz"):
        fz = env._compliance_push_fz[0, 0].item()

    x_def = [0.0, 0.0, 0.0]
    dx_def = [0.0, 0.0, 0.0]
    u_base = [0.0, 0.0, 0.0]
    kp = 0.0
    K_base = 0.0
    omega_base = 0.0
    msd_dt = 0.0
    M_cfg = 0.0
    approx_it = 0
    if hasattr(env, "compliance_manager") and env.compliance_manager is not None:
        cm = env.compliance_manager
        cfg = cm.cfg
        M_cfg = float(cfg.base_inertia)
        msd_dt = float(cfg.dt)
        if hasattr(env, "command_manager"):
            try:
                kp = float(env.command_manager.get_command("stiffness")[0, 0].item())
            except Exception:
                kp = float(cfg.base_stiffness)
        else:
            kp = float(cfg.base_stiffness)
        scale_base = float(cfg.compliant_bodies.get("base", 1.0))
        K_base = kp * scale_base
        if K_base > 0.0 and M_cfg > 0.0:
            omega_base = float(math.sqrt(K_base / M_cfg))
        try:
            from src.compliance.utils.dynamics import get_wrench

            names = list(cm._compliant_body_names)
            bi = int(names.index("base")) if "base" in names else 0
            wrench = get_wrench(cm._robot, names)
            u_base = wrench[0, bi, :3].detach().cpu().tolist()
        except Exception:
            u_base = [0.0, 0.0, 0.0]
        msd = cm._msd_system
        if msd is not None:
            x_def = msd.state["x_def"][0, 0:3].detach().cpu().tolist()
            dx_def = msd.state["dx_def"][0, 0:3].detach().cpu().tolist()
    nsp = getattr(env, "_env0_log_num_steps_per_env", None)
    if isinstance(nsp, int) and nsp > 0:
        approx_it = int(env.common_step_counter) // int(nsp)

    t = env.common_step_counter * env.step_dt
    if getattr(env, "_env0_log_extended", False):
        env._env0_log_writer.writerow(
            [
                env.common_step_counter,
                f"{t:.4f}",
                approx_it,
                f"{kp:.6f}",
                f"{K_base:.6f}",
                f"{omega_base:.6f}",
                f"{msd_dt:.6f}",
                f"{M_cfg:.6f}",
                f"{u_base[0]:.6f}",
                f"{u_base[1]:.6f}",
                f"{u_base[2]:.6f}",
                f"{fz:.4f}",
                f"{x_def[0]:.6f}",
                f"{x_def[1]:.6f}",
                f"{x_def[2]:.6f}",
                f"{dx_def[0]:.6f}",
                f"{dx_def[1]:.6f}",
                f"{dx_def[2]:.6f}",
            ]
        )
    else:
        env._env0_log_writer.writerow(
            [
                env.common_step_counter,
                f"{t:.4f}",
                f"{fz:.4f}",
                f"{x_def[0]:.6f}",
                f"{x_def[1]:.6f}",
                f"{x_def[2]:.6f}",
            ]
        )

    if env.common_step_counter % 500 == 0:
        env._env0_log_file.flush()
