import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def apply_sinusoidal_forces(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float = 10.0,
    frequency: float = 0.5,  # Hz
    on_duration: float = 2.0,  # seconds forces are applied
    off_duration: float = 1.0,  # seconds forces are zero (pause)
):
    """Apply sinusoidal forces on all 3 axes with independent phases per axis,
    with intermittent on/off duty cycle per environment.

    During the "on" window forces follow:
        F_i = amplitude * sin(2*pi*freq*t + phase_i)
    During the "off" window forces are zero.

    Each environment gets a random phase offset into the duty cycle so they
    don't all pause at the same time.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, forces applied to all envs).
        asset_cfg: Asset and body configuration.
        force_amplitude: Force amplitude in Newtons.
        frequency: Oscillation frequency in Hz.
        on_duration: How long (seconds) forces are applied per cycle.
        off_duration: How long (seconds) forces are paused per cycle.
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

    # Compute sinusoidal forces: [num_envs, num_bodies, 3]
    forces = force_amplitude * torch.sin(
        2 * torch.pi * frequency * t + env._sin_force_phases
    )
    # Apply duty cycle mask
    forces = forces * on_mask
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )
