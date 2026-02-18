import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def apply_sinusoidal_forces(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float = 10.0,
    frequency: float = 0.5,  # Hz
):
    """Apply sinusoidal forces on all 3 axes with independent phases per axis.

    Force per axis: F_i = amplitude * sin(2*pi*freq*t + phase_i)

    Writes to the standard Isaac Lab physics buffer via set_external_force_and_torque(),
    so the ComplianceManager can read them from robot._external_force_b.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused, forces applied to all envs).
        asset_cfg: Asset and body configuration.
        force_amplitude: Force amplitude in Newtons.
        frequency: Oscillation frequency in Hz.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.num_envs
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    # Initialize phase buffers on first call
    # independent phase per env, body, axis
    if not hasattr(env, "_sin_force_phases"):
        env._sin_force_phases = torch.rand(
            (num_envs, num_bodies, 3), device=device
        ) * 2 * torch.pi

    # Re-randomize phases for environments that just reset
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        env._sin_force_phases[reset_ids] = torch.rand(
            (len(reset_ids), num_bodies, 3), device=device
        ) * 2 * torch.pi

    # Global simulation time
    t = env.common_step_counter * env.step_dt

    # Compute sinusoidal forces: [num_envs, num_bodies, 3]
    forces = force_amplitude * torch.sin(
        2 * torch.pi * frequency * t + env._sin_force_phases
    )
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
    )
