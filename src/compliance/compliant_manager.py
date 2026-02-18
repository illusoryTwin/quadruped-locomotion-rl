"""Soft compliance manager using Mass-Spring-Damper model."""

import re
import torch

from compliance.compliance_manager_cfg import ComplianceManagerCfg
from compliance.utils.mass_spring_damper_model import MassSpringDamperModel
from compliance.utils.dynamics import calculate_external_torques, create_joint_mask, get_wrench, get_jacobians


class ComplianceManager:
    """Manager for soft joint compliance using Mass-Spring-Damper model."""

    def __init__(self, cfg: ComplianceManagerCfg, env):
        self.cfg = cfg
        self._env = env
        self._robot = env.scene[cfg.robot_name]
        self._device = env.device
        self._num_envs = env.num_envs

        # Compliant bodies
        self._compliant_body_names = list(cfg.compliant_bodies.keys())
        self._compliant_body_stiffness = cfg.compliant_bodies 
        self._compliant_body_stiffness_cartesian = self._build_cartesian_stiffness_scales()

        self._joint_mask = create_joint_mask(
            num_joints=self._robot.num_joints,
            active_joint_indices=list(range(self._robot.num_joints)),
            fix_base=False,
            device=self._device,
        ) 

        self._msd_system = self._setup_msd_system()
        self._deformations = None

        print(f"[ComplianceManager] Monitored bodies: {self._compliant_body_names}")


    def _build_cartesian_stiffness_scales(self):
        cartesian_scales = {}
        for i, body_name in enumerate(self._compliant_body_names):
            scale = self._compliant_body_stiffness.get(body_name, 1.0)
            for axis in range(3):
                cartesian_scales[i * 3 + axis] = scale
 
        return cartesian_scales

    def _setup_msd_system(self) -> MassSpringDamperModel:
        cfg = self.cfg
        n_bodies = len(self._compliant_body_names)

        # Create and return MSD system
        return MassSpringDamperModel(
            n_dofs=n_bodies*3,
            dt=cfg.dt,
            base_inertia=cfg.base_inertia,
            base_stiffness=cfg.base_stiffness,
            stiffness_scales=self._compliant_body_stiffness_cartesian,
            num_envs=self._num_envs,
            device=self._device,
        )

    def reset(self, env_ids: torch.Tensor = None):
        """Reset MSD state for specified environments."""
        if self._msd_system is not None:
            self._msd_system.reset(env_ids)

    def compute(self, dt: float, base_stiffness: torch.Tensor | None = None) -> torch.Tensor:
        """Compute joint deformations from external forces.

        Args:
            dt: Time step (unused, MSD uses its own dt from config)
            base_stiffness: Per-env base stiffness
                If provided, uses analytical per-env MSD update.
                If None, uses precomputed matrices with fixed stiffness from config.

        Returns:
            Joint deformations [num_envs, num_active_joints]
        """
        if len(self._compliant_body_names) == 0:
            return torch.zeros((self._num_envs, 0), device=self._device)

        # external forces
        wrench = get_wrench(self._robot, self._compliant_body_names)
        forces = wrench[:, :, :3]
        forces_flat = forces.reshape(forces.shape[0], -1)

        # Update MSD system and get deformations
        if base_stiffness is not None:
            self._msd_system.update_with_variable_stiffness(forces_flat, base_stiffness)
        else:
            self._msd_system.update_msd_state_discrete(forces_flat)

        # Get deformations for active DOFs
        x_def = self._msd_system.state['x_def']
        # x_def = x_def.reshape(num_envs, n_bodies, 3)
        x_def_3d = x_def.reshape(self._num_envs, len(self._compliant_body_names), 3)

        J = get_jacobians(self._robot, body_names=self._compliant_body_names, joint_mask=self._joint_mask)[:, :, :3, :]
        J_pinv = torch.linalg.pinv(J)
        q_def = torch.einsum('ebnj,ebj->en', J_pinv, x_def_3d)
        q_def = q_def[:, 6:] # drop 6 floating-base DOFs
        q_def = q_def.clamp(-self.cfg.max_deformation, self.cfg.max_deformation)

        self._deformations = q_def

        if self.cfg.debug:
            print(f"[ComplianceManager] Deformations: {q_def[0]}")

        return q_def
