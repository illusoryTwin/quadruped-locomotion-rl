"""Soft compliance manager using Mass-Spring-Damper model."""

import re
import torch

from compliance.compliance_manager_cfg import ComplianceManagerCfg
from compliance.utils.mass_spring_damper_model import MassSpringDamperModel
from compliance.utils.dynamics import calculate_external_torques, create_joint_mask


class ComplianceManager:
    """Manager for soft joint compliance using Mass-Spring-Damper model."""

    def __init__(self, cfg: ComplianceManagerCfg, env):
        self.cfg = cfg
        self._env = env
        self._robot = env.scene[cfg.robot_name]
        self._device = env.device
        self._num_envs = env.num_envs

        # Monitored bodies - we apply forces on these bodies
        self._monitored_body_indices = self._expand_body_indices(cfg.monitored_bodies)
        self._monitored_body_names = [
            self._robot.body_names[i] for i in self._monitored_body_indices
        ]

        # Active joints - derived from stiffness_config keys (also supports regex)
        self._active_joint_indices = self._expand_joint_indices(list(cfg.stiffness_config.keys()))

        # Build expanded stiffness config with actual joint indices
        self._stiffness_scales = self._build_stiffness_scales(cfg.stiffness_config)

        # Create joint mask for active joints
        self._joint_mask = create_joint_mask(
            num_joints=self._robot.num_joints,
            active_joint_indices=self._active_joint_indices,
            fix_base=True,
            device=self._device,
        )

        self._msd_system = self._setup_msd_system()
        self._deformations = None

        print(f"[ComplianceManager] Monitored bodies: {self._monitored_body_names}")
        print(f"[ComplianceManager] Active joints: {[self._robot.joint_names[i] for i in self._active_joint_indices]}")

    def _expand_body_indices(self, body_patterns: list) -> list:
        """Expand regex patterns to actual body indices."""
        indices = []
        for pattern in body_patterns:
            regex = re.compile(pattern)
            for i, name in enumerate(self._robot.body_names):
                if regex.fullmatch(name) and i not in indices:
                    indices.append(i)
        return indices

    def _expand_joint_indices(self, joint_patterns: list) -> list:
        """Expand regex patterns to actual joint indices."""
        indices = []
        for pattern in joint_patterns:
            regex = re.compile(pattern)
            for i, name in enumerate(self._robot.joint_names):
                if regex.fullmatch(name) and i not in indices:
                    indices.append(i)
        return sorted(indices)

    def _build_stiffness_scales(self, stiffness_config: dict) -> dict:
        """Build stiffness scales dict mapping joint index -> scale."""
        scales = {}
        for pattern, scale in stiffness_config.items():
            regex = re.compile(pattern)
            for i, name in enumerate(self._robot.joint_names):
                if regex.fullmatch(name):
                    scales[i] = scale
        return scales

    def _setup_msd_system(self) -> MassSpringDamperModel:
        cfg = self.cfg

        # Create and return MSD system
        return MassSpringDamperModel(
            n_dofs=self._robot.num_joints,
            dt=cfg.dt,
            base_inertia=cfg.base_inertia,
            base_stiffness=cfg.base_stiffness,
            stiffness_scales=self._stiffness_scales,
            num_envs=self._num_envs,
            device=self._device,
        )

    @property
    def active_joint_indices(self):
        """Return list of active joint indices."""
        return self._active_joint_indices

    @property
    def monitored_body_indices(self):
        """Return list of monitored body indices."""
        return self._monitored_body_indices

    def reset(self, env_ids: torch.Tensor = None):
        """Reset MSD state for specified environments."""
        if self._msd_system is not None:
            self._msd_system.reset(env_ids)

    def compute(self, dt: float) -> torch.Tensor:
        """Compute joint deformations from external forces.

        Args:
            dt: Time step (unused, MSD uses its own dt from config)

        Returns:
            Joint deformations [num_envs, num_active_joints]
        """
        if len(self._active_joint_indices) == 0:
            return torch.zeros((self._num_envs, 0), device=self._device)

        # Calculate joint torques from external forces using verified function
        joint_torques = calculate_external_torques(
            robot=self._robot,
            body_names=self._monitored_body_names,
            joint_mask=self._joint_mask,
            verbose=False,
        )

        # Update MSD system and get deformations
        self._msd_system.update_msd_state_discrete(joint_torques)

        # Get deformations for active DOFs
        deformations = self._msd_system.state['q_def']

        self._deformations = deformations

        if self.cfg.debug:
            print(f"[ComplianceManager] Deformations: {deformations[0]}")

        return deformations
