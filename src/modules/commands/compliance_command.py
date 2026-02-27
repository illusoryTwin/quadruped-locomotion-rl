"""Command term that wraps ComplianceManager to produce joint deformations."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.utils import configclass

from src.compliance.compliance_manager import ComplianceManager
from src.compliance.compliance_manager_cfg import ComplianceManagerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ComplianceCommand(CommandTerm):
    """Command that computes compliance deformations via MSD model.

    On each step, reads external forces from compliant bodies,
    runs the MSD integrator, and exposes joint deformations via
    the ``command`` property.
    """

    cfg: ComplianceCommandCfg

    def __init__(self, cfg: ComplianceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._compliance_manager = ComplianceManager(cfg.compliance_cfg, env)

        # Expose on env so StiffnessCommand and observations keep working
        self._env.compliance_manager = self._compliance_manager

        robot = env.scene[cfg.compliance_cfg.robot_name]
        self._deformations = torch.zeros(
            self.num_envs, robot.num_joints, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Deformations tensor [num_envs, num_joints]."""
        return self._deformations

    @property
    def compliance_manager(self) -> ComplianceManager:
        return self._compliance_manager

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset MSD state on episode reset."""
        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._compliance_manager.reset(env_ids_t)
        self._deformations[env_ids] = 0.0

    def _update_command(self):
        """Compute new deformations from current external forces."""
        # Read per-env stiffness from stiffness command if available
        base_stiffness = None
        try:
            kp_cmd = self._env.command_manager.get_command("stiffness")
            base_stiffness = kp_cmd[:, 0]  # [num_envs]
        except (KeyError, RuntimeError):
            pass

        self._deformations = self._compliance_manager.compute(base_stiffness=base_stiffness)
        # Also set on env so reward functions can read env._deformations
        self._env._deformations = self._deformations

    def _update_metrics(self):
        pass


@configclass
class ComplianceCommandCfg(CommandTermCfg):
    """Configuration for the compliance command generator."""

    class_type: type = ComplianceCommand
    compliance_cfg: ComplianceManagerCfg = ComplianceManagerCfg()
