"""Command term that generates a target base position (relative to env origin)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BasePositionCommand(CommandTerm):
    """Command term that samples a target XYZ position offset from the env origin.

    The compliant reference is then: env_origin + command + MSD deformation.
    """

    cfg: BasePositionCommandCfg

    def __init__(self, cfg: BasePositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.pos_command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The position command. Shape is (num_envs, 3)."""
        return self.pos_command

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        self.pos_command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.x)
        self.pos_command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.y)
        self.pos_command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.z)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class BasePositionCommandCfg(CommandTermCfg):
    """Configuration for the base position command generator."""

    class_type: type = BasePositionCommand

    @configclass
    class Ranges:
        """Ranges for position sampling (relative to env origin)."""

        x: tuple[float, float] = (0.0, 0.0)
        y: tuple[float, float] = (0.0, 0.0)
        z: tuple[float, float] = (0.3, 0.3)

    ranges: Ranges = Ranges()
