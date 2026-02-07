"""Command term that generates base stiffness (kp) for compliance randomization."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StiffnessCommand(CommandTerm):
    """Command term that samples base stiffness (kp) uniformly from a range.

    The sampled stiffness is used by the compliance manager to set the MSD
    spring constant: K_joint = kp * scale_joint.
    """

    cfg: StiffnessCommandCfg

    def __init__(self, cfg: StiffnessCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.stiffness_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The stiffness command. Shape is (num_envs, 1)."""
        return self.stiffness_command

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.stiffness_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.kp)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class StiffnessCommandCfg(CommandTermCfg):
    """Configuration for the stiffness command generator."""

    class_type: type = StiffnessCommand

    @configclass
    class Ranges:
        """Ranges for stiffness sampling."""

        kp: tuple[float, float] = (5.0, 20.0)
        """Range for base stiffness [min, max]."""

    ranges: Ranges = Ranges()
