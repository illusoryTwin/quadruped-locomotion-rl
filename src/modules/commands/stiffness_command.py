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


def discrete_kp_values(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("discrete_step must be positive")
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    n = int(round(span / step))
    tol = 1e-5 * max(abs(lo), abs(hi), 1.0)
    if abs(span - n * step) > tol:
        raise ValueError("Stiffness kp range width must be an integer multiple of discrete_step")
    return [lo + float(i) * step for i in range(n + 1)]


class StiffnessCommand(CommandTerm):
    """Command term that samples base stiffness (kp) uniformly from a range.

    The sampled stiffness is used by the compliance manager to set the MSD
    spring constant: K_joint = kp * scale_joint.
    """

    cfg: StiffnessCommandCfg

    def __init__(self, cfg: StiffnessCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.stiffness_command = torch.zeros(self.num_envs, 1, device=self.device)
        lo, hi = cfg.ranges.kp
        if cfg.discrete_step is not None:
            values = discrete_kp_values(lo, hi, cfg.discrete_step)
            self._kp_grid = torch.tensor(values, device=self.device, dtype=torch.float32)
        else:
            self._kp_grid = None

    @property
    def command(self) -> torch.Tensor:
        """The stiffness command. Shape is (num_envs, 1)."""
        return self.stiffness_command

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        if self._kp_grid is not None:
            idx = torch.randint(0, self._kp_grid.numel(), (n,), device=self.device)
            self.stiffness_command[env_ids, 0] = self._kp_grid[idx]
        else:
            r = torch.empty(n, device=self.device)
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
        """Ranges for base stiffness sampling."""

        kp: tuple[float, float] = (5.0, 20.0)
        """Range for base stiffness [min, max]."""

    ranges: Ranges = Ranges()
    discrete_step: float | None = None
