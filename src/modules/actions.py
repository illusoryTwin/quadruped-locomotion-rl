"""Custom action terms with EMA smoothing for jerk-free joint position control."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EMAJointPositionAction(JointPositionAction):
    """Joint position action with exponential moving average (EMA) filtering.

    Applies a low-pass filter on the processed actions before sending them as
    position commands:

        applied = alpha * processed + (1 - alpha) * prev_applied

    where alpha=1.0 means no filtering (passthrough) and lower alpha values
    produce smoother transitions. Typical values: 0.2-0.5.
    """

    cfg: EMAJointPositionActionCfg

    def __init__(self, cfg: EMAJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._alpha = cfg.alpha
        self._prev_applied_actions = torch.zeros_like(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # reset EMA history to current joint positions so there's no jump on reset
        if env_ids is None:
            env_ids = slice(None)
        self._prev_applied_actions[env_ids] = self._asset.data.joint_pos[
            env_ids, self._joint_ids
        ]

    def process_actions(self, actions: torch.Tensor):
        # standard processing: raw * scale + offset
        super().process_actions(actions)
        # apply EMA filter
        self._processed_actions[:] = (
            self._alpha * self._processed_actions
            + (1.0 - self._alpha) * self._prev_applied_actions
        )
        self._prev_applied_actions[:] = self._processed_actions[:]


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for EMA-filtered joint position action term."""

    class_type: type = EMAJointPositionAction

    alpha: float = 0.3
    """EMA smoothing factor in [0, 1]. Lower = smoother but slower response.
    - 1.0: no filtering (same as standard JointPositionAction)
    - 0.3-0.5: moderate smoothing (recommended starting point)
    - 0.1-0.2: heavy smoothing (may feel sluggish)
    """
