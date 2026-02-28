"""UniformVelocityCommand subclass with heading direction visualization."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HeadingVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand with heading direction arrows instead of velocity arrows.

    Visualizes two arrows above the robot:
      - Green: target heading direction
      - Blue: current heading direction
    """

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        n = self.num_envs
        zeros = torch.zeros(n, device=self.device)

        # goal heading arrow (green)
        goal_scale = torch.tensor(default_scale, device=self.device).repeat(n, 1)
        goal_scale[:, 0] *= 3.0
        goal_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_target)
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)

        # current heading arrow (blue)
        current_scale = torch.tensor(default_scale, device=self.device).repeat(n, 1)
        current_scale[:, 0] *= 3.0
        current_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.robot.data.heading_w)
        self.current_vel_visualizer.visualize(base_pos_w, current_quat, current_scale)


@configclass
class HeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for HeadingVelocityCommand."""

    class_type: type = HeadingVelocityCommand
