"""Command term that samples target XY positions and converts errors to body-frame velocity commands."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PositionCommand(CommandTerm):
    """Command that samples target XY positions and converts errors to body-frame velocity commands.

    Same pattern as heading -> ang_vel_z, but for position -> lin_vel_xy.
    Target positions are sampled as offsets from env origin, then position error
    is rotated into the robot's body frame and converted to velocity via a P-controller:

        pos_error_world = target_xy - current_xy
        pos_error_body  = rotate_to_body(pos_error_world, yaw)
        vel_command_b   = clip(stiffness * pos_error_body, -vel_max, vel_max)
    """

    cfg: PositionCommandCfg

    def __init__(self, cfg: PositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # target XY in world frame (absolute)
        self.pos_target_w = torch.zeros(self.num_envs, 2, device=self.device)
        # output: body-frame linear velocity commands
        self.vel_command_b = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Body-frame linear velocity command from position error. Shape is (num_envs, 2)."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        r = torch.empty(n, device=self.device)
        # sample target XY as offset from env origin
        self.pos_target_w[env_ids, 0] = self._env.scene.env_origins[env_ids, 0] + r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_target_w[env_ids, 1] = self._env.scene.env_origins[env_ids, 1] + r.uniform_(*self.cfg.ranges.pos_y)

    def _update_command(self):
        # position error in world frame
        pos_error_w = self.pos_target_w - self.robot.data.root_pos_w[:, :2]
        # rotate to body frame using robot's yaw
        heading = self.robot.data.heading_w
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        error_body_x = cos_h * pos_error_w[:, 0] + sin_h * pos_error_w[:, 1]
        error_body_y = -sin_h * pos_error_w[:, 0] + cos_h * pos_error_w[:, 1]
        # P-controller: position error -> velocity command
        stiffness = self.cfg.position_control_stiffness
        vel_min, vel_max = self.cfg.ranges.vel
        self.vel_command_b[:, 0] = torch.clip(stiffness * error_body_x, min=vel_min, max=vel_max)
        self.vel_command_b[:, 1] = torch.clip(stiffness * error_body_y, min=vel_min, max=vel_max)

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)
                self.goal_sphere_visualizer = VisualizationMarkers(self.cfg.goal_sphere_visualizer_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_sphere_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.goal_sphere_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # sphere at target position
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos[:, 0] = self.pos_target_w[:, 0]
        target_pos[:, 1] = self.pos_target_w[:, 1]
        target_pos[:, 2] = self._env.scene.env_origins[:, 2] + 0.05
        self.goal_sphere_visualizer.visualize(target_pos)
        # arrow from robot pointing toward target
        direction = self.pos_target_w - self.robot.data.root_pos_w[:, :2]
        heading_angle = torch.atan2(direction[:, 1], direction[:, 0])
        zeros = torch.zeros(self.num_envs, device=self.device)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        default_scale = self.goal_pos_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        arrow_scale[:, 0] *= torch.linalg.norm(direction, dim=1) * 2.0
        robot_pos = self.robot.data.root_pos_w.clone()
        robot_pos[:, 2] += 0.5
        self.goal_pos_visualizer.visualize(robot_pos, arrow_quat, arrow_scale)


@configclass
class PositionCommandCfg(CommandTermCfg):
    """Configuration for the position command generator."""

    class_type: type = PositionCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    position_control_stiffness: float = 1.0
    """P-controller gain to convert position error (m) to velocity command (m/s)."""

    @configclass
    class Ranges:
        """Ranges for position targets and velocity limits."""

        pos_x: tuple[float, float] = (-1.0, 1.0)
        """Range for target X offset from env origin (in m)."""

        pos_y: tuple[float, float] = (-1.0, 1.0)
        """Range for target Y offset from env origin (in m)."""

        vel: tuple[float, float] = (-1.5, 1.5)
        """Clip range for the velocity commands (in m/s)."""

    ranges: Ranges = Ranges()

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/position_goal"
    )
    """Marker config for the goal position arrow."""
    goal_pos_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    goal_sphere_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/position_target",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    """Green sphere marker at the target position."""
