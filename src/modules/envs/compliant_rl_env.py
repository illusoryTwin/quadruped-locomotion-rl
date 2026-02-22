from collections.abc import Sequence
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_apply_yaw
from src.compliance import ComplianceManager, ComplianceManagerCfg


class CompliantRLEnv(ManagerBasedRLEnv):
    """RL environment with compliance support."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize compliance manager if configured and enabled
        self.compliance_manager = None
        if hasattr(cfg, 'compliance') and cfg.compliance.enabled:
            self.compliance_manager = ComplianceManager(cfg.compliance, self)

        # Cartesian compliance buffers
        self._rigid_ref_pos = None     # [num_envs, 3] rigid reference position (world frame)
        self._compliant_ref_pos = None # [num_envs, 3] compliant reference position
        self._compliant_ref_vel = None # [num_envs, 3] compliant reference velocity

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # -- apply step-mode events (e.g. sinusoidal forces) BEFORE compliance
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step")

        # -- compute compliant Cartesian targets BEFORE rewards so the reward can use them
        self._compute_compliance_targets()

        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- log compliance metrics
        self._log_compliance_metrics()

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_compliance_targets(self):
        """Compute Cartesian compliant reference position and velocity for the base body.
        """

        if self.compliance_manager is None:
            return

        robot = self.scene["robot"]

        # Read per-env stiffness from command if available
        base_stiffness = None
        try:
            kp_cmd = self.command_manager.get_command("stiffness")
            base_stiffness = kp_cmd[:, 0]  # [num_envs]
        except (KeyError, RuntimeError):
            pass

        self.compliance_manager.compute(
            dt=self.physics_dt, base_stiffness=base_stiffness
        )

        # Extract base Cartesian deformation from MSD state (first 3 DOFs = base XYZ)
        msd = self.compliance_manager._msd_system
        x_def_base = msd.state['x_def'][:, 0:3]   # [num_envs, 3] world frame
        dx_def_base = msd.state['dx_def'][:, 0:3]  # [num_envs, 3] world frame

        # Clamp Cartesian deformation
        max_def = self.cfg.compliance.max_cartesian_deformation
        x_def_base = x_def_base.clamp(-max_def, max_def)

        # Initialize rigid reference on first call
        if self._rigid_ref_pos is None:
            self._rigid_ref_pos = robot.data.root_pos_w[:, :3].clone()

        # Get commanded velocity in body frame and rotate to world frame
        v_cmd = self.command_manager.get_command("base_velocity")  # [num_envs, 4]
        v_cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
        v_cmd_body[:, 0] = v_cmd[:, 0]  # vx
        v_cmd_body[:, 1] = v_cmd[:, 1]  # vy
        v_cmd_world = quat_apply_yaw(robot.data.root_quat_w, v_cmd_body)

        # Update rigid reference by integrating commanded velocity
        self._rigid_ref_pos = self._rigid_ref_pos + v_cmd_world * self.step_dt

        # Clamp drift: prevent rigid reference from drifting too far from actual position
        actual_pos = robot.data.root_pos_w[:, :3]
        drift = self._rigid_ref_pos - actual_pos
        drift_norm = drift.norm(dim=1, keepdim=True).clamp(min=1e-6)
        max_drift = 0.1  # meters
        clamped_drift = drift * (max_drift / drift_norm).clamp(max=1.0)
        self._rigid_ref_pos = actual_pos + clamped_drift

        # Compute compliant references
        self._compliant_ref_pos = self._rigid_ref_pos + x_def_base
        self._compliant_ref_vel = v_cmd_world + dx_def_base



    def _log_compliance_metrics(self):
        """Log Cartesian compliance metrics to extras for tensorboard."""
        if "log" not in self.extras:
            self.extras["log"] = {}

        if self.compliance_manager is None:
            return

        robot = self.scene["robot"]

        # --- MSD Cartesian deformation stats ---
        msd = self.compliance_manager._msd_system
        if msd is not None:
            x_def_base = msd.state['x_def'][:, 0:3]
            self.extras["log"]["compliance/x_def_norm"] = x_def_base.norm(dim=1).mean().item()
            self.extras["log"]["compliance/x_def_x"] = x_def_base[:, 0].abs().mean().item()
            self.extras["log"]["compliance/x_def_y"] = x_def_base[:, 1].abs().mean().item()
            self.extras["log"]["compliance/x_def_z"] = x_def_base[:, 2].abs().mean().item()

        # --- Cartesian tracking errors ---
        if self._compliant_ref_pos is not None:
            actual_pos = robot.data.root_pos_w[:, :3]
            actual_vel = robot.data.root_lin_vel_w[:, :3]

            pos_error = (actual_pos - self._compliant_ref_pos).norm(dim=1)
            vel_error = (actual_vel - self._compliant_ref_vel).norm(dim=1)

            self.extras["log"]["compliance/pos_error"] = pos_error.mean().item()
            self.extras["log"]["compliance/vel_error"] = vel_error.mean().item()

            # Rigid reference drift from actual position
            drift = (self._rigid_ref_pos - actual_pos).norm(dim=1)
            self.extras["log"]["compliance/rigid_ref_drift"] = drift.mean().item()

        # --- Stiffness command ---
        try:
            kp_cmd = self.command_manager.get_command("stiffness")
            self.extras["log"]["compliance/stiffness_kp_mean"] = kp_cmd.mean().item()
        except (KeyError, RuntimeError):
            pass

        # --- External forces on compliant bodies ---
        body_names = self.compliance_manager._compliant_body_names
        body_indices = [robot.body_names.index(n) for n in body_names]
        forces = robot._external_force_b[:, body_indices, :]
        self.extras["log"]["compliance/ext_force_norm"] = forces.norm(dim=-1).mean().item()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices."""
        # Reset compliance manager for these environments
        if self.compliance_manager is not None:
            self.compliance_manager.reset(env_ids)

        # Call parent reset
        super()._reset_idx(env_ids)

        # Reset rigid reference to actual position after parent reset
        if self._rigid_ref_pos is not None:
            robot = self.scene["robot"]
            self._rigid_ref_pos[env_ids] = robot.data.root_pos_w[env_ids, :3].clone()
