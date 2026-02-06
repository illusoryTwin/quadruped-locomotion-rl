from collections.abc import Sequence
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from compliance import ComplianceManager, ComplianceManagerCfg


class CompliantRLEnv(ManagerBasedRLEnv):
    """RL environment with compliance support."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize compliance manager if configured and enabled
        self.compliance_manager = None
        if hasattr(cfg, 'compliance') and cfg.compliance.enabled:
            self.compliance_manager = ComplianceManager(cfg.compliance, self)

        # Cached force buffers — stores forces BEFORE write_data_to_sim() clears them
        self._cached_force_b = None
        self._cached_torque_b = None

        if self.compliance_manager is not None:
            num_monitored = len(self.compliance_manager.monitored_body_indices)
            self._cached_force_b = torch.zeros(
                (self.num_envs, num_monitored, 3), device=self.device
            )
            self._cached_torque_b = torch.zeros_like(self._cached_force_b)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Cache external forces before the decimation loop clears them.
        # Otherwise they will be only applied for the first sub-step.
        robot = self.scene["robot"]
        has_compliance = self._cached_force_b is not None
        if has_compliance:
            body_indices = self.compliance_manager.monitored_body_indices
            self._cached_force_b[:] = robot._external_force_b[:, body_indices, :]
            self._cached_torque_b[:] = robot._external_torque_b[:, body_indices, :]

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()

            # Re-apply cached forces so they persist across all sub-steps
            if has_compliance:
                robot._external_force_b[:, body_indices, :] = self._cached_force_b
                robot._external_torque_b[:, body_indices, :] = self._cached_torque_b

            # set actions into simulator (sends forces to PhysX, then clears buffer)
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

        # -- apply compliance (modifies joint positions based on external forces)
        self._apply_compliance_deformations()

        # -- log compliance deformations
        self._log_compliance_metrics()

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_compliance_deformations(self):
        """Apply compliance deformations to the robot's joint positions."""
        if self.compliance_manager is None:
            print("Compliance is not supported")
            return

        # Compute deformations
        deformations = self.compliance_manager.compute(
            dt=self.physics_dt,
            cached_forces=self._cached_force_b,
            cached_torques=self._cached_torque_b,
        )

        # Access the robot articulation
        robot = self.scene["robot"]

        # Apply deformations to the robot's joint positions
        current_positions = robot.data.joint_pos.clone()
        for i, joint_idx in enumerate(self.compliance_manager.active_joint_indices):
            current_positions[:, joint_idx] += deformations[:, i]

        # Write modified positions back to robot
        robot.write_joint_state_to_sim(current_positions, robot.data.joint_vel)

    def _log_compliance_metrics(self):
        """Log compliance-related metrics to extras for tensorboard."""
        # Always ensure extras["log"] exists (RSL-RL expects it every step)
        if "log" not in self.extras:
            self.extras["log"] = {}

        if self.compliance_manager is None or self.compliance_manager._deformations is None:
            return

        deformations = self.compliance_manager._deformations

        # Log mean absolute deformation across all environments
        self.extras["log"]["compliance/mean_deformation"] = deformations.abs().mean().item()

        # Log max absolute deformation
        self.extras["log"]["compliance/max_deformation"] = deformations.abs().max().item()

        # Log RMS deformation
        self.extras["log"]["compliance/rms_deformation"] = deformations.pow(2).mean().sqrt().item()

        # Log per-joint mean deformations (averaged across environments)
        joint_names = [
            self.scene["robot"].joint_names[idx]
            for idx in self.compliance_manager.active_joint_indices
        ]
        mean_per_joint = deformations.abs().mean(dim=0)
        for i, name in enumerate(joint_names):
            self.extras["log"][f"compliance/deformation_{name}"] = mean_per_joint[i].item()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices."""
        # Reset compliance manager for these environments
        if self.compliance_manager is not None:
            self.compliance_manager.reset(env_ids)

        # Reset cached force buffers so that they don't leak into new episodes
        if self._cached_force_b is not None:
            self._cached_force_b[env_ids] = 0
            self._cached_torque_b[env_ids] = 0

        # Call parent reset
        super()._reset_idx(env_ids)
