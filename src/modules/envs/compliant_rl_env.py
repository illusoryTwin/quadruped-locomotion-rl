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

        # deformed joint position targets for use in reward computation
        self._compliant_joint_targets = None

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

        # -- compute compliant targets before rewards so the reward can use them
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

        # -- log compliance deformations
        self._log_compliance_metrics()

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_compliance_targets(self):
        """Compute compliant joint position targets from deformations.

        Stores deformed targets on self._compliant_joint_targets for use
        by the reward function. Does NOT modify PD targets — the policy
        must learn to reach these positions through reward shaping.
        """
        if self.compliance_manager is None:
            return

        deformations = self.compliance_manager.compute(dt=self.physics_dt)

        robot = self.scene["robot"]

        targets = robot._data.joint_pos_target.clone()
        for i, joint_idx in enumerate(self.compliance_manager.active_joint_indices):
            targets[:, joint_idx] += deformations[:, i]

        self._compliant_joint_targets = targets

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

        # Call parent reset
        super()._reset_idx(env_ids)
