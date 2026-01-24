
class SoftComplianceManager:    
    """Manager for soft joint compliance using Mass-Spring-Damper model."""

    def __init__(self, cfg: SoftComplianceManagerCfg, env):
        self.cfg = cfg 
        self._env = env
        self._robot = env.scene.articulations[cfg.robot_name]
        self._device = env.device 
        self._num_envs = env.num_envs
        # self._deformations = None 

        # Monitored bodies - from config directly
        self._monitored_body_indices = [
            self._robot.body_names.index(name)
            for name in cfg.monitored_bodies
        ]

        # Active joints - derived from stiffness_config keys
        self._active_joint_indices = [
            self._robot.joint_names.index(name)
            for name in cfg.stiffness_config.keys()
            if name in self._robot.joint_names
        ]
        
        self._msd_system = self._setup_msd_system()
        self._deformations = None

    def _setup_msd_system(self) -> MSDSystem:
        cfg = self.cfg
        n_dofs = self._robot.num_joints

        # Build DOF index to scale mapping
        stiffness_scales = {}  # Maps DOF index -> stiffness scale

        # Process joints in stiffness_config
        for joint_name, scale in cfg.stiffness_config.items():
            if scale > 0 and joint_name in self._robot.joint_names:
                joint_idx = self._robot.joint_names.index(joint_name)
                stiffness_scales[joint_idx] = scale

        # Get active DOF indices
        active_dof_indices = np.array(sorted(stiffness_scales.keys()), dtype=np.int32)
        print(f"[SoftComplianceManager] Number of active DOFs: {len(active_dof_indices)}")

        # Create and return MSD system
        return MSDSystem(
            n_dofs=n_dofs,
            active_dof_indices=active_dof_indices,
            stiffness_scales=stiffness_scales,
            dt=cfg.dt,
            base_inertia=cfg.base_inertia,
            base_stiffness=cfg.base_stiffness
        )

    @property
    def active_joint_indices(self):
        """Return list of active joint indices."""
        return self._active_joint_indices

    def reset(self, env_ids: torch.Tensor):
        """Reset MSD state for specified environments."""
        # reset all for now
        if self._msd_system is not None:
            self._msd_system.reset()

    def compute(self, dt: float) -> torch.Tensor:
        """Compute joint deformations from external forces."""

        # Compute joint torques from wrenches using Jacobian transpose
        joint_torques = self._compute_joint_torques()

        # Update MSD system and get deformations
        torques_np = joint_torques[0].cpu().numpy()
        self._msd_system.update_msd_state_discrete(torques_np)

        # Get deformations for active DOFs only
        q_def = self._msd_system.state['q_def']

        # Convert to torch and broadcast to all environments
        deformations = torch.tensor(
            q_def, dtype=torch.float32, device=self._device
        ).unsqueeze(0).expand(self._num_envs, -1)

        self._deformations = deformations

        if self.cfg.debug:
            print(f"[SoftComplianceManager] Deformations: {deformations[0]}")

        return deformations


    def _compute_joint_torques(self) -> torch.Tensor:
        """Calculate joint torques from COMPLIANCE forces only (in body frame):
        tau = J^T @ wrench
        """
        verbose = False
        robot = self._robot
        body_names = self.cfg.monitored_bodies
        body_indices = [robot.body_names.index(name) for name in body_names]

        # Get Jacobian in world frame: [num_envs, num_bodies, 6, num_joints]
        jacobians_w = robot.root_physx_view.get_jacobians()[:, body_indices, :, :]

        # Get body orientations for frame transformation
        body_quat_w = robot.data.body_quat_w[:, body_indices, :]

        # Transform Jacobian from world frame to body frame
        jacobians_b = self._transform_jacobian_world2body(jacobians_w, body_quat_w)

        # Get wrenches from COMPLIANCE buffers
        if hasattr(self._env, '_compliance_force_b') and self._env._compliance_force_b is not None:
            forces_b = self._env._compliance_force_b[:, body_indices, :]
            torques_b = self._env._compliance_torque_b[:, body_indices, :]
        else:
            # use standard buffers
            forces_b = robot._external_force_b[:, body_indices, :]
            torques_b = robot._external_torque_b[:, body_indices, :]

        num_envs = jacobians_b.shape[0]
        num_bodies = len(body_indices)
        jacobian_cols = jacobians_b.shape[-1]

        if verbose:
            print("\n=== BODY FRAME ===")
            for i, name in enumerate(body_names):
                print(f"\nBody: {name}")
                print(f"  Wrench (body): force={forces_b[0, i].cpu().numpy()}, torque={torques_b[0, i].cpu().numpy()}")
                print(f"  Jacobian (body) shape: {jacobians_b[0, i].shape}")
                print(f"  Jacobian (body):\n{jacobians_b[0, i].cpu().numpy()}")

        total_torques = torch.zeros((num_envs, jacobian_cols), device=jacobians_b.device)

        for i in range(num_bodies):
            jacobian_b = jacobians_b[:, i, :, :]
            wrench_b = torch.cat([forces_b[:, i, :], torques_b[:, i, :]], dim=1).unsqueeze(-1)
            # Compute joint torques: tau = J_b^T @ wrench_b
            torques = torch.bmm(jacobian_b.transpose(1, 2), wrench_b).squeeze(-1)
            total_torques += torques
        
        verbose = False # True
        if verbose:
            print(f"\nTotal torques (body frame): {total_torques[0].cpu().numpy()}")

        return total_torques
    

    def _transform_jacobian_world2body(self, jacobian_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform Jacobian from world frame to body frame.

        Returns:
            Jacobian in body frame [num_envs, num_bodies, 6, num_joints]
        """
        num_envs = jacobian_w.shape[0]
        num_bodies = jacobian_w.shape[1]
        num_joints = jacobian_w.shape[3]

        jacobian_b = torch.zeros_like(jacobian_w)

        for env in range(num_envs):
            for i in range(num_bodies):
                # R_b_w = R_w_b^T = rotation from world to body
                R_w_b = matrix_from_quat(body_quat_w[env, i])
                R_b_w = R_w_b.T

                # Transform linear part (rows 0:3)
                jacobian_b[env, i, :3, :] = R_b_w @ jacobian_w[env, i, :3, :]
                # Transform angular part (rows 3:6)
                jacobian_b[env, i, 3:, :] = R_b_w @ jacobian_w[env, i, 3:, :]

        return jacobian_b
