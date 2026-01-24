import numpy as np



class MSDSystem:
    """Mass-Spring-Damper system for modeling compliance."""

    def __init__(self, n_dofs, active_dof_indices, stiffness_scales, dt, base_inertia, base_stiffness):
        """
        Initialize MSD system.

        Args:
            n_dofs: Total number of DOFs in the system
            active_dof_indices: Array of DOF indices to apply MSD to
            ACTIVE_DOFS_MSD: Dict mapping DOF index -> stiffness scale factor
            dt: Simulation timestep
            base_inertia
            base_stiffness
        """
        self.n_dofs = n_dofs
        self.active_idx = active_dof_indices
        self.n_active = len(active_dof_indices)
        self.dt = dt

        # Initialize MSD matrices
        self.M = np.ones(n_dofs) * base_inertia
        self.K = np.zeros(n_dofs)
        self.D = np.zeros(n_dofs)

        # Build DOF mask and apply stiffness scales
        self.dof_mask = np.zeros(n_dofs, dtype=bool)
        for dof_idx, scale in stiffness_scales.items():
            self.K[dof_idx] = base_stiffness * scale
            self.dof_mask[dof_idx] = True

        # Compute critical damping: D = 2*sqrt(M*K)
        self.D = 2 * np.sqrt(self.M * self.K)

        # Compute discrete-time state-space matrices
        self.Ad, self.Bd, self.active_idx = self._compute_discrete_matrices()

        # MSD state: [q, dq] for active DOFs
        # self.state = np.zeros(2 * self.n_active)
        self.state = self._create_msd_state()


    def _create_msd_state(self):
        """Create MSD state dictionary for tracking deformations (only active DOFs)."""
        return {
            'q_def': np.zeros(len(self.active_idx)),      # Deformation position (rad or m) - active DOFs only
            'qd_def': np.zeros(len(self.active_idx)),     # Deformation velocity (rad/s or m/s) - active DOFs only
        }


    def get_state_dict(self):
        """Get current MSD state as dictionary."""
        return self.state
 

    def _compute_discrete_matrices(self):
        """Compute discrete-time state-space matrices for active DOFs."""
        from scipy.linalg import expm
        
        active_idx = np.where(self.dof_mask)[0]
        n_active = len(active_idx)
    
        if self.n_active == 0:
            return np.eye(0), np.zeros((0, 0))

        M_active = self.M[self.active_idx]
        D_active = self.D[self.active_idx]
        K_active = self.K[self.active_idx]

        # Build continuous-time state matrices (reduced dimension)
        # A = [[0, I], [-M^-1*K, -M^-1*D]]
        A = np.zeros((2*self.n_active, 2*self.n_active))
        B = np.zeros((2*self.n_active, self.n_active))

        # Upper-right block: I
        A[:self.n_active, self.n_active:] = np.eye(self.n_active)

        # Lower-left block: -M^-1*K
        A[self.n_active:, :self.n_active] = np.diag(-K_active / M_active)

        # Lower-right block: -M^-1*D
        A[self.n_active:, self.n_active:] = np.diag(-D_active / M_active)

        # Input matrix B = [[0], [M^-1]]
        B[self.n_active:, :] = np.diag(1.0 / M_active)

        # Compute discrete-time matrices
        # Ad = exp(A * dt)
        Ad = expm(A * self.dt)

        # Bd = A^-1 * (Ad - I) * B
        A_inv = np.linalg.inv(A)
        Bd = A_inv @ (Ad - np.eye(2*self.n_active)) @ B

        return Ad, Bd, active_idx


    def update_msd_state_discrete(self, external_torques_full):
        """
        Update Mass-Spring-Damper state using analytical discrete-time solution.
        Works only with active DOFs internally.

        Uses precomputed discrete-time state-space matrices:
            x[k+1] = Ad*x[k] + Bd*u[k]
            where x = [q_def; qd_def], u = tau_ext (active DOFs only)

        Args:
            external_torques_full: External torques on joints (nv,) - full DOF vector
        """
        if self.n_active == 0:
            return  # No active DOFs

        # Extract only active DOF torques
        external_torques_active = external_torques_full[self.active_idx]

        # Pack state vector: x = [q_def; qd_def] (active DOFs only)
        x = np.concatenate([self.state['q_def'], self.state['qd_def']])

        # Discrete-time state update: x[k+1] = Ad*x[k] + Bd*u[k]
        x_next = self.Ad @ x + self.Bd @ external_torques_active

        # Unpack state vector (active DOFs only)
        n_active = len(self.active_idx)
        self.state['q_def'][:] = x_next[:n_active]
        self.state['qd_def'][:] = x_next[n_active:]


    def reset(self):
        """Reset MSD state to zero."""
        self.state['q_def'][:] = 0.0
        self.state['qd_def'][:] = 0.0


