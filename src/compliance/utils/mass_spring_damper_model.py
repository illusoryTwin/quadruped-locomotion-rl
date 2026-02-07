# import numpy as np



# class MSDSystem:
#     """Mass-Spring-Damper system for modeling compliance."""

#     def __init__(self, n_dofs, active_dof_indices, stiffness_scales, dt, base_inertia, base_stiffness):
#         """
#         Initialize MSD system.

#         Args:
#             n_dofs: Total number of DOFs in the system
#             active_dof_indices: Array of DOF indices to apply MSD to
#             ACTIVE_DOFS_MSD: Dict mapping DOF index -> stiffness scale factor
#             dt: Simulation timestep
#             base_inertia
#             base_stiffness
#         """
#         self.n_dofs = n_dofs
#         self.active_idx = active_dof_indices
#         self.n_active = len(active_dof_indices)
#         self.dt = dt

#         # Initialize MSD matrices
#         self.M = np.ones(n_dofs) * base_inertia
#         self.K = np.zeros(n_dofs)
#         self.D = np.zeros(n_dofs)

#         # Build DOF mask and apply stiffness scales
#         self.dof_mask = np.zeros(n_dofs, dtype=bool)
#         for dof_idx, scale in stiffness_scales.items():
#             self.K[dof_idx] = base_stiffness * scale
#             self.dof_mask[dof_idx] = True

#         # Compute critical damping: D = 2*sqrt(M*K)
#         self.D = 2 * np.sqrt(self.M * self.K)

#         # Compute discrete-time state-space matrices
#         self.Ad, self.Bd, self.active_idx = self._compute_discrete_matrices()

#         # MSD state: [q, dq] for active DOFs
#         # self.state = np.zeros(2 * self.n_active)
#         self.state = self._create_msd_state()


#     def _create_msd_state(self):
#         """Create MSD state dictionary for tracking deformations (only active DOFs)."""
#         return {
#             'q_def': np.zeros(len(self.active_idx)),      # Deformation position (rad or m) - active DOFs only
#             'qd_def': np.zeros(len(self.active_idx)),     # Deformation velocity (rad/s or m/s) - active DOFs only
#         }


#     def get_state_dict(self):
#         """Get current MSD state as dictionary."""
#         return self.state
 

#     def _compute_discrete_matrices(self):
#         """Compute discrete-time state-space matrices for active DOFs."""
#         from scipy.linalg import expm
        
#         active_idx = np.where(self.dof_mask)[0]
#         n_active = len(active_idx)
    
#         if self.n_active == 0:
#             return np.eye(0), np.zeros((0, 0)), np.array([], dtype=np.int32)

#         M_active = self.M[self.active_idx]
#         D_active = self.D[self.active_idx]
#         K_active = self.K[self.active_idx]

#         # Build continuous-time state matrices (reduced dimension)
#         # A = [[0, I], [-M^-1*K, -M^-1*D]]
#         A = np.zeros((2*self.n_active, 2*self.n_active))
#         B = np.zeros((2*self.n_active, self.n_active))

#         # Upper-right block: I
#         A[:self.n_active, self.n_active:] = np.eye(self.n_active)

#         # Lower-left block: -M^-1*K
#         A[self.n_active:, :self.n_active] = np.diag(-K_active / M_active)

#         # Lower-right block: -M^-1*D
#         A[self.n_active:, self.n_active:] = np.diag(-D_active / M_active)

#         # Input matrix B = [[0], [M^-1]]
#         B[self.n_active:, :] = np.diag(1.0 / M_active)

#         # Compute discrete-time matrices
#         # Ad = exp(A * dt)
#         Ad = expm(A * self.dt)

#         # Bd = A^-1 * (Ad - I) * B
#         A_inv = np.linalg.inv(A)
#         Bd = A_inv @ (Ad - np.eye(2*self.n_active)) @ B

#         return Ad, Bd, active_idx


#     def update_msd_state_discrete(self, external_torques_full):
#         """
#         Update Mass-Spring-Damper state using analytical discrete-time solution.
#         Works only with active DOFs internally.

#         Uses precomputed discrete-time state-space matrices:
#             x[k+1] = Ad*x[k] + Bd*u[k]
#             where x = [q_def; qd_def], u = tau_ext (active DOFs only)

#         Args:
#             external_torques_full: External torques on joints (nv,) - full DOF vector
#         """
#         if self.n_active == 0:
#             return  # No active DOFs

#         # Extract only active DOF torques
#         external_torques_active = external_torques_full[self.active_idx]

#         # Pack state vector: x = [q_def; qd_def] (active DOFs only)
#         x = np.concatenate([self.state['q_def'], self.state['qd_def']])

#         # Discrete-time state update: x[k+1] = Ad*x[k] + Bd*u[k]
#         x_next = self.Ad @ x + self.Bd @ external_torques_active

#         # Unpack state vector (active DOFs only)
#         n_active = len(self.active_idx)
#         self.state['q_def'][:] = x_next[:n_active]
#         self.state['qd_def'][:] = x_next[n_active:]


#     def reset(self):
#         """Reset MSD state to zero."""
#         self.state['q_def'][:] = 0.0
#         self.state['qd_def'][:] = 0.0


import numpy as np
import torch

class MassSpringDamperModel:
    """Mass-Spring-Damper system for modeling compliance."""

    def __init__(
        self,
        n_dofs: int,
        dt: float,
        base_inertia: float,
        base_stiffness: float,
        stiffness_scales: dict[int, float],
        num_envs=1,
        device="cuda"
    ):
        """                              
        Initialize MSD system.

        Args:
            n_dofs: Total number of DOFs in the system
            dt: Simulation timestep in seconds
            base_inertia: Base inertia value applied to all DOFs
            base_stiffness: Base stiffness value, scaled per-DOF by stiffness_scales
            stiffness_scales: Mapping of DOF index to stiffness scale factor.     
                Only DOFs present in this dict are modeled as compliant.
            num_envs: Number of parallel environments for batched simulation.
            device: Torch device for computation ('cuda' or 'cpu')
        """
        self.n_dofs = n_dofs
        self.dt = dt
        self.active_idx = np.array(sorted(stiffness_scales.keys()))
        self.n_active = len(self.active_idx)
        self.num_envs = num_envs
        self.device = device

        # Initialize MSD matrices (numpy for matrix computation)
        self.M = np.ones(n_dofs) * base_inertia
        self.K = np.zeros(n_dofs)
        self.D = np.zeros(n_dofs)

        for dof_idx, scale in stiffness_scales.items():
            self.K[dof_idx] = base_stiffness * scale

        # Compute critical damping: D = 2*sqrt(M*K)
        self.D = 2 * np.sqrt(self.M * self.K)

        # Compute discrete-time state-space matrices and convert to Torch
        Ad_np, Bd_np = self._compute_discrete_matrices()
        self.Ad = torch.tensor(Ad_np, dtype=torch.float32, device=device)
        self.Bd = torch.tensor(Bd_np, dtype=torch.float32, device=device)

        # Convert active_idx to torch for indexing
        self.active_idx_torch = torch.tensor(self.active_idx, dtype=torch.long, device=device)

        # Store per-DOF parameters as torch tensors for the analytical path
        scales_ordered = [stiffness_scales[idx] for idx in self.active_idx]
        self._scales_t = torch.tensor(scales_ordered, dtype=torch.float32, device=device)
        self._M_active_t = torch.tensor(self.M[self.active_idx], dtype=torch.float32, device=device)

        # MSD state: batched [num_envs, n_active] for q_def and qd_def
        self.state = self._create_msd_state()

    def _create_msd_state(self):
        """Create MSD state dictionary for tracking deformations (only active DOFs)."""
        return {
            'q_def': torch.zeros((self.num_envs, self.n_active), dtype=torch.float32, device=self.device),
            'qd_def': torch.zeros((self.num_envs, self.n_active), dtype=torch.float32, device=self.device),
        }

    def get_state_dict(self):
        """Get current MSD state as dictionary."""
        return self.state

    def _compute_discrete_matrices(self):
        """Compute discrete-time state-space matrices for active DOFs."""
        from scipy.linalg import expm

        active_idx = self.active_idx

        if self.n_active == 0:
            return np.eye(0), np.zeros((0, 0)), active_idx

        M_active = self.M[self.active_idx]
        D_active = self.D[self.active_idx]
        K_active = self.K[self.active_idx]

        # Build continuous-time state matrices (reduced dimension)
        # A = [[0, I], [-M^-1*K, -M^-1*D]]
        A = np.zeros((2 * self.n_active, 2 * self.n_active))
        B = np.zeros((2 * self.n_active, self.n_active))

        # Upper-right block: I
        A[: self.n_active, self.n_active :] = np.eye(self.n_active)

        # Lower-left block: -M^-1*K
        A[self.n_active :, : self.n_active] = np.diag(-K_active / M_active)

        # Lower-right block: -M^-1*D
        A[self.n_active :, self.n_active :] = np.diag(-D_active / M_active)

        # Input matrix B = [[0], [M^-1]]
        B[self.n_active :, :] = np.diag(1.0 / M_active)

        # Compute discrete-time matrices
        # Ad = exp(A * dt)
        Ad = expm(A * self.dt)

        # Bd = A^-1 * (Ad - I) * B
        A_inv = np.linalg.inv(A)
        Bd = A_inv @ (Ad - np.eye(2 * self.n_active)) @ B

        return Ad, Bd

    def update_msd_state_discrete(self, external_torques: torch.Tensor):
        """
        Update Mass-Spring-Damper state using analytical discrete-time solution.
        Batched computation for all environments.

        Uses precomputed discrete-time state-space matrices:
            x[k+1] = Ad*x[k] + Bd*u[k]
            where x = [q_def; qd_def], u = tau_ext (active DOFs only)

        Args:
            external_torques: External torques on joints [num_envs, n_dofs] - full DOF vector
        """
        if self.n_active == 0:
            return  # No active DOFs

        # Ensure torques are on the correct device
        external_torques = external_torques.to(device=self.device)

        # Extract only active DOF torques: [num_envs, n_active]
        external_torques_active = external_torques[:, self.active_idx_torch]

        # Pack state vector: x = [q_def; qd_def] (active DOFs only)
        x = torch.cat([self.state["q_def"], self.state["qd_def"]], dim=1)

        # Discrete-time state update: x[k+1] = Ad @ x[k] + Bd @ u[k]
        x_next = x @ self.Ad.T + external_torques_active @ self.Bd.T

        # Unpack state vector (active DOFs only)
        # x_next shape: [num_envs, 2*n_active] where first n_active cols are q_def
        self.state["q_def"][:] = x_next[:, :self.n_active]
        self.state["qd_def"][:] = x_next[:, self.n_active:]

    def update_with_variable_stiffness(
        self, external_torques: torch.Tensor, base_stiffness: torch.Tensor
    ):
        """Update MSD state using per-env stiffness via analytical critically-damped solution.

        For a critically-damped 2nd order system (D = 2*sqrt(M*K)), the exact
        discrete update per DOF is computed analytically without matrix expm.
        This allows each environment to have a different base_stiffness value.

        Args:
            external_torques: External torques [num_envs, n_dofs] - full DOF vector
            base_stiffness: Per-env base stiffness [num_envs] or [num_envs, 1]
        """
        if self.n_active == 0:
            return

        external_torques = external_torques.to(device=self.device)
        tau = external_torques[:, self.active_idx_torch]  # [num_envs, n_active]

        # Reshape base_stiffness to [num_envs, 1] for broadcasting
        kp = base_stiffness.view(-1, 1).to(device=self.device)

        # Per-env, per-DOF stiffness: K[e,j] = kp[e] * scale[j]
        K = kp * self._scales_t  # [num_envs, n_active]
        M = self._M_active_t     # [n_active] broadcasts

        # Natural frequency and decay
        omega = torch.sqrt(K / M)           # [num_envs, n_active]
        alpha = torch.exp(-omega * self.dt)  # [num_envs, n_active]
        odt = omega * self.dt                # [num_envs, n_active]

        q = self.state["q_def"]
        qd = self.state["qd_def"]

        # Analytical discrete-time coefficients for critically-damped system
        # Ad = exp(-w*dt) * [[1+w*dt, dt], [-w^2*dt, 1-w*dt]]
        # Bd = [[(1 - alpha*(1+w*dt)) / (w^2 * M)], [alpha*dt / M]]
        ad11 = alpha * (1.0 + odt)
        ad12 = alpha * self.dt
        ad21 = -alpha * omega * omega * self.dt
        ad22 = alpha * (1.0 - odt)

        # Clamp omega away from zero to avoid division by zero for zero-stiffness DOFs
        omega_sq_safe = torch.clamp(omega * omega, min=1e-12)
        bd1 = (1.0 - ad11) / (omega_sq_safe * M)
        bd2 = alpha * self.dt / M

        self.state["q_def"][:] = ad11 * q + ad12 * qd + bd1 * tau
        self.state["qd_def"][:] = ad21 * q + ad22 * qd + bd2 * tau

    def reset(self, env_ids: torch.Tensor = None):
        """Reset MSD state to zero for specified environments."""
        if env_ids is None:
            self.state['q_def'][:] = 0.0
            self.state['qd_def'][:] = 0.0
        else:
            self.state['q_def'][env_ids] = 0.0
            self.state['qd_def'][env_ids] = 0.0
