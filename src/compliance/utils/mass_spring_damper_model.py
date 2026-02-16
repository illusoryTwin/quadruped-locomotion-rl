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
        self.stiffness_scales = stiffness_scales

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

        # MSD state: batched [num_envs, n_active] for q_def and dx_def
        self.state = self._create_msd_state()

    def set_stiffness(self, base_stiffness: float):
        """Update base stiffness and recompute MSD matrices.

        Args:
            base_stiffness: New base stiffness value, scaled per-DOF by stiffness_scales.
        """
        # Recompute stiffness for active DOFs
        self.K = np.zeros(self.n_dofs)
        for dof_idx, scale in self.stiffness_scales.items():
            self.K[dof_idx] = base_stiffness * scale

        # Recompute critical damping: D = 2*sqrt(M*K)
        self.D = 2 * np.sqrt(self.M * self.K)

        # Recompute discrete-time matrices
        Ad_np, Bd_np = self._compute_discrete_matrices()
        self.Ad = torch.tensor(Ad_np, dtype=torch.float32, device=self.device)
        self.Bd = torch.tensor(Bd_np, dtype=torch.float32, device=self.device)

    def _create_msd_state(self):
        """Create MSD state dictionary for tracking deformations in task space."""
        return {
            'x_def': torch.zeros((self.num_envs, self.n_active), dtype=torch.float32, device=self.device),
            'dx_def': torch.zeros((self.num_envs, self.n_active), dtype=torch.float32, device=self.device),
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

    def update_msd_state_discrete(self, external_forces: torch.Tensor):
        """
        Update Mass-Spring-Damper state using analytical discrete-time solution.
        Batched computation for all environments.

        Uses precomputed discrete-time state-space matrices:
            x[k+1] = Ad*x[k] + Bd*u[k]
            where x = [x_def; dx_def], u = tau_ext (active DOFs only)

        Args:
            external_forces: External forces on joints [num_envs, n_dofs] - full DOF vector
        """
        if self.n_active == 0:
            return  # No active DOFs

        # Ensure forces are on the correct device
        external_forces = external_forces.to(device=self.device)

        # Extract only active DOF forces: [num_envs, n_active]
        external_forces_active = external_forces[:, self.active_idx_torch]

        # Pack state vector: x = [x_def; dx_def] (active DOFs only)
        x = torch.cat([self.state["x_def"], self.state["dx_def"]], dim=1)

        # Discrete-time state update: x[k+1] = Ad @ x[k] + Bd @ u[k]
        x_next = x @ self.Ad.T + external_forces_active @ self.Bd.T

        # Unpack state vector (active DOFs only)
        self.state["x_def"][:] = x_next[:, :self.n_active]
        self.state["dx_def"][:] = x_next[:, self.n_active:]

    def update_with_variable_stiffness(
        self, external_forces: torch.Tensor, base_stiffness: torch.Tensor
    ):
        """Update MSD state with per-environment variable stiffness.

        Uses analytical critically-damped solution for diagonal MSD systems.
        Each DOF is independent, so we compute per-env stiffness scaling.

        Args:
            external_forces: External forces [num_envs, n_dofs]
            base_stiffness: Per-env base stiffness [num_envs]
        """
        if self.n_active == 0:
            return

        external_forces = external_forces.to(device=self.device)
        external_forces_active = external_forces[:, self.active_idx_torch]

        # Per-DOF stiffness scales (from config): [n_active]
        scales = torch.tensor(
            [self.stiffness_scales[idx] for idx in self.active_idx],
            dtype=torch.float32, device=self.device,
        )

        # Per-env, per-DOF stiffness: K[e,d] = base_stiffness[e] * scale[d]
        # base_stiffness: [num_envs] -> [num_envs, 1]
        K = base_stiffness.unsqueeze(-1) * scales.unsqueeze(0)  # [num_envs, n_active]

        # Per-DOF inertia (constant)
        M = self.M[self.active_idx[0]]  # scalar, same for all DOFs

        # Critical damping: D = 2*sqrt(M*K)
        D = 2.0 * torch.sqrt(M * K)

        # Analytical 2nd-order critically damped discrete update:
        # For each DOF independently: m*q'' + d*q' + k*q = F
        # omega = sqrt(k/m), state update via exact discretization
        omega = torch.sqrt(K / M)  # [num_envs, n_active]
        exp_term = torch.exp(-omega * self.dt)

        q = self.state["x_def"]
        qd = self.state["dx_def"]

        # Steady-state displacement from external torque: q_ss = tau / K
        # Clamp K to avoid division by zero
        q_ss = external_forces_active / K.clamp(min=1e-6)

        # Critically damped homogeneous solution:
        # q(t) = (C1 + C2*t) * exp(-omega*t)  +  q_ss
        # q'(t) = (C2 - omega*(C1 + C2*t)) * exp(-omega*t)
        # At t=0: C1 = q - q_ss, C2 = qd + omega*C1
        C1 = q - q_ss
        C2 = qd + omega * C1

        # Evaluate at t = dt
        self.state["x_def"][:] = (C1 + C2 * self.dt) * exp_term + q_ss
        self.state["dx_def"][:] = (C2 - omega * (C1 + C2 * self.dt)) * exp_term

    def reset(self, env_ids: torch.Tensor = None):
        """Reset MSD state to zero for specified environments."""
        if env_ids is None:
            self.state['x_def'][:] = 0.0
            self.state['dx_def'][:] = 0.0
        else:
            self.state['x_def'][env_ids] = 0.0
            self.state['dx_def'][env_ids] = 0.0
