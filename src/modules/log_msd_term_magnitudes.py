# SPDX-License-Identifier: BSD-3-Clause

"""Log MSD dynamics for env 0: norms of Kq, Dq', Mq'', F, plus q, q', q''.

``l2_norm_Mqdd`` is ``||F - Dq' - Kq||`` (same vector as ``M q''`` in the diagonal model).
``l2_norm_qdd`` is ``||q''||`` with ``q'' = (F - Dq' - Kq) / M`` element-wise, so it grows
when the scalar inertia ``M`` is reduced for the same residual force.
``l2_norm_q`` and ``l2_norm_qdot`` are ``||q||`` and ``||q'||`` in the same active MSD
deflection coordinates as ``q_c*`` / ``qdot_c*`` / ``qdd_c*``.
"""

from __future__ import annotations

import csv
import os

import torch

from src.compliance.utils.dynamics import get_wrench


def _resolved_log_path(env, log_path: str) -> str:
    if os.path.isabs(log_path):
        return log_path
    base = getattr(env.cfg, "log_dir", None)
    if isinstance(base, str) and len(base) > 0:
        return os.path.join(base, os.path.basename(log_path))
    return log_path


def log_env0_msd_term_magnitudes(
    env,
    env_ids: torch.Tensor,
    log_path: str = "env0_msd_dynamics_terms.csv",
    num_steps_per_env: int | None = 24,
    enabled: bool = False,
):
    """Append one CSV row: force-split norms, ``q``, ``q'``, ``q''`` norms and components.

    When ``enabled`` is False (default), the event is a no-op so training stays free of CSV I/O.
    Set ``enabled`` True via the event term ``params`` or env cfg (see task env ``__post_init__``).
    """
    if not enabled:
        return
    if not hasattr(env, "compliance_manager") or env.compliance_manager is None:
        return
    cm = env.compliance_manager
    msd = cm._msd_system
    if msd is None or msd.n_active == 0:
        return
    device = env.device
    names = list(cm._compliant_body_names)
    wrench = get_wrench(cm._robot, names)
    forces = wrench[:, :, :3]
    flat = forces.reshape(forces.shape[0], -1)
    idx = msd.active_idx_torch
    f0 = flat[0, idx]
    q = msd.state["x_def"][0].to(device=device, dtype=torch.float32)
    qd = msd.state["dx_def"][0].to(device=device, dtype=torch.float32)
    scales = torch.tensor(
        [float(msd.stiffness_scales[int(i)]) for i in msd.active_idx],
        dtype=torch.float32,
        device=device,
    )
    base_stiffness = None
    if hasattr(env, "command_manager"):
        try:
            kp = env.command_manager.get_command("stiffness")[:, 0]
            base_stiffness = kp[0]
        except (KeyError, RuntimeError, AttributeError):
            base_stiffness = None
    if base_stiffness is not None:
        kvec = base_stiffness * scales
        m0 = float(msd.M[int(msd.active_idx[0])])
        mvec = torch.full_like(kvec, m0)
        dvec = 2.0 * torch.sqrt(mvec * kvec)
    else:
        kvec = torch.tensor(msd.K[msd.active_idx], dtype=torch.float32, device=device)
        dvec = torch.tensor(msd.D[msd.active_idx], dtype=torch.float32, device=device)
        mvec = torch.tensor(msd.M[msd.active_idx], dtype=torch.float32, device=device)
    kq = kvec * q
    dqd = dvec * qd
    qdd = (f0 - dqd - kq) / mvec
    mqdd = mvec * qdd
    n_kq = float(torch.linalg.norm(kq).item())
    n_dqd = float(torch.linalg.norm(dqd).item())
    n_mqdd = float(torch.linalg.norm(mqdd).item())
    n_f = float(torch.linalg.norm(f0).item())
    n_qdd = float(torch.linalg.norm(qdd).item())
    n_q = float(torch.linalg.norm(q).item())
    n_qdot = float(torch.linalg.norm(qd).item())
    q_components = [float(q[i].item()) for i in range(int(msd.n_active))]
    qdot_components = [float(qd[i].item()) for i in range(int(msd.n_active))]
    qdd_components = [float(qdd[i].item()) for i in range(int(msd.n_active))]
    step = int(env.common_step_counter)
    t = float(step) * float(env.step_dt)
    approx = 0
    if isinstance(num_steps_per_env, int) and num_steps_per_env > 0:
        approx = step // int(num_steps_per_env)
    path = _resolved_log_path(env, log_path)
    if not hasattr(env, "_msd_terms_writer"):
        fh = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        header = [
            "step",
            "sim_time",
            "approx_learning_iter",
            "l2_norm_Kq",
            "l2_norm_Dqdot",
            "l2_norm_Mqdd",
            "l2_norm_F",
            "l2_norm_qdd",
            "l2_norm_q",
            "l2_norm_qdot",
        ]
        header.extend([f"q_c{i}" for i in range(int(msd.n_active))])
        header.extend([f"qdot_c{i}" for i in range(int(msd.n_active))])
        header.extend([f"qdd_c{i}" for i in range(int(msd.n_active))])
        w.writerow(header)
        env._msd_terms_file = fh
        env._msd_terms_writer = w
    row = [
        step,
        f"{t:.6f}",
        approx,
        f"{n_kq:.8f}",
        f"{n_dqd:.8f}",
        f"{n_mqdd:.8f}",
        f"{n_f:.8f}",
        f"{n_qdd:.8f}",
        f"{n_q:.8f}",
        f"{n_qdot:.8f}",
    ]
    row.extend([f"{c:.8f}" for c in q_components])
    row.extend([f"{c:.8f}" for c in qdot_components])
    row.extend([f"{c:.8f}" for c in qdd_components])
    env._msd_terms_writer.writerow(row)
    env._msd_terms_file.flush()
