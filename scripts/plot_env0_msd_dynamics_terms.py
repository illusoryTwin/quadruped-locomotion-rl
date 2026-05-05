# SPDX-License-Identifier: BSD-3-Clause

"""Plot columns from env0_msd_dynamics_terms.csv (MSD term L2 norms, q, q', q'')."""

import argparse
import csv
import re
import sys
from pathlib import Path


def load_rows(path: Path, max_points: int | None):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names:
            print("CSV has no header row", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)
    if max_points is not None and len(rows) > int(max_points) > 0:
        stride = max(1, len(rows) // int(max_points))
        rows = rows[::stride]
    return rows


def column(rows, key: str, default=float("nan")):
    out = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            out.append(default)
            continue
        try:
            out.append(float(value))
        except ValueError:
            out.append(default)
    return out


def run():
    parser = argparse.ArgumentParser(description="Plot env0_msd_dynamics_terms.csv")
    parser.add_argument("csv", type=Path, help="Path to env0_msd_dynamics_terms.csv")
    parser.add_argument(
        "--x",
        choices=("step", "sim_time", "approx_learning_iter"),
        default="step",
        help="Horizontal axis column",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Downsample to about this many rows",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure as PNG (omit to open interactive window)",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    rows = load_rows(args.csv, args.max_points)
    if not rows:
        print("CSV has no data rows", file=sys.stderr)
        sys.exit(1)
    keys = set(rows[0].keys())
    xkey = args.x
    if xkey not in keys and "step" in keys:
        xkey = "step"
    import matplotlib.pyplot as plt

    xv = column(rows, xkey)
    kq = column(rows, "l2_norm_Kq")
    dqd = column(rows, "l2_norm_Dqdot")
    mq = column(rows, "l2_norm_Mqdd")
    fn = column(rows, "l2_norm_F")
    qdd_keys = sorted(
        [k for k in keys if re.fullmatch(r"qdd_c\d+", k)],
        key=lambda s: int(s.replace("qdd_c", "")),
    )
    q_keys = sorted(
        [k for k in keys if re.fullmatch(r"q_c\d+", k)],
        key=lambda s: int(s.replace("q_c", "")),
    )
    qdot_keys = sorted(
        [k for k in keys if re.fullmatch(r"qdot_c\d+", k)],
        key=lambda s: int(s.replace("qdot_c", "")),
    )
    has_qdd = "l2_norm_qdd" in keys
    has_q = "l2_norm_q" in keys
    ax_q = None
    ax1 = None
    if has_q and has_qdd:
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
        ax0, ax_q, ax1 = axes[0], axes[1], axes[2]
    elif has_qdd:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        ax0, ax1 = axes[0], axes[1]
    elif has_q:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        ax0, ax_q = axes[0], axes[1]
    else:
        fig, ax0 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax0.plot(xv, kq, label="||Kq||", color="C0", linewidth=0.9)
    ax0.plot(xv, dqd, label="||Dq'||", color="C1", linewidth=0.9)
    ax0.plot(xv, mq, label="||Mq''|| (=||F-Dq'-Kq||)", color="C2", linewidth=0.9)
    ax0.plot(xv, fn, label="||F||", color="C3", linewidth=0.9, alpha=0.85)
    ax0.set_xlabel(xkey)
    ax0.set_ylabel("L2 norm (forces)")
    ax0.set_title("MSD split magnitudes (env 0)")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    if ax_q is not None:
        ax_q.plot(xv, column(rows, "l2_norm_q"), label="||q||", color="C5", linewidth=1.0)
        ax_q.plot(xv, column(rows, "l2_norm_qdot"), label="||q'||", color="C6", linewidth=1.0)
        for i, k in enumerate(q_keys):
            ax_q.plot(xv, column(rows, k), linewidth=0.7, alpha=0.85, label=k)
        for i, k in enumerate(qdot_keys):
            ax_q.plot(xv, column(rows, k), linewidth=0.7, alpha=0.85, linestyle="--", label=k)
        ax_q.set_xlabel(xkey)
        ax_q.set_ylabel("deflection / rate (active MSD coords)")
        ax_q.set_title("q and q' (same coords as Kq, Dq')")
        ax_q.legend(loc="best", fontsize=7)
        ax_q.grid(True, alpha=0.3)
    if ax1 is not None:
        nq = column(rows, "l2_norm_qdd")
        ax1.plot(xv, nq, label="||q''||", color="C4", linewidth=1.0)
        for i, k in enumerate(qdd_keys):
            ax1.plot(xv, column(rows, k), linewidth=0.7, alpha=0.85, label=k)
        ax1.set_xlabel(xkey)
        ax1.set_ylabel("acceleration (diag MSD coords)")
        ax1.set_title("q'' from (F - Dq' - Kq) / M")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    run()
