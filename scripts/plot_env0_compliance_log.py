# SPDX-License-Identifier: BSD-3-Clause

"""Plot time series from env0_compliance_log.csv (extended MSD or legacy six-column layout)."""

import argparse
import csv
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
    return names, rows


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


def plot_extended(rows, xkey: str, output: Path | None):
    import matplotlib.pyplot as plt

    xv = column(rows, xkey)
    fz = column(rows, "force_z_event")
    if all(x != x for x in fz):
        fz = column(rows, "force_z")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax0 = axes[0, 0]
    ax0.plot(xv, fz, color="C0", linewidth=0.8)
    ax0.set_ylabel("normal force (event)")
    ax0.set_title("Vertical force")
    ax0.grid(True, alpha=0.3)
    ax1 = axes[0, 1]
    for label, color in (("x_def_x", "C0"), ("x_def_y", "C1"), ("x_def_z", "C2")):
        yv = column(rows, label)
        ax1.plot(xv, yv, label=label, color=color, linewidth=0.8)
    ax1.set_ylabel("deformation")
    ax1.set_title("MSD deformation (base)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1, 0]
    for label, color in (("dx_def_x", "C0"), ("dx_def_y", "C1"), ("dx_def_z", "C2")):
        yv = column(rows, label)
        ax2.plot(xv, yv, label=label, color=color, linewidth=0.8)
    ax2.set_ylabel("deformation rate")
    ax2.set_title("MSD deformation rate")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 1]
    kp = column(rows, "kp")
    ob = column(rows, "omega_base")
    ax3.plot(xv, kp, color="C3", linewidth=0.8, label="kp")
    ax3.set_ylabel("kp", color="C3")
    ax3b = ax3.twinx()
    ax3b.plot(xv, ob, color="C4", linewidth=0.8, label="omega_base")
    ax3b.set_ylabel("omega_base", color="C4")
    ax3.set_title("Stiffness and nominal base frequency")
    ax3.grid(True, alpha=0.3)
    fig.supxlabel(xkey)
    if output is not None:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()


def plot_legacy(rows, xkey: str, output: Path | None):
    import matplotlib.pyplot as plt

    xv = column(rows, xkey)
    fz = column(rows, "force_z")
    xd = [column(rows, f"x_def_{a}") for a in ("x", "y", "z")]
    lengths = [len(xv), len(fz)] + [len(c) for c in xd]
    n = min(lengths) if lengths else 0
    xv = xv[:n]
    fz = fz[:n]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(xv, fz, color="C0", linewidth=0.8)
    axes[0].set_ylabel("force_z")
    axes[0].set_title("Legacy log: force")
    axes[0].grid(True, alpha=0.3)
    for label, series, color in zip(("x", "y", "z"), xd, ("C0", "C1", "C2")):
        axes[1].plot(xv, series[:n], label=f"x_def_{label}", color=color, linewidth=0.8)
    axes[1].set_ylabel("deformation")
    axes[1].set_title("MSD deformation (base)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.supxlabel(xkey)
    if output is not None:
        fig.savefig(output, dpi=150)
        print(f"Wrote {output}")
    else:
        plt.show()


def run():
    parser = argparse.ArgumentParser(description="Plot env0_compliance_log.csv")
    parser.add_argument("csv", type=Path, help="Path to env0_compliance_log.csv")
    parser.add_argument(
        "--x",
        choices=("step", "sim_time", "approx_learning_iter"),
        default="step",
        help="Horizontal axis column (default: step; use approx_learning_iter for extended logs)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Downsample to about this many rows for faster plotting",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (PNG). If omitted, open an interactive window",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    _, rows = load_rows(args.csv, args.max_points)
    if not rows:
        print("CSV has no data rows", file=sys.stderr)
        sys.exit(1)
    header = set(rows[0].keys())
    xkey = args.x
    if xkey not in header and "step" in header:
        xkey = "step"
    extended = "force_z_event" in header or ("kp" in header and "x_def_x" in header)
    if extended:
        plot_extended(rows, xkey, args.output)
    else:
        plot_legacy(rows, xkey, args.output)


if __name__ == "__main__":
    run()
