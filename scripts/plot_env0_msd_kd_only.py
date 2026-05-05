# SPDX-License-Identifier: BSD-3-Clause

"""Plot only ||Kq|| and ||Dq'|| from env0_msd_dynamics_terms.csv."""

import argparse
import csv
import sys
from pathlib import Path


def load_rows(path: Path, max_points: int | None):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
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
    parser = argparse.ArgumentParser(description="Plot ||Kq|| and ||Dq'|| only (MSD terms CSV)")
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
        "--logy",
        action="store_true",
        help="Log scale on y (useful when ||Dq'|| is much smaller than ||Kq||)",
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
    if "l2_norm_Kq" not in keys or "l2_norm_Dqdot" not in keys:
        print("CSV missing l2_norm_Kq or l2_norm_Dqdot columns", file=sys.stderr)
        sys.exit(1)
    import matplotlib.pyplot as plt

    xv = column(rows, xkey)
    kq = column(rows, "l2_norm_Kq")
    dqd = column(rows, "l2_norm_Dqdot")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(xv, kq, label="||Kq||", color="C0", linewidth=1.0)
    ax.plot(xv, dqd, label="||Dq'||", color="C1", linewidth=1.0)
    ax.set_xlabel(xkey)
    ax.set_ylabel("L2 norm (log scale)" if args.logy else "L2 norm")
    ax.set_title("MSD elastic and damping magnitudes (env 0)")
    if args.logy:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"Wrote {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    run()
