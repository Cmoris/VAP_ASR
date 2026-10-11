"""
Example: generate a nodding pitch time series with ``maai.util.generate_natural_nod``
and plot it with matplotlib.

Cubic spline interpolation requires **scipy** (ImportError if missing).
"""

from __future__ import annotations

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

from maai.util import generate_natural_nod


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate and visualize nodding motion (cubic spline / scipy)."
    )
    p.add_argument(
        "--range-rad",
        type=float,
        default=0.2,
        help="Nod depth in radians (absolute value).",
    )
    p.add_argument("--count", type=int, default=2, help="Number of nods (>=1).")
    p.add_argument(
        "--no-pre-rise",
        action="store_true",
        help="Disable the initial pre-rise segment.",
    )
    p.add_argument(
        "--velocity",
        type=float,
        default=0.75,
        help="Target mean angular velocity (rad/s).",
    )
    p.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    p.add_argument(
        "--decay",
        type=float,
        default=0.6,
        help="Amplitude decay factor per nod (0–1).",
    )
    p.add_argument(
        "--pre-rise-ratio",
        type=float,
        default=0.8,
        help="Pre-rise amplitude as a fraction of range_rad.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Path to save a PNG (omit to only show the figure).",
    )
    args = p.parse_args()

    motion, time_axis = generate_natural_nod(
        range_rad=float(args.range_rad),
        count=int(args.count),
        use_pre_rise=not args.no_pre_rise,
        velocity=float(args.velocity),
        fps=int(args.fps),
        decay_rate=float(args.decay),
        pre_rise_ratio=float(args.pre_rise_ratio),
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, motion, color="steelblue", linewidth=1.8, label="pitch (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (rad)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    ax_deg = ax.twinx()
    ax_deg.plot(time_axis, np.rad2deg(motion), color="coral", alpha=0.35, linewidth=1.0)
    ax_deg.set_ylabel("Pitch (deg)")

    title = (
        f"Natural nod motion (count={args.count}, pre_rise={not args.no_pre_rise}, "
        f"range={args.range_rad:.3f} rad, fps={args.fps})"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("scipy is required, e.g.: pip install scipy", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.")
