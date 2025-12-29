from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Layer 3 Verification & Diagnosis Results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("layer3_results.json"),
        help="Layer 3 results JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer3_diagnosis.png"),
        help="Output image file.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Specific action dimension to plot (optional).",
    )
    parser.add_argument(
        "--case-rank",
        type=int,
        default=None,
        help="Which case_rank to plot if input contains multiple entries.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if args.case_rank is not None:
            entry = next(
                (item for item in data if int(item.get("case_rank", -1)) == args.case_rank),
                None,
            )
            if entry is None:
                raise SystemExit(f"case_rank {args.case_rank} not found in input.")
        else:
            entry = data[0] if data else None
        if entry is None:
            raise SystemExit("No entries found in input.")
    else:
        entry = data

    alphas = np.array(entry["exactline"]["alphas"], dtype=np.float32)
    actions = np.array(entry["exactline"]["actions"], dtype=np.float32)
    crown_lower = np.array(entry["crown"]["lower"], dtype=np.float32)
    crown_upper = np.array(entry["crown"]["upper"], dtype=np.float32)
    nominal = np.array(entry["crown"]["output"], dtype=np.float32)

    if args.dim is None:
        variation = actions.max(axis=0) - actions.min(axis=0)
        target_dim = int(np.argmax(variation))
        print(f"Auto-selected dimension with max variation: {target_dim}")
    else:
        target_dim = int(args.dim)

    if target_dim < 0 or target_dim >= actions.shape[1]:
        raise SystemExit(f"Invalid dim {target_dim}, action dims={actions.shape[1]}")

    x_vals = alphas
    y_vals = actions[:, target_dim]

    plt.figure(figsize=(10, 6))

    lb = float(crown_lower[target_dim])
    ub = float(crown_upper[target_dim])
    plt.axhspan(
        lb,
        ub,
        color="red",
        alpha=0.1,
        label="Linearized Bounds (Jacobian)",
    )

    plt.plot(x_vals, y_vals, color="blue", linewidth=2.5, label="Exact Response")
    plt.scatter(
        [0.0],
        [float(nominal[target_dim])],
        color="black",
        s=80,
        zorder=5,
        label="Nominal (t*)",
    )

    slopes = np.abs(np.diff(y_vals) / (np.diff(x_vals) + 1e-9))
    if slopes.size:
        avg_slope = float(np.mean(slopes))
        kink_indices = np.where(slopes > avg_slope * 1.5)[0]
        if kink_indices.size:
            plt.scatter(
                x_vals[kink_indices],
                y_vals[kink_indices],
                color="orange",
                s=16,
                alpha=0.6,
                label="Nonlinear Kinks",
            )

    plt.title(
        f"Layer 3 Diagnosis: Rank {entry.get('case_rank')} @ t={entry.get('t_star'):.3f}s\n"
        f"Action Dimension {target_dim}"
    )
    plt.xlabel(f"Input Perturbation (epsilon = {entry.get('epsilon')})")
    plt.ylabel("Action Output")
    plt.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")

    max_dev = float(np.max(np.abs(y_vals - nominal[target_dim])))
    linear_dev = max(abs(ub - nominal[target_dim]), abs(lb - nominal[target_dim]))
    non_linearity_ratio = max_dev / (linear_dev + 1e-9)

    info_text = (
        f"Max Deviation: {max_dev:.4f}\n"
        f"Linear Est: {linear_dev:.4f}\n"
        f"Non-linearity: {non_linearity_ratio:.2f}x"
    )
    plt.text(
        0.95,
        0.05,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    if args.show:
        plt.show()

    print(f"Diagnosis plot saved to {args.output}")


if __name__ == "__main__":
    main()
