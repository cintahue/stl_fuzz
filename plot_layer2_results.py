from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Layer 2 phase search results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("layer2_results.json"),
        help="Layer 2 results JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer2_analysis.png"),
        help="Output image file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Robustness vs. Attack Phase (Layer 2 Search)",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    data = json.loads(args.input.read_text(encoding="utf-8"))

    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap("tab10")

    for idx, entry in enumerate(data):
        phase_res = entry.get("phase_results", [])
        if not phase_res:
            continue
        phases = [item["phase"] for item in phase_res]
        robustness = [item["robustness"] for item in phase_res]
        rank = entry.get("rank", idx + 1)
        plt.plot(
            phases,
            robustness,
            marker="o",
            linewidth=1.5,
            label=f"Rank {rank}",
            color=cmap(idx % 10),
        )

    plt.xlabel("Gait Phase (0.0 - 1.0)")
    plt.ylabel("STL Robustness")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
