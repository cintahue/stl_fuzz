from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline calibration for STL thresholds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DeployConfig.default_config_path(),
        help="Path to deploy config yaml.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of baseline episodes to run.",
    )
    parser.add_argument(
        "--skip-initial-s",
        type=float,
        default=1.0,
        help="Skip initial seconds when collecting stats.",
    )
    parser.add_argument(
        "--commands",
        type=str,
        default="1,0,0;0.5,0,0;1,0.3,0;0,0,0.5",
        help="Semicolon-separated cmd list, each as vx,vy,wz.",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=Path("baseline_stats.json"),
        help="Output JSON stats path.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=False)
    return parser.parse_args()


def _parse_commands(value: str) -> list[np.ndarray]:
    commands = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        items = [p.strip() for p in part.split(",")]
        if len(items) != 3:
            raise ValueError(f"Expected 3 values in command, got: {part}")
        commands.append(np.array([float(x) for x in items], dtype=np.float32))
    if not commands:
        raise ValueError("No commands provided.")
    return commands


def _collect_values(trace, signal: str, skip_initial_s: float) -> np.ndarray:
    if signal not in trace.signals:
        return np.array([], dtype=np.float32)
    time = trace.time
    mask = time >= float(skip_initial_s)
    values = trace.signals[signal][mask]
    values = values[np.isfinite(values)]
    return values.astype(np.float32)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "P1": _percentile(values, 1),
        "P5": _percentile(values, 5),
        "P99": _percentile(values, 99),
        "P99_9": _percentile(values, 99.9),
    }


def _recommend_thresholds(stats: dict[str, dict[str, Any]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    height = stats.get("height", {})
    if height:
        thresholds["h_min"] = float(height["min"]) - 2.0 * float(height["std"])

    tilt = stats.get("tilt", {})
    if tilt:
        thresholds["max_tilt_deg"] = float(tilt["max"]) + 2.0 * float(tilt["std"])

    torque = stats.get("max_torque", {})
    if torque:
        thresholds["max_torque"] = float(torque["P99_9"]) * 1.1

    zmp = stats.get("zmp_margin", {})
    if zmp:
        thresholds["zmp_margin"] = float(zmp["P1"]) * 0.8

    ang = stats.get("angular_velocity", {})
    if ang:
        thresholds["max_angular_velocity"] = float(ang["P99"]) * 1.2

    vel = stats.get("velocity_error", {})
    if vel:
        thresholds["max_velocity_error"] = float(vel["max"]) + 2.0 * float(vel["std"])

    foot = stats.get("foot_clearance", {})
    if foot:
        thresholds["min_foot_clearance"] = float(foot["min"]) * 0.5

    action = stats.get("action_delta", {})
    if action:
        thresholds["max_action_delta"] = float(action["P99"]) * 1.2

    stability = stats.get("stability_margin", {})
    if stability:
        thresholds["stability_margin"] = float(stability["P1"]) * 0.8

    return thresholds


def main() -> None:
    args = parse_args()
    commands = _parse_commands(args.commands)

    config = DeployConfig.from_yaml(args.config)
    if args.policy_path is not None:
        config = DeployConfig(
            **{
                **config.__dict__,
                "policy_path": args.policy_path.expanduser().resolve(),
            }
        )

    policy = TorchScriptPolicy(config.policy_path)
    task = WalkingTask.from_config(config)
    runner = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=False,
        render=args.render,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    all_values: dict[str, list[np.ndarray]] = {
        "height": [],
        "tilt": [],
        "max_torque": [],
        "zmp_margin": [],
        "stability_margin": [],
        "angular_velocity": [],
        "velocity_error": [],
        "foot_clearance": [],
        "action_delta": [],
    }
    total_steps = 0

    for idx in range(args.episodes):
        cmd = commands[idx % len(commands)]
        runner.env.cmd = cmd.copy()
        runner.run_episode()
        trace = runner.metrics.build_trace()
        total_steps += len(trace.time)
        for key in all_values:
            all_values[key].append(_collect_values(trace, key, args.skip_initial_s))

    stats: dict[str, Any] = {}
    for key, chunks in all_values.items():
        if chunks:
            merged = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
            stats[key] = _stats(merged)

    thresholds = _recommend_thresholds(stats)
    thresholds["evaluation_start_time"] = float(args.skip_initial_s)

    baseline_payload = {
        "baseline_stats": {
            "episodes": int(args.episodes),
            "total_steps": int(total_steps),
            "skip_initial_s": float(args.skip_initial_s),
            **stats,
        },
        "calibrated_thresholds": thresholds,
    }
    args.output_stats.write_text(
        json.dumps(baseline_payload, indent=2), encoding="utf-8"
    )

    spec = dict(config.stl_config)
    spec.update(thresholds)
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_config["stl"] = spec
    args.config.write_text(
        yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8"
    )

    print(f"Baseline stats saved to {args.output_stats}")
    print(f"Calibrated STL config updated in {args.config}")


if __name__ == "__main__":
    main()
