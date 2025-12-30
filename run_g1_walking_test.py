from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np

from robostl.core.config import DeployConfig
from robostl.attacks.force import ForcePerturbation
from robostl.attacks.observation import GaussianObservationNoise
from robostl.attacks.terrain import FloorFrictionModifier, HeightfieldBump, HeightfieldPit
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G1 12-DoF walking test in MuJoCo (official config)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DeployConfig.default_config_path(),
        help="Path to Unitree deploy_mujoco config yaml.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy path (TorchScript).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("walking_test_results.json"),
        help="Output JSON file for results.",
    )
    parser.add_argument(
        "--stop-on-fall",
        action="store_true",
        help="Stop an episode immediately when fall is detected.",
    )
    parser.add_argument(
        "--no-stop-on-fall",
        dest="stop_on_fall",
        action="store_false",
        help="Continue simulation even after fall detection.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render MuJoCo viewer during simulation.",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable MuJoCo rendering.",
    )
    parser.add_argument(
        "--no-real-time",
        dest="real_time",
        action="store_false",
        help="Run simulation as fast as possible (no real-time sync).",
    )
    parser.add_argument(
        "--push-force",
        type=str,
        default=None,
        help="External force vector fx,fy,fz applied to pelvis.",
    )
    parser.add_argument(
        "--push-torque",
        type=str,
        default=None,
        help="External torque vector tx,ty,tz applied to pelvis.",
    )
    parser.add_argument(
        "--push-start",
        type=float,
        default=1.0,
        help="Push start time in seconds.",
    )
    parser.add_argument(
        "--push-duration",
        type=float,
        default=0.2,
        help="Push duration in seconds.",
    )
    parser.add_argument(
        "--floor-friction",
        type=str,
        default=None,
        help="Override floor friction mu1,mu2,mu3.",
    )
    parser.add_argument(
        "--pit-center",
        type=str,
        default=None,
        help="Static pit center x,y (meters).",
    )
    parser.add_argument(
        "--pit-radius",
        type=float,
        default=0.15,
        help="Pit radius (meters).",
    )
    parser.add_argument(
        "--pit-depth",
        type=float,
        default=0.03,
        help="Pit depth (meters).",
    )
    parser.add_argument(
        "--bump-center",
        type=str,
        default=None,
        help="Static bump center x,y (meters).",
    )
    parser.add_argument(
        "--bump-radius",
        type=float,
        default=0.15,
        help="Bump radius (meters).",
    )
    parser.add_argument(
        "--bump-height",
        type=float,
        default=0.03,
        help="Bump height (meters).",
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std for observations.",
    )
    parser.add_argument(
        "--obs-noise-targets",
        type=str,
        default="omega,gravity,qj,dqj",
        help="Comma-separated obs targets (omega,gravity,cmd,qj,dqj,action,phase).",
    )
    parser.set_defaults(stop_on_fall=True)
    parser.set_defaults(render=True, real_time=True)
    return parser.parse_args()


def _parse_vec3(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {value}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _parse_vec2(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 2 comma-separated values, got: {value}")
    return np.array([float(p) for p in parts], dtype=np.float32)



def main() -> None:
    args = parse_args()

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
    attacks = []

    if args.push_force is not None:
        force = _parse_vec3(args.push_force)
        torque = _parse_vec3(args.push_torque) if args.push_torque else None
        attacks.append(
            ForcePerturbation(
                force=force,
                torque=torque,
                start_time=args.push_start,
                duration=args.push_duration,
            )
        )

    if args.floor_friction is not None:
        friction = _parse_vec3(args.floor_friction)
        attacks.append(FloorFrictionModifier(friction=friction))

    obs_attacks = []
    if args.obs_noise_std and args.obs_noise_std > 0:
        targets = tuple(
            t.strip() for t in args.obs_noise_targets.split(",") if t.strip()
        )
        obs_attacks.append(GaussianObservationNoise(std=args.obs_noise_std, targets=targets))

    if args.pit_center is not None:
        center = _parse_vec2(args.pit_center)
        attacks.append(
            HeightfieldPit(center_xy=center, radius=args.pit_radius, depth=args.pit_depth)
        )
    if args.bump_center is not None:
        center = _parse_vec2(args.bump_center)
        attacks.append(
            HeightfieldBump(
                center_xy=center,
                radius=args.bump_radius,
                height=args.bump_height,
            )
        )

    runner = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=args.stop_on_fall,
        render=args.render,
        real_time=args.real_time,
        attacks=attacks,
        obs_attacks=obs_attacks,
    )

    results = []
    for idx in range(args.episodes):
        result = runner.run_episode()
        results.append({**result.metrics, "stl": result.stl})

    mean_speed = float(np.mean([r["mean_speed_mps"] for r in results]))
    mean_distance = float(np.mean([r["distance_x_m"] for r in results]))
    fall_rate = float(np.mean([1.0 if r["fallen"] else 0.0 for r in results]))
    stl_ok_rate = float(np.mean([1.0 if r["stl"]["ok"] else 0.0 for r in results]))
    stl_robustness = float(np.mean([r["stl"]["robustness"] for r in results]))

    def _jsonify(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    payload = {
        "config": str(args.config),
        "policy_path": str(args.policy_path) if args.policy_path is not None else None,
        "episodes": int(args.episodes),
        "results": results,
        "aggregate": {
            "mean_speed_mps": mean_speed,
            "mean_distance_x_m": mean_distance,
            "fall_rate": fall_rate,
            "stl_ok_rate": stl_ok_rate,
            "stl_robustness": stl_robustness,
        },
    }
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
