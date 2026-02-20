from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a failure case with MuJoCo visualization."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Failure case directory or case.json path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override deploy config yaml.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy path.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable MuJoCo rendering.",
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
        help="Run simulation as fast as possible.",
    )
    parser.add_argument(
        "--stop-on-fall",
        action="store_true",
        help="Stop simulation immediately when a fall is detected.",
    )
    parser.add_argument(
        "--no-stop-on-fall",
        dest="stop_on_fall",
        action="store_false",
        help="Continue simulation after fall detection.",
    )
    parser.set_defaults(render=True, real_time=True, stop_on_fall=True)
    return parser.parse_args()


def _load_case(input_path: Path) -> dict:
    if input_path.is_dir():
        case_file = input_path / "case.json"
    else:
        case_file = input_path
    return json.loads(case_file.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    entry = _load_case(args.input)

    config_path = args.config or Path(
        entry.get("config", DeployConfig.default_config_path())
    )
    config = DeployConfig.from_yaml(config_path)
    if args.policy_path is not None:
        config = DeployConfig(
            **{
                **config.__dict__,
                "policy_path": args.policy_path.expanduser().resolve(),
            }
        )
    elif entry.get("policy_path"):
        config = DeployConfig(
            **{
                **config.__dict__,
                "policy_path": Path(entry["policy_path"]).expanduser().resolve(),
            }
        )

    policy = TorchScriptPolicy(config.policy_path)
    task = WalkingTask.from_config(config)
    runner = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=args.stop_on_fall,
        render=args.render,
        real_time=args.real_time,
        attacks=[],
        obs_attacks=None,
    )

    cmd = np.array(entry.get("cmd", task.command), dtype=np.float32)
    runner.env.cmd = cmd.copy()
    terrain = entry.get("terrain", {"mode": "flat"})
    runner.env.set_terrain(terrain)

    push = np.array(entry.get("push", [0.0, 0.0, 0.0]), dtype=np.float32)
    friction = np.array(entry.get("friction", [1.0, 0.005, 0.0001]), dtype=np.float32)
    push_start = float(entry.get("push_start", 0.0))
    push_duration = float(entry.get("push_duration", 0.2))
    push_body = str(entry.get("push_body", "pelvis"))

    attacks = [
        FloorFrictionModifier(friction=friction),
        ForcePerturbation(
            body_name=push_body,
            force=push,
            start_time=push_start,
            duration=push_duration,
        ),
    ]
    runner.env.attacks = attacks

    joint_pos_offset = (
        np.array(entry.get("joint_pos_offset"), dtype=np.float32)
        if entry.get("joint_pos_offset") is not None
        else None
    )
    joint_vel_offset = (
        np.array(entry.get("joint_vel_offset"), dtype=np.float32)
        if entry.get("joint_vel_offset") is not None
        else None
    )

    runner.run_episode_with_midpoint_perturbation(
        perturbation_time=push_start,
        joint_pos_offset=joint_pos_offset,
        joint_vel_offset=joint_vel_offset,
    )


if __name__ == "__main__":
    main()
