from __future__ import annotations
"""Layer 2: 相位搜索（在固定 L1 宏观参数下搜索最坏攻击时刻）。

输入：Layer 1 的 Top-K 参数集合；
输出：每个参数集合在不同相位下的鲁棒性曲线与最坏相位结果。
"""

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.metrics.stability import resolve_ground_geom_ids
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


@dataclass
class PhaseProbeResult:
    """步态周期探测结果。"""
    period_s: float
    raw_period_s: float
    cycle_start_s: float
    contact_times: dict[str, list[float]]
    period_source: str


def parse_args() -> argparse.Namespace:
    """解析 Layer 2 参数。"""
    parser = argparse.ArgumentParser(description="Layer 2 phase search for STL robustness.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("layer1_top_k.json"),
        help="Layer 1 Top-K JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer2_results.json"),
        help="Output JSON file for phase search results.",
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
        help="Override policy path (TorchScript).",
    )
    parser.add_argument("--phase-step", type=float, default=0.05, help="Phase grid step.")
    parser.add_argument("--probe-duration", type=float, default=2.0, help="Probe duration (s).")
    parser.add_argument(
        "--min-step-duration",
        type=float,
        default=0.15,
        help="Minimum contact interval to accept (debounce).",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=0.2,
        help="Minimum gait period before falling back to default.",
    )
    parser.add_argument(
        "--default-period",
        type=float,
        default=0.8,
        help="Fallback gait period when probing is unstable.",
    )
    parser.add_argument("--push-duration", type=float, default=0.2, help="Push duration (s).")
    parser.add_argument("--push-body", type=str, default="pelvis", help="Push body name.")
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="Delay before applying phase-based attacks (seconds).",
    )
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo rendering.")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate Top-K entries in parallel.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker count for parallel evaluation.",
    )
    parser.set_defaults(render=False)
    return parser.parse_args()


def _extract_params(entry: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """从 Layer 1 条目中提取 push / friction。"""
    params = np.array(entry["params"], dtype=np.float32)
    names = entry.get("param_names", [])
    mapping = {name: float(value) for name, value in zip(names, params)}

    push = None
    if all(key in mapping for key in ("push_fx", "push_fy", "push_fz")):
        push = np.array(
            [mapping["push_fx"], mapping["push_fy"], mapping["push_fz"]],
            dtype=np.float32,
        )

    friction = None
    if all(key in mapping for key in ("fric_mu1", "fric_mu2", "fric_mu3")):
        friction = np.array(
            [mapping["fric_mu1"], mapping["fric_mu2"], mapping["fric_mu3"]],
            dtype=np.float32,
        )

    return push, friction, names


def _build_attacks(
    push: Optional[np.ndarray],
    friction: Optional[np.ndarray],
    push_start: Optional[float],
    push_duration: float,
    push_body: str,
) -> list:
    """构建该相位下的攻击列表（摩擦 + 可选推力）。"""
    attacks = []
    if friction is not None:
        attacks.append(FloorFrictionModifier(friction=friction))
    if push is not None and push_start is not None:
        attacks.append(
            ForcePerturbation(
                body_name=push_body,
                force=push,
                start_time=push_start,
                duration=push_duration,
            )
        )
    return attacks


def _build_foot_geom_map(model: mujoco.MjModel) -> dict[int, str]:
    """根据命名规则识别左右脚接触 geom。"""
    foot_keywords = ("foot", "ankle", "toe", "heel", "sole")
    mapping: dict[int, str] = {}
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        name = geom_name.lower()
        body = body_name.lower()
        if not any(keyword in name or keyword in body for keyword in foot_keywords):
            continue
        side = None
        if "left" in name or "left" in body:
            side = "left"
        if "right" in name or "right" in body:
            side = "right"
        if side is not None:
            mapping[geom_id] = side
    return mapping


def _probe_phase(
    env: WalkingTestRunner,
    terrain: Optional[dict],
    duration_s: float,
    min_step_duration: float,
    min_period: float,
    default_period: float,
) -> PhaseProbeResult:
    """通过脚-地接触事件估计步态周期。

    特点：
    - 使用触地上升沿记录接触时刻；
    - 用 `min_step_duration` 去抖；
    - 周期异常时回退到 `default_period`。
    """
    runner_env = env.env
    runner_env.set_terrain(terrain)
    model = runner_env.model
    data = runner_env.data
    ground_geom_ids = resolve_ground_geom_ids(model, env.config.ground_geom_names)
    foot_geom_map = _build_foot_geom_map(model)

    if not ground_geom_ids or not foot_geom_map:
        return PhaseProbeResult(
            period_s=default_period,
            raw_period_s=0.0,
            cycle_start_s=0.0,
            contact_times={},
            period_source="default_missing_contacts",
        )

    contacts = {"left": False, "right": False}
    contact_times = {"left": [], "right": []}

    runner_env.reset()
    steps = int(duration_s / env.config.simulation_dt)
    for _ in range(steps):
        runner_env.step()
        current = {"left": False, "right": False}
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            if geom1 in ground_geom_ids:
                foot_geom = geom2
            elif geom2 in ground_geom_ids:
                foot_geom = geom1
            else:
                continue
            side = foot_geom_map.get(foot_geom)
            if side is not None:
                current[side] = True

        for side in ("left", "right"):
            if current[side] and not contacts[side]:
                last_time = contact_times[side][-1] if contact_times[side] else -1.0
                if runner_env.sim_time - last_time >= min_step_duration:
                    contact_times[side].append(runner_env.sim_time)
            contacts[side] = current[side]

    diffs = []
    for side in ("left", "right"):
        times = contact_times[side]
        if len(times) >= 2:
            diffs.extend([b - a for a, b in zip(times[:-1], times[1:])])

    raw_period_s = float(np.median(diffs)) if diffs else 0.0
    if raw_period_s >= min_period:
        period_s = raw_period_s
        period_source = "measured"
    else:
        period_s = default_period
        period_source = "default_unstable"

    cycle_start_s = 0.0
    if contact_times.get("left"):
        cycle_start_s = float(contact_times["left"][0])
    elif contact_times.get("right"):
        cycle_start_s = float(contact_times["right"][0])

    return PhaseProbeResult(
        period_s=period_s,
        raw_period_s=raw_period_s,
        cycle_start_s=cycle_start_s,
        contact_times=contact_times,
        period_source=period_source,
    )


def _evaluate_entry(
    entry: dict,
    args_dict: dict[str, Any],
    config_path: str,
    policy_path: Optional[str],
) -> dict:
    """评估单个 Layer 1 候选在全相位网格上的鲁棒性。"""
    args = argparse.Namespace(**args_dict)

    config = DeployConfig.from_yaml(Path(config_path))
    if policy_path is not None:
        config = DeployConfig(
            **{
                **config.__dict__,
                "policy_path": Path(policy_path).expanduser().resolve(),
            }
        )

    policy = TorchScriptPolicy(config.policy_path)
    task = WalkingTask.from_config(config)
    runner = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=True,
        render=bool(getattr(args, "render", False)),
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    push, friction, names = _extract_params(entry)
    terrain = entry.get("terrain", {"mode": "flat"})
    runner.env.set_terrain(terrain)
    base_attacks = _build_attacks(
        push=push,
        friction=friction,
        push_start=None,
        push_duration=args.push_duration,
        push_body=args.push_body,
    )
    runner.env.attacks = base_attacks
    probe = _probe_phase(
        runner,
        terrain,
        args.probe_duration,
        args.min_step_duration,
        args.min_period,
        args.default_period,
    )

    phases = np.arange(0.0, 1.0 + 1e-6, args.phase_step)
    phase_results = []
    best = {"phase": None, "robustness": float("inf"), "push_start": None}

    # 将相位起点平移到 settle_time 之后的第一个完整周期，避免初始化阶段干扰。
    cycle_start = probe.cycle_start_s
    if args.settle_time > cycle_start and probe.period_s > 0:
        offset_cycles = math.ceil((args.settle_time - cycle_start) / probe.period_s)
        cycle_start = cycle_start + offset_cycles * probe.period_s

    for phase in phases:
        if push is None:
            push_start = None
        else:
            push_start = cycle_start + float(phase) * probe.period_s
            if push_start + args.push_duration > config.simulation_duration:
                continue
        runner.env.set_terrain(terrain)

        attacks = _build_attacks(
            push=push,
            friction=friction,
            push_start=push_start,
            push_duration=args.push_duration,
            push_body=args.push_body,
        )
        runner.env.attacks = attacks
        episode = runner.run_episode()
        robustness = float(episode.metrics.get("stl_robustness", 0.0))

        phase_results.append(
            {
                "phase": float(phase),
                "push_start": push_start,
                "robustness": robustness,
            }
        )

        if robustness < best["robustness"]:
            best = {
                "phase": float(phase),
                "robustness": robustness,
                "push_start": push_start,
            }

    return {
        "rank": entry.get("rank"),
        "param_names": names,
        "params": entry["params"],
        "terrain": terrain,
        "probe": {
            "period_s": probe.period_s,
            "raw_period_s": probe.raw_period_s,
            "period_source": probe.period_source,
            "cycle_start_s": probe.cycle_start_s,
            "contact_times": probe.contact_times,
        },
        "phase_results": phase_results,
        "best": best,
    }


def main() -> None:
    """Layer 2 主流程。

    支持串行/并行两种评估模式，最终输出按 rank 排序的结果 JSON。
    """
    args = parse_args()
    top_k = json.loads(args.input.read_text(encoding="utf-8"))

    # 并行进程下不使用 viewer，避免上下文冲突。
    if args.parallel and args.render:
        print("[Info] Parallel mode disables rendering to avoid viewer conflicts.")
        args.render = False

    args_dict = {
        "phase_step": args.phase_step,
        "probe_duration": args.probe_duration,
        "push_duration": args.push_duration,
        "push_body": args.push_body,
        "min_step_duration": args.min_step_duration,
        "min_period": args.min_period,
        "default_period": args.default_period,
        "render": args.render,
        "settle_time": args.settle_time,
    }

    results = []
    if args.parallel:
        from concurrent.futures import ProcessPoolExecutor

        config_path = str(args.config)
        policy_path = str(args.policy_path) if args.policy_path is not None else None
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_entry,
                    entry,
                    args_dict,
                    config_path,
                    policy_path,
                )
                for entry in top_k
            ]
            results = [future.result() for future in futures]
    else:
        config_path = str(args.config)
        policy_path = str(args.policy_path) if args.policy_path is not None else None
        for entry in top_k:
            results.append(_evaluate_entry(entry, args_dict, config_path, policy_path))

    results = sorted(results, key=lambda r: r.get("rank", 0))

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Layer 2 results saved to {args.output}")


if __name__ == "__main__":
    main()
