from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.metrics.stability import resolve_ground_geom_ids
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.run_layer3_search import (
    CROWNSensitivityAnalyzer,
    Layer3SearchConfig,
    Layer3StateSearcher,
)
from robostl.tasks.walking import WalkingTask


@dataclass
class SeedEntry:
    push: np.ndarray
    friction: np.ndarray
    phase: float
    push_start: float
    robustness: float
    keep_score: float
    joint_pos_offset: Optional[np.ndarray]
    joint_vel_offset: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated L1->L2->L3 fuzzing loop."
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
    parser.add_argument("--iterations", type=int, default=50, help="Loop iterations.")
    parser.add_argument("--seed-pool-size", type=int, default=200, help="Seed pool capacity.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("robostl/fuzz_outputs"),
        help="Output directory for seed pool and failures.",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="seeds",
        help="Directory for seed json files.",
    )
    parser.add_argument(
        "--failure-dir",
        type=str,
        default="failures",
        help="Directory for failure cases.",
    )
    parser.add_argument(
        "--random-zero-prob",
        type=float,
        default=0.1,
        help="Probability to zero macro perturbations in random mode.",
    )
    parser.add_argument(
        "--exploit-cap",
        type=float,
        default=0.7,
        help="Max exploitation ratio.",
    )
    parser.add_argument(
        "--push-min",
        type=str,
        default="-80,-80,-80",
        help="Min push force vector.",
    )
    parser.add_argument(
        "--push-max",
        type=str,
        default="80,80,80",
        help="Max push force vector.",
    )
    parser.add_argument(
        "--friction-min",
        type=str,
        default="0.2,0.001,0.00001",
        help="Min friction vector.",
    )
    parser.add_argument(
        "--friction-max",
        type=str,
        default="2.0,0.02,0.01",
        help="Max friction vector.",
    )
    parser.add_argument(
        "--friction-default",
        type=str,
        default="1.0,0.005,0.0001",
        help="Default friction vector for zero macro case.",
    )
    parser.add_argument(
        "--phase-step",
        type=float,
        default=0.05,
        help="Phase grid step.",
    )
    parser.add_argument(
        "--phase-mutation-range",
        type=float,
        default=0.1,
        help="Phase mutation range in exploit mode.",
    )
    parser.add_argument(
        "--phase-mutation-samples",
        type=int,
        default=5,
        help="Phase samples in exploit mode.",
    )
    parser.add_argument("--probe-duration", type=float, default=2.0)
    parser.add_argument("--min-step-duration", type=float, default=0.15)
    parser.add_argument("--min-period", type=float, default=0.2)
    parser.add_argument("--default-period", type=float, default=0.8)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.2)
    parser.add_argument("--push-body", type=str, default="pelvis")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=False)
    parser.add_argument("--use-crown", action="store_true", default=False)
    parser.add_argument("--no-crown", dest="use_crown", action="store_false")
    parser.add_argument("--prescan-samples", type=int, default=50)
    parser.add_argument("--local-iterations", type=int, default=30)
    parser.add_argument("--local-population", type=int, default=16)
    parser.add_argument("--local-sigma", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--state-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--vel-perturbation-scale", type=float, default=0.02)
    parser.add_argument("--sensitive-scale-factor", type=float, default=0.5)
    parser.add_argument("--nonsensitive-scale-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _parse_vec(value: str, length: int) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != length:
        raise ValueError(f"Expected {length} values, got: {value}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _build_attacks(
    push: Optional[np.ndarray],
    friction: Optional[np.ndarray],
    push_start: Optional[float],
    push_duration: float,
    push_body: str,
) -> list:
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
    runner: WalkingTestRunner,
    friction: Optional[np.ndarray],
    duration_s: float,
    min_step_duration: float,
    min_period: float,
    default_period: float,
) -> tuple[float, float]:
    runner.env.attacks = _build_attacks(
        push=None,
        friction=friction,
        push_start=None,
        push_duration=0.0,
        push_body="pelvis",
    )
    runner.env.reset()

    model = runner.env.model
    data = runner.env.data
    ground_geom_ids = resolve_ground_geom_ids(model, runner.config.ground_geom_names)
    foot_geom_map = _build_foot_geom_map(model)

    if not ground_geom_ids or not foot_geom_map:
        return default_period, 0.0

    contacts = {"left": False, "right": False}
    contact_times = {"left": [], "right": []}

    steps = int(duration_s / runner.config.simulation_dt)
    for _ in range(steps):
        runner.env.step()
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
                if runner.env.sim_time - last_time >= min_step_duration:
                    contact_times[side].append(runner.env.sim_time)
            contacts[side] = current[side]

    diffs = []
    for side in ("left", "right"):
        times = contact_times[side]
        if len(times) >= 2:
            diffs.extend([b - a for a, b in zip(times[:-1], times[1:])])

    raw_period = float(np.median(diffs)) if diffs else 0.0
    if raw_period >= min_period:
        period = raw_period
    else:
        period = default_period

    if contact_times["left"]:
        cycle_start = float(contact_times["left"][0])
    elif contact_times["right"]:
        cycle_start = float(contact_times["right"][0])
    else:
        cycle_start = 0.0

    return period, cycle_start


def _phase_to_time(
    phase: float,
    period: float,
    cycle_start: float,
    settle_time: float,
) -> float:
    cycle = cycle_start
    if settle_time > cycle and period > 0:
        cycle += math.ceil((settle_time - cycle) / period) * period
    return cycle + phase * period


def _evaluate_phases(
    runner: WalkingTestRunner,
    push: Optional[np.ndarray],
    friction: Optional[np.ndarray],
    phases: list[float],
    period: float,
    cycle_start: float,
    settle_time: float,
    push_duration: float,
    push_body: str,
) -> tuple[float, float]:
    best_phase = phases[0]
    best_robustness = float("inf")
    for phase in phases:
        push_start = _phase_to_time(phase, period, cycle_start, settle_time)
        attacks = _build_attacks(push, friction, push_start, push_duration, push_body)
        runner.env.attacks = attacks
        result = runner.run_episode()
        robustness = float(result.metrics.get("stl_robustness", 0.0))
        if robustness < best_robustness:
            best_robustness = robustness
            best_phase = phase
    return best_phase, best_robustness


def _weighted_distance(vec: np.ndarray, pool: list[SeedEntry], weights: np.ndarray) -> float:
    if not pool:
        return 0.0
    distances = []
    for seed in pool:
        seed_vec = np.concatenate([seed.push, seed.friction, np.array([seed.phase])])
        diff = (vec - seed_vec) * weights
        distances.append(float(np.linalg.norm(diff)))
    return float(min(distances)) if distances else 0.0


def _normalize(value: float, vmin: float, vmax: float) -> float:
    return (value - vmin) / max(vmax - vmin, 1e-6)


def _compute_keep_score(
    robustness: float,
    distance: float,
    rob_min: float,
    rob_max: float,
    dist_min: float,
    dist_max: float,
) -> float:
    rob_norm = _normalize(robustness, rob_min, rob_max)
    dist_norm = _normalize(distance, dist_min, dist_max)
    return (1.0 - rob_norm) + 0.5 * dist_norm


def _select_seed(pool: list[SeedEntry]) -> SeedEntry:
    if not pool:
        raise ValueError("Seed pool is empty.")
    if random.random() < 0.5:
        scores = np.array([s.keep_score for s in pool], dtype=np.float32)
        min_score = float(scores.min())
        weights = scores - min_score + 1e-6
        total = float(weights.sum())
        if total <= 0:
            return random.choice(pool)
        probs = (weights / total).tolist()
        return random.choices(pool, weights=probs, k=1)[0]
    return random.choice(pool)


def _mutate_l1(
    base: SeedEntry,
    pool: list[SeedEntry],
    push_min: np.ndarray,
    push_max: np.ndarray,
    fric_min: np.ndarray,
    fric_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.1:
        mode = "keep"
    else:
        mode = random.choice(["gaussian", "jump", "crossover"])
    if mode == "gaussian":
        sigma_push = (push_max - push_min) * 0.1
        sigma_fric = (fric_max - fric_min) * 0.1
        push = base.push + np.random.normal(0.0, sigma_push)
        friction = base.friction + np.random.normal(0.0, sigma_fric)
    elif mode == "jump":
        push = base.push.copy()
        friction = base.friction.copy()
        for idx in random.sample(range(3), k=random.randint(1, 2)):
            push[idx] = random.uniform(push_min[idx], push_max[idx])
        for idx in random.sample(range(3), k=random.randint(1, 2)):
            friction[idx] = random.uniform(fric_min[idx], fric_max[idx])
    elif mode == "crossover":
        push = base.push.copy()
        friction = base.friction.copy()
        if pool:
            mate = random.choice([s for s in pool if s is not base] or [base])
            mask_push = np.random.rand(3) < 0.5
            mask_fric = np.random.rand(3) < 0.5
            push[mask_push] = mate.push[mask_push]
            friction[mask_fric] = mate.friction[mask_fric]
    else:
        push = base.push.copy()
        friction = base.friction.copy()
    push = np.clip(push, push_min, push_max)
    friction = np.clip(friction, fric_min, fric_max)
    return push, friction


def _mutate_phase(
    base_phase: float,
    mode: str,
    samples: int,
    span: float,
    mate_phase: Optional[float] = None,
) -> list[float]:
    if mode == "util":
        center = base_phase
        phases = [
            float(np.clip(center + random.uniform(-span, span), 0.0, 1.0))
            for _ in range(samples)
        ]
        return phases
    if mode == "grid":
        step = span
        count = max(2, int(1.0 / max(step, 1e-6)))
        return [i / float(count) for i in range(count + 1)]
    if mode == "crossover" and mate_phase is not None:
        blend = 0.5 * (base_phase + mate_phase) + random.uniform(-0.05, 0.05)
        return [
            float(np.clip(base_phase, 0.0, 1.0)),
            float(np.clip(mate_phase, 0.0, 1.0)),
            float(np.clip(blend, 0.0, 1.0)),
        ]
    return [random.random() for _ in range(samples)]


def _min_distance(
    seed: SeedEntry,
    pool: list[SeedEntry],
    weights: np.ndarray,
) -> float:
    distances = []
    for other in pool:
        if other is seed:
            continue
        vec = np.concatenate([seed.push, seed.friction, np.array([seed.phase])])
        other_vec = np.concatenate([other.push, other.friction, np.array([other.phase])])
        diff = (vec - other_vec) * weights
        distances.append(float(np.linalg.norm(diff)))
    return float(min(distances)) if distances else 0.0


def _recompute_keep_scores(pool: list[SeedEntry], weights: np.ndarray) -> None:
    if not pool:
        return
    robustness_vals = [s.robustness for s in pool]
    distance_vals = [_min_distance(s, pool, weights) for s in pool]

    rob_min = float(min(robustness_vals))
    rob_max = float(max(robustness_vals))
    dist_min = float(min(distance_vals))
    dist_max = float(max(distance_vals))

    for seed, dist in zip(pool, distance_vals):
        seed.keep_score = _compute_keep_score(
            seed.robustness, dist, rob_min, rob_max, dist_min, dist_max
        )


def _save_failure(failure_dir: Path, iteration: int, payload: dict) -> None:
    case_dir = failure_dir / f"failure_{iteration:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "case.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_seed_pool(seed_dir: Path) -> list[SeedEntry]:
    seed_pool: list[SeedEntry] = []
    if not seed_dir.exists():
        return seed_pool
    for path in sorted(seed_dir.glob("seed_*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        seed_pool.append(
            SeedEntry(
                push=np.array(entry["push"], dtype=np.float32),
                friction=np.array(entry["friction"], dtype=np.float32),
                phase=float(entry["phase"]),
                push_start=float(entry["push_start"]),
                robustness=float(entry["robustness"]),
                keep_score=float(entry.get("keep_score", 0.0)),
                joint_pos_offset=np.array(entry["joint_pos_offset"], dtype=np.float32)
                if entry.get("joint_pos_offset") is not None
                else None,
                joint_vel_offset=np.array(entry["joint_vel_offset"], dtype=np.float32)
                if entry.get("joint_vel_offset") is not None
                else None,
            )
        )
    return seed_pool


def _save_seed_pool(seed_dir: Path, seed_pool: list[SeedEntry]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    for path in seed_dir.glob("seed_*.json"):
        path.unlink()
    for idx, seed in enumerate(seed_pool, start=1):
        payload = {
            "push": seed.push.tolist(),
            "friction": seed.friction.tolist(),
            "phase": seed.phase,
            "push_start": seed.push_start,
            "robustness": seed.robustness,
            "keep_score": seed.keep_score,
            "joint_pos_offset": seed.joint_pos_offset.tolist()
            if seed.joint_pos_offset is not None
            else None,
            "joint_vel_offset": seed.joint_vel_offset.tolist()
            if seed.joint_vel_offset is not None
            else None,
        }
        path = seed_dir / f"seed_{idx:05d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    push_min = _parse_vec(args.push_min, 3)
    push_max = _parse_vec(args.push_max, 3)
    fric_min = _parse_vec(args.friction_min, 3)
    fric_max = _parse_vec(args.friction_max, 3)
    fric_default = _parse_vec(args.friction_default, 3)

    weights = np.concatenate(
        [
            1.0 / np.maximum(push_max - push_min, 1e-6),
            1.0 / np.maximum(fric_max - fric_min, 1e-6),
            np.array([1.0], dtype=np.float32),
        ]
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / args.seed_dir
    failure_dir = output_dir / args.failure_dir
    seed_pool = _load_seed_pool(seed_dir)
    if seed_pool:
        _recompute_keep_scores(seed_pool, weights)

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

    runner_l2 = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=True,
        render=args.render,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    analyzer = CROWNSensitivityAnalyzer(policy, epsilon=args.epsilon, use_crown=args.use_crown)
    search_config = Layer3SearchConfig(
        prescan_samples=args.prescan_samples,
        epsilon=args.epsilon,
        local_iterations=args.local_iterations,
        local_population=args.local_population,
        local_sigma=args.local_sigma,
        state_perturbation_scale=args.state_perturbation_scale,
        vel_perturbation_scale=args.vel_perturbation_scale,
        sensitive_scale_factor=args.sensitive_scale_factor,
        nonsensitive_scale_factor=args.nonsensitive_scale_factor,
    )
    searcher = Layer3StateSearcher(runner_l2, analyzer, search_config)

    for iteration in range(args.iterations):
        exploit_ratio = min(args.exploit_cap, len(seed_pool) * 0.02)
        use_exploit = random.random() < exploit_ratio

        if use_exploit and seed_pool:
            seed = _select_seed(seed_pool)
            push, friction = _mutate_l1(
                seed,
                seed_pool,
                push_min,
                push_max,
                fric_min,
                fric_max,
            )
            phase_mode = random.choice(["util", "random", "crossover"])
            mate_phase = None
            if phase_mode == "crossover" and len(seed_pool) > 1:
                mate = random.choice([s for s in seed_pool if s is not seed] or [seed])
                mate_phase = mate.phase
            phases = _mutate_phase(
                seed.phase,
                phase_mode,
                args.phase_mutation_samples,
                args.phase_mutation_range,
                mate_phase=mate_phase,
            )
        else:
            if random.random() < args.random_zero_prob:
                push = np.zeros(3, dtype=np.float32)
                friction = fric_default.copy()
            else:
                push = np.random.uniform(push_min, push_max).astype(np.float32)
                friction = np.random.uniform(fric_min, fric_max).astype(np.float32)
            phase_mode = "grid"
            phases = [i * args.phase_step for i in range(int(1.0 / args.phase_step) + 1)]

        period, cycle_start = _probe_phase(
            runner_l2,
            friction=friction,
            duration_s=args.probe_duration,
            min_step_duration=args.min_step_duration,
            min_period=args.min_period,
            default_period=args.default_period,
        )

        best_phase, best_phase_robustness = _evaluate_phases(
            runner_l2,
            push=push,
            friction=friction,
            phases=phases,
            period=period,
            cycle_start=cycle_start,
            settle_time=args.settle_time,
            push_duration=args.push_duration,
            push_body=args.push_body,
        )
        push_start = _phase_to_time(best_phase, period, cycle_start, args.settle_time)

        refined_robustness, best_perturbation, sensitivity_history = searcher.search(
            base_push=push,
            base_friction=friction,
            base_push_start=push_start,
            push_duration=args.push_duration,
            push_body=args.push_body,
        )

        vec = np.concatenate([push, friction, np.array([best_phase], dtype=np.float32)])
        distance = _weighted_distance(vec, seed_pool, weights)
        rob_vals = [s.robustness for s in seed_pool] + [refined_robustness]
        dist_vals = [
            _min_distance(s, seed_pool, weights) for s in seed_pool
        ] + [distance]
        keep_score = _compute_keep_score(
            refined_robustness,
            distance,
            float(min(rob_vals)),
            float(max(rob_vals)),
            float(min(dist_vals)),
            float(max(dist_vals)),
        )

        if sensitivity_history:
            best_sens = max(sensitivity_history, key=lambda s: s.sensitivity_score)
            sensitivity_summary = {
                "mean_score": float(
                    np.mean([s.sensitivity_score for s in sensitivity_history])
                ),
                "max_score": float(best_sens.sensitivity_score),
                "lipschitz_estimate": float(best_sens.lipschitz_estimate),
                "sensitive_dims": best_sens.sensitive_dims.tolist(),
            }
        else:
            sensitivity_summary = {}

        new_seed = SeedEntry(
            push=push,
            friction=friction,
            phase=best_phase,
            push_start=push_start,
            robustness=refined_robustness,
            keep_score=keep_score,
            joint_pos_offset=best_perturbation[: config.num_actions],
            joint_vel_offset=best_perturbation[config.num_actions :],
        )

        if len(seed_pool) < args.seed_pool_size or keep_score >= min(s.keep_score for s in seed_pool):
            seed_pool.append(new_seed)
            _recompute_keep_scores(seed_pool, weights)
            seed_pool.sort(key=lambda s: s.keep_score, reverse=True)
            if len(seed_pool) > args.seed_pool_size:
                seed_pool.pop()

        if refined_robustness < 0:
            _save_failure(
                failure_dir,
                iteration,
                {
                    "iteration": iteration,
                    "mode": "exploit" if use_exploit else "random",
                    "exploit_ratio": exploit_ratio,
                    "random_zero_macro": bool(
                        not use_exploit and np.allclose(push, 0.0)
                    ),
                    "push": push.tolist(),
                    "friction": friction.tolist(),
                    "phase": best_phase,
                    "push_start": push_start,
                    "push_duration": args.push_duration,
                    "push_body": args.push_body,
                    "period_s": period,
                    "cycle_start_s": cycle_start,
                    "settle_time": args.settle_time,
                    "joint_pos_offset": new_seed.joint_pos_offset.tolist()
                    if new_seed.joint_pos_offset is not None
                    else None,
                    "joint_vel_offset": new_seed.joint_vel_offset.tolist()
                    if new_seed.joint_vel_offset is not None
                    else None,
                    "robustness": refined_robustness,
                    "keep_score": new_seed.keep_score,
                    "sensitivity": sensitivity_summary,
                    "search_config": {
                        "epsilon": args.epsilon,
                        "prescan_samples": args.prescan_samples,
                        "local_iterations": args.local_iterations,
                        "local_population": args.local_population,
                        "local_sigma": args.local_sigma,
                        "state_perturbation_scale": args.state_perturbation_scale,
                        "vel_perturbation_scale": args.vel_perturbation_scale,
                        "sensitive_scale_factor": args.sensitive_scale_factor,
                        "nonsensitive_scale_factor": args.nonsensitive_scale_factor,
                    },
                    "l1_bounds": {
                        "push_min": push_min.tolist(),
                        "push_max": push_max.tolist(),
                        "friction_min": fric_min.tolist(),
                        "friction_max": fric_max.tolist(),
                        "friction_default": fric_default.tolist(),
                    },
                    "l2_search": {
                        "phase_step": args.phase_step,
                        "phase_mode": phase_mode,
                        "phase_candidates": phases,
                    },
                    "config": str(args.config),
                    "policy_path": str(args.policy_path)
                    if args.policy_path is not None
                    else None,
                },
            )

        _save_seed_pool(seed_dir, seed_pool)

        print(
            f"[{iteration+1}/{args.iterations}] mode={'exploit' if use_exploit else 'random'} "
            f"robustness={refined_robustness:.4f} keep={keep_score:.4f} seeds={len(seed_pool)}"
        )


if __name__ == "__main__":
    main()
