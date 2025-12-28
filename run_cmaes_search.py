from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.search.cmaes import CMAES
from robostl.tasks.walking import WalkingTask


@dataclass
class SearchSpace:
    names: list[str]
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass
class ResultEntry:
    robustness: float
    params: np.ndarray
    metrics: dict


def _parse_vec3(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {value}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _build_search_space(args: argparse.Namespace) -> SearchSpace:
    names: list[str] = []
    mean: list[float] = []
    low: list[float] = []
    high: list[float] = []

    if not args.disable_push:
        push_min = _parse_vec3(args.push_force_min)
        push_max = _parse_vec3(args.push_force_max)
        names.extend(["push_fx", "push_fy", "push_fz"])
        mean.extend([0.0, 0.0, 0.0])
        low.extend(push_min.tolist())
        high.extend(push_max.tolist())

    if not args.disable_friction:
        fric_min = _parse_vec3(args.friction_min)
        fric_max = _parse_vec3(args.friction_max)
        fric_init = _parse_vec3(args.friction_init)
        names.extend(["fric_mu1", "fric_mu2", "fric_mu3"])
        mean.extend(fric_init.tolist())
        low.extend(fric_min.tolist())
        high.extend(fric_max.tolist())

    if not names:
        raise ValueError("No search variables enabled (push/friction both disabled).")

    return SearchSpace(
        names=names,
        mean=np.array(mean, dtype=np.float32),
        low=np.array(low, dtype=np.float32),
        high=np.array(high, dtype=np.float32),
    )


def _build_attacks(
    params: np.ndarray,
    push_start: float,
    push_duration: float,
    push_body: str,
    friction_geom: str,
    use_push: bool,
    use_friction: bool,
) -> list:
    attacks = []
    cursor = 0

    if use_push:
        force = params[cursor : cursor + 3]
        cursor += 3
        attacks.append(
            ForcePerturbation(
                body_name=push_body,
                force=force,
                start_time=push_start,
                duration=push_duration,
            )
        )

    if use_friction:
        friction = params[cursor : cursor + 3]
        cursor += 3
        attacks.append(
            FloorFrictionModifier(
                geom_name=friction_geom,
                friction=friction,
            )
        )

    return attacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-ES search for STL robustness.")
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
    parser.add_argument("--iterations", type=int, default=15, help="CMA-ES generations.")
    parser.add_argument("--population", type=int, default=None, help="CMA-ES population size.")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial CMA-ES sigma.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo rendering.")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate candidates in parallel using multiple processes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker count for parallel evaluation.",
    )
    parser.set_defaults(render=True)

    parser.add_argument("--disable-push", action="store_true", help="Disable push search.")
    parser.add_argument("--disable-friction", action="store_true", help="Disable friction search.")
    parser.add_argument("--push-force-min", type=str, default="-80,-80,-80")
    parser.add_argument("--push-force-max", type=str, default="80,80,80")
    parser.add_argument("--push-start", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.2)
    parser.add_argument("--push-body", type=str, default="pelvis")

    parser.add_argument("--friction-min", type=str, default="0.2,0.001,0.00001")
    parser.add_argument("--friction-max", type=str, default="2.0,0.02,0.01")
    parser.add_argument("--friction-init", type=str, default="1.0,0.005,0.0001")
    parser.add_argument("--friction-geom", type=str, default="terrain")

    parser.add_argument(
        "--target-robustness",
        type=float,
        default=None,
        help="Stop early when best robustness <= target.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer1_result.json"),
        help="Output JSON file for best parameters.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to keep for Layer 2.",
    )
    parser.add_argument(
        "--top-k-output",
        type=Path,
        default=Path("layer1_top_k.json"),
        help="Output JSON file for top-K candidates.",
    )
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.5,
        help="Minimum weighted L2 distance between saved candidates.",
    )
    parser.add_argument(
        "--diversity-weights",
        type=str,
        default=None,
        help="Comma-separated weights for diversity distance (per-dimension).",
    )
    return parser.parse_args()


def objective_function(
    params: np.ndarray,
    runner: WalkingTestRunner,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    attacks = _build_attacks(
        params=params,
        push_start=args.push_start,
        push_duration=args.push_duration,
        push_body=args.push_body,
        friction_geom=args.friction_geom,
        use_push=not args.disable_push,
        use_friction=not args.disable_friction,
    )
    runner.env.attacks = attacks
    result = runner.run_episode()
    loss = float(result.metrics.get("stl_robustness", 0.0))
    return loss, result.metrics


def select_top_k_diverse(
    entries: list[ResultEntry],
    top_k: int,
    diversity_threshold: float,
    diversity_weights: np.ndarray,
) -> list[ResultEntry]:
    if top_k <= 0 or not entries:
        return []

    ordered = sorted(entries, key=lambda e: e.robustness)
    selected: list[ResultEntry] = []

    for entry in ordered:
        if diversity_threshold > 0.0 and selected:
            for chosen in selected:
                scaled = (entry.params - chosen.params) * diversity_weights
                distance = float(np.linalg.norm(scaled))
                if distance < diversity_threshold:
                    break
            else:
                selected.append(entry)
        else:
            selected.append(entry)

        if len(selected) >= top_k:
            return selected

    return selected


def _worker_objective(
    params: np.ndarray,
    args_dict: dict[str, Any],
    config_path: str,
    policy_path: Optional[str],
) -> tuple[float, dict]:
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
        render=False,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    args_ns = argparse.Namespace(**args_dict)
    return objective_function(params, runner, args_ns)


def _parse_weights(value: Optional[str], size: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != size:
        raise ValueError(f"Expected {size} weights, got {len(parts)}: {value}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _build_diversity_weights(args: argparse.Namespace, space: SearchSpace) -> np.ndarray:
    provided = _parse_weights(args.diversity_weights, space.mean.size)
    if provided is not None:
        return provided
    span = np.maximum(space.high - space.low, 1e-6)
    return 1.0 / span


def main() -> None:
    args = parse_args()
    space = _build_search_space(args)
    diversity_weights = _build_diversity_weights(args, space)

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
        stop_on_fall=True,
        render=args.render,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    optimizer = CMAES(
        mean=space.mean,
        sigma=args.sigma,
        bounds=(space.low, space.high),
        population_size=args.population,
        seed=args.seed,
    )

    best_loss = float("inf")
    best_params: Optional[np.ndarray] = None
    best_metrics: Optional[dict] = None
    hall_of_fame: list[ResultEntry] = []
    all_results: list[ResultEntry] = []

    if args.parallel and args.render:
        print("[Info] Parallel mode disables rendering to avoid multi-process viewer issues.")
        args.render = False
        runner.render = False

    if args.parallel:
        from concurrent.futures import ProcessPoolExecutor

        args_dict = vars(args).copy()
        args_dict["render"] = False
        config_path = str(args.config)
        policy_path = str(args.policy_path) if args.policy_path is not None else None

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for gen in range(args.iterations):
                candidates = optimizer.ask()
                futures = [
                    executor.submit(
                        _worker_objective,
                        params,
                        args_dict,
                        config_path,
                        policy_path,
                    )
                    for params in candidates
                ]
                results = [future.result() for future in futures]

                losses = np.array([item[0] for item in results], dtype=np.float32)
                for params, (loss, metrics) in zip(candidates, results):
                    if loss < best_loss:
                        best_loss = loss
                        best_params = params.copy()
                        best_metrics = metrics
                    all_results.append(ResultEntry(loss, params.copy(), metrics))

                state = optimizer.tell(losses)
                mean_loss = float(np.mean(losses))
                print(
                    f"Gen {gen + 1:02d} | "
                    f"best={best_loss:.4f} | "
                    f"mean={mean_loss:.4f} | "
                    f"sigma={state.sigma:.4f}"
                )

                if args.target_robustness is not None and best_loss <= args.target_robustness:
                    break
    else:
        for gen in range(args.iterations):
            candidates = optimizer.ask()
            losses = np.zeros(candidates.shape[0], dtype=np.float32)

            for idx, params in enumerate(candidates):
                loss, metrics = objective_function(params, runner, args)
                losses[idx] = loss

                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                    best_metrics = metrics
                all_results.append(ResultEntry(loss, params.copy(), metrics))

            state = optimizer.tell(losses)
            mean_loss = float(np.mean(losses))
            print(
                f"Gen {gen + 1:02d} | "
                f"best={best_loss:.4f} | "
                f"mean={mean_loss:.4f} | "
                f"sigma={state.sigma:.4f}"
            )

            if args.target_robustness is not None and best_loss <= args.target_robustness:
                break

    if best_params is None:
        print("No valid result.")
        return

    print("Best params:")
    for name, value in zip(space.names, best_params):
        print(f"  {name} = {value:.6f}")
    if best_metrics:
        print(f"Best STL robustness: {best_metrics.get('stl_robustness')}")

    output = args.output
    result_data = {
        "best_params": best_params.tolist(),
        "param_names": space.names,
        "best_robustness": float(best_loss),
        "metrics": best_metrics,
        "diversity_weights": diversity_weights.tolist(),
        "config": str(args.config),
    }
    output.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Search result saved to {output}")

    hall_of_fame = select_top_k_diverse(
        all_results,
        args.top_k,
        args.diversity_threshold,
        diversity_weights,
    )
    if hall_of_fame:
        top_k_data = [
            {
                "rank": idx + 1,
                "params": entry.params.tolist(),
                "param_names": space.names,
                "robustness": float(entry.robustness),
                "metrics": entry.metrics,
            }
            for idx, entry in enumerate(hall_of_fame)
        ]
        top_k_output = args.top_k_output
        top_k_output.write_text(json.dumps(top_k_data, indent=2), encoding="utf-8")
        print(f"Top-K results saved to {top_k_output}")


if __name__ == "__main__":
    main()
