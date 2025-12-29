from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer 3 verification: exact line scan over failure cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("robostl/fuzz_outputs/failures"),
        help="Failure case JSON file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file when input is a single case.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="analysis.json",
        help="Output file name inside each failure folder.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="analysis.png",
        help="Plot image file name inside each failure folder.",
    )
    parser.add_argument(
        "--plot-dim",
        type=int,
        default=None,
        help="Action dimension to plot (default: auto).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to deploy config yaml.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy path (TorchScript).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="L-infinity perturbation radius for observation.",
    )
    parser.add_argument(
        "--line-samples",
        type=int,
        default=81,
        help="Number of samples for exact line scan.",
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=None,
        help="If set, optimize and report this action dimension.",
    )
    parser.add_argument(
        "--observe-offset",
        type=float,
        default=0.0,
        help="Offset added to t* when capturing the observation.",
    )
    return parser.parse_args()


def _load_failure_cases(input_path: Path) -> list[tuple[str, dict, Path]]:
    if input_path.is_dir():
        cases = []
        for case_dir in sorted(input_path.glob("failure_*")):
            case_file = case_dir / "case.json"
            if case_file.exists():
                cases.append((case_dir.name, json.loads(case_file.read_text(encoding="utf-8")), case_dir))
        return cases
    return [(input_path.stem, json.loads(input_path.read_text(encoding="utf-8")), input_path.parent)]


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


def _run_to_time_with_perturbation(
    runner: WalkingTestRunner,
    target_time: float,
    joint_pos_offset: Optional[np.ndarray],
    joint_vel_offset: Optional[np.ndarray],
) -> np.ndarray:
    env = runner.env
    state = env.reset()
    dt = runner.config.simulation_dt
    max_steps = int(runner.config.simulation_duration / dt)
    target_steps = min(max_steps, int(np.ceil(target_time / dt)))
    for _ in range(target_steps):
        state = env.step()
        if state.time >= target_time:
            break
    if joint_pos_offset is not None or joint_vel_offset is not None:
        env.apply_joint_perturbation(joint_pos_offset, joint_vel_offset)
    obs = env.obs_builder.build(env.data, env.action, env.cmd, env.sim_time, env.counter)
    return obs


def _snapshot_memory(
    module: torch.jit.ScriptModule,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(module, "hidden_state") and hasattr(module, "cell_state"):
        hidden = module.hidden_state.detach().clone()
        cell = module.cell_state.detach().clone()
        return hidden, cell
    return None


def _restore_memory(
    module: torch.jit.ScriptModule,
    snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if snapshot is None:
        return
    hidden, cell = snapshot
    if hasattr(module, "hidden_state"):
        module.hidden_state.copy_(hidden)
    if hasattr(module, "cell_state"):
        module.cell_state.copy_(cell)


def _safe_forward_np(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> np.ndarray:
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        _restore_memory(module, snapshot)
        y = module(x).squeeze(0).cpu().numpy()
        _restore_memory(module, snapshot)
    return y


def _extract_attack(entry: dict) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    push = np.array(entry.get("push", [0.0, 0.0, 0.0]), dtype=np.float32)
    friction = np.array(entry.get("friction", [1.0, 0.005, 0.0001]), dtype=np.float32)
    push_start = float(entry.get("push_start", 0.0))
    push_duration = float(entry.get("push_duration", 0.2))
    push_body = str(entry.get("push_body", "pelvis"))
    return push, friction, push_start, push_duration, push_body


def _attack_direction(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    action_index: Optional[int],
) -> dict:
    snapshot = _snapshot_memory(module)
    has_memory = snapshot is not None

    if not has_memory:
        try:
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            x.requires_grad_(True)
            y = module(x).squeeze(0)
            if action_index is not None:
                if action_index < 0 or action_index >= y.numel():
                    raise ValueError("action_index out of range.")
                objective = y[action_index]
                obj_name = f"action[{action_index}]"
            else:
                objective = torch.linalg.norm(y, ord=2)
                obj_name = "action_l2"

            grad = torch.autograd.grad(objective, x, retain_graph=False)[0]
            direction = grad.squeeze(0).detach().cpu().numpy()
            if np.allclose(direction, 0.0):
                direction = np.ones_like(direction)
            direction = np.sign(direction)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            return {"objective": obj_name, "direction": direction.tolist()}
        except RuntimeError:
            has_memory = True

    step = 1e-4
    if action_index is not None:
        obj_name = f"action[{action_index}]"
    else:
        obj_name = "action_l2"

    grad = np.zeros_like(obs, dtype=np.float32)
    for i in range(obs.shape[0]):
        obs_plus = obs.copy()
        obs_minus = obs.copy()
        obs_plus[i] += step
        obs_minus[i] -= step
        y_plus = _safe_forward_np(module, obs_plus, snapshot)
        y_minus = _safe_forward_np(module, obs_minus, snapshot)
        if action_index is not None:
            f_plus = y_plus[action_index]
            f_minus = y_minus[action_index]
        else:
            f_plus = float(np.linalg.norm(y_plus))
            f_minus = float(np.linalg.norm(y_minus))
        grad[i] = (f_plus - f_minus) / (2.0 * step)

    if np.allclose(grad, 0.0):
        grad = np.ones_like(grad)
    direction = np.sign(grad)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return {"objective": obj_name, "direction": direction.tolist()}


def _exact_line_scan(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    samples: int,
) -> dict:
    snapshot = _snapshot_memory(module)
    alphas = np.linspace(-epsilon, epsilon, samples, dtype=np.float32)
    actions = []
    norms = []
    for alpha in alphas:
        x = obs + alpha * direction
        y = _safe_forward_np(module, x, snapshot)
        actions.append(y.tolist())
        norms.append(float(np.linalg.norm(y)))

    slope_norms = []
    for i in range(len(alphas) - 1):
        da = float(alphas[i + 1] - alphas[i])
        dy = np.array(actions[i + 1]) - np.array(actions[i])
        slope_norms.append(float(np.linalg.norm(dy) / max(da, 1e-6)))

    return {
        "alphas": alphas.tolist(),
        "actions": actions,
        "action_norms": norms,
        "slope_norms": slope_norms,
    }


def _rank_nonlinearity_dimensions(result: dict) -> list[dict]:
    alphas = np.array(result["exactline"]["alphas"], dtype=np.float32)
    actions = np.array(result["exactline"]["actions"], dtype=np.float32)
    if actions.size == 0 or actions.ndim != 2:
        return []

    slopes = np.diff(actions, axis=0) / (np.diff(alphas)[:, None] + 1e-9)
    if slopes.shape[0] < 2:
        return []

    slope_changes = np.abs(np.diff(slopes, axis=0))
    nonlinearity = slope_changes.max(axis=0)
    ranking = np.argsort(nonlinearity)[::-1]

    return [
        {
            "dim": int(idx),
            "nonlinearity": float(nonlinearity[idx]),
        }
        for idx in ranking
    ]


def _plot_analysis(
    result: dict,
    output_path: Path,
    dim: Optional[int],
) -> None:
    alphas = np.array(result["exactline"]["alphas"], dtype=np.float32)
    actions = np.array(result["exactline"]["actions"], dtype=np.float32)

    if dim is None:
        variation = actions.max(axis=0) - actions.min(axis=0)
        target_dim = int(np.argmax(variation))
    else:
        target_dim = int(dim)

    x_vals = alphas
    y_vals = actions[:, target_dim]
    nominal = y_vals[np.argmin(np.abs(alphas))]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, color="blue", linewidth=2.5, label="Exact Response")
    plt.scatter([0.0], [float(nominal)], color="black", s=80, zorder=5, label="Nominal (t*)")

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

    case_label = result.get("case_id", "case")
    plt.title(
        f"Layer 3 Diagnosis: {case_label} @ t={result.get('t_star'):.3f}s\n"
        f"Action Dimension {target_dim}"
    )
    plt.xlabel(f"Input Perturbation (epsilon = {result.get('epsilon')})")
    plt.ylabel("Action Output")
    plt.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")

    attack = result.get("attack", {})
    perturb = result.get("perturbation", {})
    sensitivity = result.get("sensitivity") or "N/A"
    ranking = result.get("nonlinearity_ranking") or []
    top_dims = [r.get("dim") for r in ranking[:3] if "dim" in r]
    info_lines = [
        f"robustness: {result.get('robustness')}",
        f"phase: {attack.get('phase')}",
        f"push: {attack.get('push')}",
        f"friction: {attack.get('friction')}",
        f"push_start: {attack.get('push_start')}",
        f"keep_score: {result.get('keep_score')}",
        f"sensitivity: {sensitivity}",
        f"top_dims: {top_dims}",
        f"pos_offset: {perturb.get('joint_pos_offset')}",
        f"vel_offset: {perturb.get('joint_vel_offset')}",
    ]
    info_text = "\n".join(str(line) for line in info_lines)
    plt.text(
        0.98,
        0.02,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    cases = _load_failure_cases(args.input)
    if not cases:
        raise SystemExit(f"No cases found under: {args.input}")

    results = []
    for case_id, entry, case_dir in cases:
        config_path = args.config or Path(entry.get("config", DeployConfig.default_config_path()))
        config = DeployConfig.from_yaml(config_path)
        if args.policy_path is not None:
            policy_path = args.policy_path
        else:
            policy_path = entry.get("policy_path")
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

        push, friction, push_start, push_duration, push_body = _extract_attack(entry)
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

        attacks = _build_attacks(push, friction, push_start, push_duration, push_body)
        runner.env.attacks = attacks

        t_star = max(0.0, float(push_start) + args.observe_offset)
        obs = _run_to_time_with_perturbation(
            runner, t_star, joint_pos_offset, joint_vel_offset
        )

        direction = _attack_direction(policy.module, obs, args.action_index)
        line_scan = _exact_line_scan(
            policy.module,
            obs,
            np.array(direction["direction"], dtype=np.float32),
            args.epsilon,
            args.line_samples,
        )
        nonlinearity_ranking = _rank_nonlinearity_dimensions(
            {
                "exactline": line_scan,
            }
        )

        result = {
            "case_id": case_id,
            "iteration": entry.get("iteration"),
            "robustness": entry.get("robustness"),
            "t_star": t_star,
            "epsilon": args.epsilon,
            "attack": {
                "push": push.tolist(),
                "friction": friction.tolist(),
                "push_start": push_start,
                "push_duration": push_duration,
                "push_body": push_body,
                "phase": entry.get("phase"),
            },
            "perturbation": {
                "joint_pos_offset": joint_pos_offset.tolist()
                if joint_pos_offset is not None
                else None,
                "joint_vel_offset": joint_vel_offset.tolist()
                if joint_vel_offset is not None
                else None,
            },
            "sensitivity": entry.get("sensitivity", {}),
            "observation": obs.tolist(),
            "attack_direction": direction,
            "exactline": line_scan,
            "nonlinearity_ranking": nonlinearity_ranking,
        }
        results.append(result)

        if args.input.is_dir():
            out_path = case_dir / args.output_name
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            plot_path = case_dir / args.plot_name
            _plot_analysis(result, plot_path, args.plot_dim)

    if args.input.is_file():
        output = args.output or Path("layer3_results.json")
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Layer 3 results saved to {output}")


if __name__ == "__main__":
    main()
