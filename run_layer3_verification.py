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

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import WalkingTestRunner
from robostl.tasks.walking import WalkingTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer 3 verification: CROWN-like bounds + exact line scan."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("layer2_results.json"),
        help="Layer 2 results JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer3_results.json"),
        help="Output JSON file.",
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
    parser.add_argument(
        "--case-rank",
        type=int,
        default=None,
        help="Which rank from layer2_results.json to analyze (default: all).",
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
        "--crown-method",
        type=str,
        default="auto_lirpa",
        choices=["auto_lirpa", "linearized"],
        help="CROWN backend: auto_lirpa (verified) or linearized (diagnosis).",
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


def _extract_params(entry: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
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


def _select_case(data: list[dict], rank: int) -> dict:
    for entry in data:
        if int(entry.get("rank", -1)) == rank:
            return entry
    raise ValueError(f"Rank {rank} not found in layer2 results.")


def _run_to_time(
    runner: WalkingTestRunner,
    target_time: float,
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


def _crown_bounds(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    epsilon: float,
) -> dict:
    snapshot = _snapshot_memory(module)
    has_memory = snapshot is not None

    if not has_memory:
        try:
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            x.requires_grad_(True)
            y = module(x)
            y_vec = y.squeeze(0)

            grads = []
            for i in range(int(y_vec.numel())):
                grad = torch.autograd.grad(
                    y_vec[i],
                    x,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is None:
                    grad = torch.zeros_like(x)
                grads.append(grad.squeeze(0))
            jac = torch.stack(grads, dim=0)
            abs_jac = jac.abs()
            eps_vec = torch.full_like(abs_jac[0], float(epsilon))
            delta = abs_jac @ eps_vec

            lower = (y_vec - delta).detach().cpu().numpy()
            upper = (y_vec + delta).detach().cpu().numpy()
            max_dev = float(delta.max().detach().cpu().item()) if delta.numel() else 0.0

            return {
                "method": "linearized_linf",
                "output": y_vec.detach().cpu().numpy().tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "max_deviation": max_dev,
            }
        except RuntimeError:
            has_memory = True

    step = max(1e-4, float(epsilon) * 0.1)
    nominal = _safe_forward_np(module, obs, snapshot)
    action_dim = int(nominal.shape[0])
    obs_dim = int(obs.shape[0])
    jac = np.zeros((action_dim, obs_dim), dtype=np.float32)

    for i in range(obs_dim):
        obs_plus = obs.copy()
        obs_minus = obs.copy()
        obs_plus[i] += step
        obs_minus[i] -= step
        y_plus = _safe_forward_np(module, obs_plus, snapshot)
        y_minus = _safe_forward_np(module, obs_minus, snapshot)
        jac[:, i] = (y_plus - y_minus) / (2.0 * step)

    delta = np.abs(jac).dot(np.full((obs_dim,), float(epsilon), dtype=np.float32))
    lower = nominal - delta
    upper = nominal + delta
    max_dev = float(np.max(delta)) if delta.size else 0.0

    return {
        "method": "linearized_linf",
        "output": nominal.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "max_deviation": max_dev,
    }


def _crown_bounds_auto_lirpa(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    epsilon: float,
) -> dict:
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ModuleNotFoundError as exc:
        raise RuntimeError("auto_LiRPA is required for CROWN verification.") from exc

    model = module
    if isinstance(module, torch.jit.ScriptModule):
        module.eval()

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        nominal = model(x).squeeze(0)

    try:
        bounded_model = BoundedModule(model, x)
    except RuntimeError as exc:
        msg = str(exc)
        if "PolicyExporterLSTM" in msg or "not part of the active trace" in msg:
            raise RuntimeError(
                "auto_LiRPA could not trace the TorchScript LSTM policy. "
                "Please provide a non-recurrent policy (e.g., policy_1.pt) or "
                "use --crown-method linearized for diagnosis."
            ) from exc
        raise
    ptb = PerturbationLpNorm(norm=np.inf, eps=float(epsilon))
    bounded_x = BoundedTensor(x, ptb)

    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")
    lb = lb.squeeze(0).detach().cpu().numpy()
    ub = ub.squeeze(0).detach().cpu().numpy()
    nominal_np = nominal.detach().cpu().numpy()
    max_dev = float(
        np.max(np.maximum(ub - nominal_np, nominal_np - lb)) if ub.size else 0.0
    )

    return {
        "method": "auto_lirpa_crown",
        "output": nominal_np.tolist(),
        "lower": lb.tolist(),
        "upper": ub.tolist(),
        "max_deviation": max_dev,
    }


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


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.case_rank is not None:
        entries = [_select_case(data, args.case_rank)]
    else:
        entries = list(data)

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
        render=False,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )

    results = []
    for entry in entries:
        push, friction, names = _extract_params(entry)
        push_start = None
        push_duration = 0.2
        push_body = "pelvis"
        best = entry.get("best", {})
        if isinstance(best, dict):
            push_start = best.get("push_start")

        attacks = _build_attacks(push, friction, push_start, push_duration, push_body)
        runner.env.attacks = attacks

        t_star = float(push_start) if push_start is not None else 0.0
        t_star = max(0.0, t_star + args.observe_offset)
        obs = _run_to_time(runner, t_star)

        if args.crown_method == "auto_lirpa":
            try:
                crown = _crown_bounds_auto_lirpa(policy.module, obs, args.epsilon)
            except RuntimeError as exc:
                print(f"[Layer3] auto_LiRPA failed, fallback to linearized: {exc}")
                crown = _crown_bounds(policy.module, obs, args.epsilon)
        else:
            crown = _crown_bounds(policy.module, obs, args.epsilon)
        direction = _attack_direction(policy.module, obs, args.action_index)
        line_scan = _exact_line_scan(
            policy.module,
            obs,
            np.array(direction["direction"], dtype=np.float32),
            args.epsilon,
            args.line_samples,
        )

        results.append(
            {
                "case_rank": entry.get("rank"),
                "param_names": names,
                "params": entry.get("params"),
                "t_star": t_star,
                "epsilon": args.epsilon,
                "observation": obs.tolist(),
                "crown": crown,
                "attack_direction": direction,
                "exactline": line_scan,
            }
        )

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Layer 3 results saved to {args.output}")


if __name__ == "__main__":
    main()
