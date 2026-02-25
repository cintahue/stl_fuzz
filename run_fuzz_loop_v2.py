"""RoboSTL-Fuzz v2 — Surrogate + LLM enhanced fuzzing loop.

Architecture changes over v1 (run_fuzz_loop.py):
  1. GlobalSurrogate (GPR + MLP) replaces blind random search:
       - GPR + EI for L1 candidate selection (BayesianOptimizer)
       - GPR pre-filter: skip provably safe L1 configs
       - MLP active-learning inside L2-L3 search (replaces most CMA-ES calls)
  2. LLM integration (optional):
       - Offline seed generation from natural-language descriptions
       - Batch failure root-cause analysis → search-space bias + new seeds
  3. L2-L3 combined Sobol + active-learning search:
       - ~40 Sobol samples   (phase × L3)
       - ~25 active-learning evaluations (5 rounds × 5 candidates)
       - ~3  verification episodes on MLP top-k candidates
       Total ≈ 68 sims per iteration  vs ~500 in v1

Non-modified modules: attacks/, sim/, specs/, metrics/, policies/, tasks/.
"""
from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.metrics.stability import resolve_ground_geom_ids
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import EpisodeResult, WalkingTestRunner
from robostl.run_layer3_search import CROWNSensitivityAnalyzer, Layer3SearchConfig
from robostl.surrogate.acquisition import select_by_acquisition
from robostl.surrogate.data_buffer import DataBuffer, encode_l1
from robostl.surrogate.global_surrogate import GlobalSurrogate
from robostl.search.bayesian_optimizer import BayesianOptimizer
from robostl.tasks.walking import WalkingTask

# Re-use helper functions from v1 (non-private utilities)
from robostl.run_fuzz_loop import (
    SeedEntry,
    _build_attacks,
    _build_foot_geom_map,
    _compute_keep_score,
    _load_seed_pool,
    _min_distance,
    _mutate_l1,
    _mutate_phase,
    _mutate_terrain,
    _mutate_terrain_weighted,
    _normalize,
    _parse_range,
    _parse_rect,
    _parse_vec,
    _phase_to_time,
    _probe_phase,
    _recompute_keep_scores,
    _sample_terrain,
    _save_failure,
    _save_seed_pool,
    _select_seed,
    _weighted_distance,
)

# ---------------------------------------------------------------------------
# Sobol sampling helper
# ---------------------------------------------------------------------------

try:
    from scipy.stats.qmc import Sobol as _Sobol

    def _sobol_sample(n: int, d: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        m = int(np.ceil(np.log2(max(n, 2))))
        raw = _Sobol(d=d, scramble=True).random_base2(m)[:n]
        return (raw * (hi - lo) + lo).astype(np.float32)

    _SOBOL_AVAILABLE = True
except ImportError:
    _SOBOL_AVAILABLE = False

    def _sobol_sample(n: int, d: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.random.uniform(lo, hi, (n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Argument parsing (extends v1 with surrogate/LLM flags)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoboSTL-Fuzz v2: Surrogate + LLM enhanced fuzzing loop."
    )
    # ── Core (same as v1) ──────────────────────────────────────────────────
    parser.add_argument("--config", type=Path, default=DeployConfig.default_config_path())
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Loop iterations (0 means run indefinitely until interrupted).",
    )
    parser.add_argument("--seed-pool-size", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("fuzz_outputs"))
    parser.add_argument("--seed-dir", type=str, default="seeds")
    parser.add_argument("--failure-dir", type=str, default="failures")
    parser.add_argument("--exploit-cap", type=float, default=0.7)
    parser.add_argument("--push-min", type=str, default="-80,-80,-80")
    parser.add_argument("--push-max", type=str, default="80,80,80")
    parser.add_argument("--cmd-min", type=str, default="-1.0,-0.3,-0.5")
    parser.add_argument("--cmd-max", type=str, default="1.0,0.3,0.5")
    parser.add_argument("--friction-min", type=str, default="0.2,0.001,0.00001")
    parser.add_argument("--friction-max", type=str, default="2.0,0.02,0.01")
    parser.add_argument("--terrain-modes", type=str, default="flat,pit,bump")
    parser.add_argument("--terrain-center-range", type=str, default="0.5,2.0,-0.5,0.5")
    parser.add_argument("--terrain-radius-range", type=str, default="0.08,0.2")
    parser.add_argument("--terrain-depth-range", type=str, default="0.01,0.05")
    parser.add_argument("--terrain-height-range", type=str, default="0.01,0.05")
    parser.add_argument("--terrain-baseline", type=float, default=0.5)
    parser.add_argument("--phase-step", type=float, default=0.05)
    parser.add_argument("--l1-max-resample", type=int, default=5)
    parser.add_argument("--phase-mutation-range", type=float, default=0.1)
    parser.add_argument("--phase-mutation-samples", type=int, default=5)
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
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--state-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--vel-perturbation-scale", type=float, default=0.02)
    parser.add_argument("--sensitive-scale-factor", type=float, default=0.5)
    parser.add_argument("--nonsensitive-scale-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    # ── Surrogate (v2) ────────────────────────────────────────────────────
    parser.add_argument(
        "--surrogate-cold-start",
        type=int,
        default=60,
        help="Buffer size before surrogate is activated.",
    )
    parser.add_argument(
        "--surrogate-safe-threshold",
        type=float,
        default=0.5,
        help="Skip L1 configs where GPR predicts mu - 2*sigma > threshold.",
    )
    parser.add_argument(
        "--sobol-samples",
        type=int,
        default=40,
        help="Sobol initial samples per L2-L3 search.",
    )
    parser.add_argument(
        "--active-rounds",
        type=int,
        default=5,
        help="Active-learning rounds inside L2-L3 search.",
    )
    parser.add_argument(
        "--active-candidates",
        type=int,
        default=5,
        help="Episode evaluations per active-learning round.",
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=3,
        help="High-fidelity verification episodes at end of L2-L3 search.",
    )
    parser.add_argument(
        "--bo-candidates",
        type=int,
        default=1000,
        help="Sobol candidates evaluated by BayesianOptimizer per suggestion.",
    )
    # ── LLM (v2, all optional) ────────────────────────────────────────────
    parser.add_argument("--llm-base-url", type=str, default="")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--llm-api-key-env", type=str, default="OPENAI_API_KEY",
        help="Environment variable containing the LLM API key.",
    )
    parser.add_argument("--llm-seed-file", type=str, default="")
    parser.add_argument(
        "--llm-analysis-batch", type=int, default=5,
        help="Failure cases per LLM analysis call.",
    )
    # ── Shadow simulation (deviation-based STL) ───────────────────────────
    parser.add_argument(
        "--use-shadow", action="store_true", default=True,
        help="Use shadow simulation for deviation-based STL evaluation.",
    )
    parser.add_argument("--no-shadow", dest="use_shadow", action="store_false")
    # ── LLM mutation strategy ─────────────────────────────────────────────
    parser.add_argument(
        "--random-exploration-ratio", type=float, default=0.25,
        help="Fraction of iterations using pure random (not LLM-guided) exploration.",
    )
    parser.add_argument(
        "--strategy-update-interval", type=int, default=25,
        help="Update LLM mutation strategy every N iterations.",
    )
    parser.add_argument(
        "--strategy-failure-min", type=int, default=5,
        help="Min failure cases before triggering strategy generation.",
    )
    parser.add_argument(
        "--strategy-valid-iters", type=int, default=25,
        help="Default validity window (iterations) for each mutation strategy.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# L2-L3 combined surrogate search
# ---------------------------------------------------------------------------


def _l2l3_surrogate_search(
    runner: WalkingTestRunner,
    surrogate: GlobalSurrogate,
    buffer: DataBuffer,
    push: np.ndarray,
    friction: np.ndarray,
    terrain: dict,
    x_l1: np.ndarray,
    period: float,
    cycle_start: float,
    args: argparse.Namespace,
    analyzer: CROWNSensitivityAnalyzer,
    config: DeployConfig,
    deviation_config=None,
) -> tuple[float, np.ndarray, Optional[EpisodeResult], float]:
    """L2-L3 combined search: Sobol init → MLP active learning → verification.

    Returns:
        (best_rho, best_l3_full[24], best_episode, best_phase)
    """
    num_joints = config.num_actions  # 12
    l3_dim = num_joints * 2           # 24
    settle_time = args.settle_time
    push_duration = args.push_duration
    push_body = args.push_body
    pos_scale = args.state_perturbation_scale
    vel_scale = args.vel_perturbation_scale

    # Determine sensitive dims from CROWN for search-bound scaling
    sensitive_set: set[int] = set()
    if args.use_crown:
        try:
            import torch

            # Run a quick probe episode and capture the observation just before
            # the attack window.  G1MujocoRunner builds observations inside
            # step() via obs_builder.build(); after the final step the latest
            # obs is stored in obs_builder.obs_buffer.obs.
            runner.env.set_terrain(terrain)
            runner.env.attacks = _build_attacks(
                push, friction, settle_time + 0.5, push_duration, push_body
            )
            runner.env.reset()
            probe_steps = int((settle_time + 0.5) / config.simulation_dt)
            for _ in range(probe_steps):
                runner.env.step()
            # Retrieve the observation that was built during the last step
            obs_np = runner.env.obs_builder.build(
                runner.env.data,
                runner.env.action,
                runner.env.cmd,
                runner.env.sim_time,
                runner.env.counter,
            )
            sensitivity = analyzer.compute_sensitivity(
                torch.tensor(obs_np, dtype=torch.float32)
            )
            for d in sensitivity.sensitive_dims[:10]:
                # sensitive_dims are policy-input (obs) indices;
                # joint positions start at obs[9], so offset accordingly.
                joint_idx = max(0, int(d) - 9) % num_joints
                sensitive_set.add(joint_idx)
        except Exception:
            pass  # CROWN unavailable or failed; fall back to uniform scaling

    # Build per-joint position and velocity scales
    pos_scales = np.array(
        [
            pos_scale * (args.sensitive_scale_factor if j in sensitive_set
                         else args.nonsensitive_scale_factor)
            for j in range(num_joints)
        ],
        dtype=np.float32,
    )
    vel_scales = np.full(num_joints, vel_scale, dtype=np.float32)

    # Search space: [phase(1), pos_offsets(12), vel_offsets(12)] = 25D
    search_dim = 1 + l3_dim
    lo = np.concatenate([[0.0], -pos_scales, -vel_scales])
    hi = np.concatenate([[1.0], pos_scales, vel_scales])

    # ── Pre-compute shadow trace (zero-perturbation) for deviation-based STL ──
    shadow_trace = None
    if deviation_config is not None and getattr(args, "use_shadow", True):
        runner.env.set_terrain(terrain)
        shadow_trace = runner.run_shadow_episode(friction=friction, terrain=terrain)

    best_rho = float("inf")
    best_phase = 0.0
    best_l3 = np.zeros(l3_dim, dtype=np.float32)
    best_episode: Optional[EpisodeResult] = None

    # ── Step 2a: Sobol initial sampling ─────────────────────────────────
    n_sobol = args.sobol_samples
    sobol_cands = _sobol_sample(n_sobol, search_dim, lo, hi)

    for cand in sobol_cands:
        phase = float(np.clip(cand[0], 0.0, 1.0))
        x_l3_cand = cand[1:].copy()
        push_start = _phase_to_time(phase, period, cycle_start, settle_time)

        try:
            if shadow_trace is not None:
                rho = runner.run_episode_fast_shadow(
                    push=push,
                    friction=friction,
                    push_start=push_start,
                    push_duration=push_duration,
                    push_body=push_body,
                    joint_pos_offset=x_l3_cand[:num_joints].copy(),
                    joint_vel_offset=x_l3_cand[num_joints:].copy(),
                    terrain=terrain,
                    shadow_trace=shadow_trace,
                    deviation_config=deviation_config,
                )
            else:
                runner.env.set_terrain(terrain)
                runner.env.attacks = _build_attacks(push, friction, push_start, push_duration, push_body)
                ep = runner.run_episode_with_midpoint_perturbation(
                    perturbation_time=push_start,
                    joint_pos_offset=x_l3_cand[:num_joints].copy(),
                    joint_vel_offset=x_l3_cand[num_joints:].copy(),
                )
                rho = float(ep.metrics.get("stl_robustness", 0.0))
        except Exception:
            rho = 0.0

        x_l3_full = np.zeros(24, dtype=np.float32)
        x_l3_full[:l3_dim] = x_l3_cand
        buffer.add(x_l1, phase, x_l3_full, rho)

        if rho < best_rho:
            best_rho = rho
            best_phase = phase
            best_l3 = x_l3_full.copy()
            # best_episode populated in verification phase

    # Retrain MLP after initial batch
    if buffer.size >= 20:
        surrogate.retrain_mlp(buffer, mlp_epochs=50)

    # ── Step 2b: Active-learning rounds ──────────────────────────────────
    n_pool = 4096  # candidates evaluated by MLP per round
    n_active_rounds = args.active_rounds
    n_per_round = args.active_candidates

    for _round in range(n_active_rounds):
        pool = _sobol_sample(n_pool, search_dim, lo, hi)

        # Build full 36D feature vectors for MLP
        X_pool = np.zeros((n_pool, 36), dtype=np.float32)
        for i, cand in enumerate(pool):
            phase = float(np.clip(cand[0], 0.0, 1.0))
            x_l3_pool = np.zeros(24, dtype=np.float32)
            x_l3_pool[:l3_dim] = cand[1:]
            X_pool[i] = np.concatenate([x_l1, [phase], x_l3_pool])

        predictions = surrogate.mlp_predict_batch(X_pool)

        # Get already-sampled X_full for diversity reference
        _, existing_X, _ = buffer.get_training_data()
        selected_idx = select_by_acquisition(
            predictions,
            existing_X if len(existing_X) > 0 else None,
            n=n_per_round,
            exploitation_ratio=0.7,
        )

        for idx in selected_idx:
            cand = pool[idx]
            phase = float(np.clip(cand[0], 0.0, 1.0))
            x_l3_cand = cand[1:].copy()
            push_start = _phase_to_time(phase, period, cycle_start, settle_time)

            try:
                if shadow_trace is not None:
                    rho = runner.run_episode_fast_shadow(
                        push=push,
                        friction=friction,
                        push_start=push_start,
                        push_duration=push_duration,
                        push_body=push_body,
                        joint_pos_offset=x_l3_cand[:num_joints].copy(),
                        joint_vel_offset=x_l3_cand[num_joints:].copy(),
                        terrain=terrain,
                        shadow_trace=shadow_trace,
                        deviation_config=deviation_config,
                    )
                else:
                    runner.env.set_terrain(terrain)
                    runner.env.attacks = _build_attacks(
                        push, friction, push_start, push_duration, push_body
                    )
                    ep = runner.run_episode_with_midpoint_perturbation(
                        perturbation_time=push_start,
                        joint_pos_offset=x_l3_cand[:num_joints].copy(),
                        joint_vel_offset=x_l3_cand[num_joints:].copy(),
                    )
                    rho = float(ep.metrics.get("stl_robustness", 0.0))
            except Exception:
                rho = 0.0

            x_l3_full = np.zeros(24, dtype=np.float32)
            x_l3_full[:l3_dim] = x_l3_cand
            buffer.add(x_l1, phase, x_l3_full, rho)

            if rho < best_rho:
                best_rho = rho
                best_phase = phase
                best_l3 = x_l3_full.copy()
                # best_episode populated in verification phase

        # Partial MLP retrain after each round
        surrogate.retrain_mlp(buffer, mlp_epochs=30)

    # ── Step 2c: Verification — MLP top-k high-fidelity episodes ─────────
    n_verify = args.verify_top_k
    top_cands = surrogate.get_top_k(
        x_l1,
        k=n_verify,
        n_candidates=5000,
        l3_scale=float(np.maximum(pos_scales.max(), vel_scales.max())),
    )

    for x_full_cand in top_cands:
        phase = float(np.clip(x_full_cand[DataBuffer.L1_DIM], 0.0, 1.0))
        x_l3_cand = x_full_cand[DataBuffer.L1_DIM + 1 :].copy()
        push_start = _phase_to_time(phase, period, cycle_start, settle_time)

        try:
            if shadow_trace is not None:
                result = runner.run_episode_with_shadow(
                    push=push,
                    friction=friction,
                    push_start=push_start,
                    push_duration=push_duration,
                    push_body=push_body,
                    joint_pos_offset=x_l3_cand[:num_joints].copy(),
                    joint_vel_offset=x_l3_cand[num_joints:].copy(),
                    terrain=terrain,
                    shadow_trace=shadow_trace,
                    deviation_config=deviation_config,
                )
                rho = float(result.metrics.get("stl_robustness", 0.0))
            else:
                runner.env.set_terrain(terrain)
                runner.env.attacks = _build_attacks(
                    push, friction, push_start, push_duration, push_body
                )
                result = runner.run_episode_with_midpoint_perturbation(
                    perturbation_time=push_start,
                    joint_pos_offset=x_l3_cand[:num_joints].copy(),
                    joint_vel_offset=x_l3_cand[num_joints:].copy(),
                )
                rho = float(result.metrics.get("stl_robustness", 0.0))
        except Exception:
            rho = 0.0
            result = None

        x_l3_full = np.zeros(24, dtype=np.float32)
        x_l3_full[:l3_dim] = x_l3_cand[:l3_dim]
        buffer.add(x_l1, phase, x_l3_full, rho)

        if rho < best_rho:
            best_rho = rho
            best_phase = phase
            best_l3 = x_l3_full.copy()
            best_episode = result

    return best_rho, best_l3, best_episode, best_phase


# ---------------------------------------------------------------------------
# Cold-start fallback: v1 grid + CMA-ES style search using existing code path
# ---------------------------------------------------------------------------


def _l2l3_coldstart_search(
    runner: WalkingTestRunner,
    push: np.ndarray,
    friction: np.ndarray,
    terrain: dict,
    x_l1: np.ndarray,
    period: float,
    cycle_start: float,
    buffer: DataBuffer,
    args: argparse.Namespace,
    config: DeployConfig,
    deviation_config=None,
) -> tuple[float, np.ndarray, Optional[EpisodeResult], float]:
    """Simplified cold-start search: grid phase scan + random L3 perturbations.

    Used until the surrogate has enough data (cold_start_threshold).
    Collects data into the buffer to warm up the surrogate.
    """
    num_joints = config.num_actions
    settle_time = args.settle_time
    push_duration = args.push_duration
    push_body = args.push_body

    # Phase grid scan
    phases = [i * args.phase_step for i in range(int(1.0 / args.phase_step) + 1)]

    best_rho = float("inf")
    best_phase = 0.0
    best_l3 = np.zeros(num_joints * 2, dtype=np.float32)
    best_episode: Optional[EpisodeResult] = None

    runner.env.set_terrain(terrain)

    # Pre-compute shadow trace for deviation-based STL
    shadow_trace = None
    if deviation_config is not None and getattr(args, "use_shadow", True):
        shadow_trace = runner.run_shadow_episode(friction=friction, terrain=terrain)

    for phase in phases:
        push_start = _phase_to_time(phase, period, cycle_start, settle_time)
        pos_offset = np.random.uniform(
            -args.state_perturbation_scale,
            args.state_perturbation_scale,
            num_joints,
        ).astype(np.float32)
        vel_offset = np.random.uniform(
            -args.vel_perturbation_scale,
            args.vel_perturbation_scale,
            num_joints,
        ).astype(np.float32)

        try:
            if shadow_trace is not None:
                rho = runner.run_episode_fast_shadow(
                    push=push,
                    friction=friction,
                    push_start=push_start,
                    push_duration=push_duration,
                    push_body=push_body,
                    joint_pos_offset=pos_offset,
                    joint_vel_offset=vel_offset,
                    terrain=terrain,
                    shadow_trace=shadow_trace,
                    deviation_config=deviation_config,
                )
            else:
                runner.env.attacks = _build_attacks(push, friction, push_start, push_duration, push_body)
                ep = runner.run_episode_with_midpoint_perturbation(
                    perturbation_time=push_start,
                    joint_pos_offset=pos_offset,
                    joint_vel_offset=vel_offset,
                )
                rho = float(ep.metrics.get("stl_robustness", 0.0))
        except Exception:
            rho = 0.0

        x_l3_full = np.zeros(24, dtype=np.float32)
        x_l3_full[:num_joints] = pos_offset
        x_l3_full[num_joints : num_joints * 2] = vel_offset
        buffer.add(x_l1, phase, x_l3_full, rho)

        if rho < best_rho:
            best_rho = rho
            best_phase = phase
            best_l3 = x_l3_full.copy()

    return best_rho, best_l3, best_episode, best_phase


# ---------------------------------------------------------------------------
# LLM seed injection helper
# ---------------------------------------------------------------------------


def _inject_llm_seeds(
    seed_pool: list[SeedEntry],
    llm_seeds: list[dict],
    args: argparse.Namespace,
) -> int:
    """Convert raw LLM seed dicts into SeedEntry objects and append to pool."""
    push_min = _parse_vec(args.push_min, 3)
    push_max = _parse_vec(args.push_max, 3)
    fric_min = _parse_vec(args.friction_min, 3)
    fric_max = _parse_vec(args.friction_max, 3)
    cmd_min = _parse_vec(args.cmd_min, 3)
    cmd_max = _parse_vec(args.cmd_max, 3)

    added = 0
    for seed_dict in llm_seeds:
        try:
            push = np.clip(
                np.array(seed_dict.get("push", [0, 0, 0]), dtype=np.float32),
                push_min, push_max,
            )
            friction = np.clip(
                np.array(seed_dict.get("friction", [1.0, 0.01, 0.001]), dtype=np.float32),
                fric_min, fric_max,
            )
            cmd = np.clip(
                np.array(seed_dict.get("cmd", [0.5, 0.0, 0.0]), dtype=np.float32),
                cmd_min, cmd_max,
            )
            terrain = seed_dict.get("terrain", {"mode": "flat"})
            seed_pool.append(
                SeedEntry(
                    push=push,
                    friction=friction,
                    cmd=cmd,
                    terrain=terrain,
                    phase=0.0,
                    push_start=0.0,
                    robustness=0.0,
                    keep_score=0.0,
                    joint_pos_offset=None,
                    joint_vel_offset=None,
                )
            )
            added += 1
        except Exception as exc:
            print(f"[LLM] Seed inject error: {exc}")
    return added


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    push_min = _parse_vec(args.push_min, 3)
    push_max = _parse_vec(args.push_max, 3)
    cmd_min = _parse_vec(args.cmd_min, 3)
    cmd_max = _parse_vec(args.cmd_max, 3)
    fric_min = _parse_vec(args.friction_min, 3)
    fric_max = _parse_vec(args.friction_max, 3)

    weights = np.concatenate(
        [
            1.0 / np.maximum(push_max - push_min, 1e-6),
            1.0 / np.maximum(fric_max - fric_min, 1e-6),
            np.array([1.0], dtype=np.float32),
        ]
    )

    # ── Output directories ────────────────────────────────────────────────
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / args.seed_dir
    failure_dir = output_dir / args.failure_dir

    # ── Seed pool ─────────────────────────────────────────────────────────
    seed_pool = _load_seed_pool(seed_dir)
    if seed_pool:
        _recompute_keep_scores(seed_pool, weights)

    # ── Config + policy ───────────────────────────────────────────────────
    config = DeployConfig.from_yaml(args.config)
    if args.policy_path is not None:
        config = DeployConfig(
            **{**config.__dict__, "policy_path": args.policy_path.expanduser().resolve()}
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

    # ── Deviation STL config (shadow simulation mode) ─────────────────────
    deviation_config = None
    if getattr(args, "use_shadow", True):
        try:
            from robostl.specs.deviation_config import DeviationConfig
            deviation_config = DeviationConfig.from_dict(config.deviation_stl_config)
            print(f"[Shadow] Deviation STL enabled: {deviation_config}")
        except Exception as exc:
            print(f"[Shadow] Failed to init deviation config: {exc}. Shadow mode disabled.")

    # ── Surrogate model ───────────────────────────────────────────────────
    buffer_path = output_dir / "surrogate_data.npz"
    buffer = DataBuffer.load_or_create(buffer_path)
    surrogate = GlobalSurrogate(l1_dim=DataBuffer.L1_DIM, full_dim=DataBuffer.FULL_DIM)
    use_surrogate = False
    if buffer.size >= args.surrogate_cold_start:
        print(f"[Surrogate] Warm-starting from {buffer.size} existing data points.")
        surrogate.train(buffer)
        use_surrogate = True

    # L1 bounds array for BayesianOptimizer  [push(3), friction(3), terrain_enc(5)] = 11D
    # Note: terrain dims have loose bounds; EI will find the interesting region
    l1_lo = np.concatenate([push_min, fric_min, np.zeros(5, dtype=np.float32)])
    l1_hi = np.concatenate([push_max, fric_max, np.ones(5, dtype=np.float32) * 2.0])
    bo_bounds = np.stack([l1_lo, l1_hi], axis=1)  # [11, 2]
    bo = BayesianOptimizer(bounds=bo_bounds, n_candidates=args.bo_candidates)

    # ── CROWN sensitivity analyzer (kept from v1) ─────────────────────────
    analyzer = CROWNSensitivityAnalyzer(
        policy, epsilon=args.epsilon, use_crown=args.use_crown
    )

    # ── LLM components (optional) ─────────────────────────────────────────
    llm_client = None
    seed_generator = None
    failure_analyzer = None

    if args.llm_base_url:
        try:
            from robostl.llm.client import LLMClient
            from robostl.llm.seed_generator import ScenarioSeedGenerator
            from robostl.llm.root_cause_analyzer import FailureAnalyzer

            api_key = os.environ.get(args.llm_api_key_env, "dummy")
            llm_client = LLMClient(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=api_key,
            )
            seed_generator = ScenarioSeedGenerator(llm_client)
            failure_analyzer = FailureAnalyzer(
                llm_client, batch_size=args.llm_analysis_batch
            )
            print(f"[LLM] Enabled: {args.llm_base_url} / {args.llm_model}")
        except Exception as exc:
            print(f"[LLM] Init failed: {exc}. LLM features disabled.")

    # Inject pre-generated LLM seeds
    if args.llm_seed_file and Path(args.llm_seed_file).exists():
        from robostl.llm.seed_generator import ScenarioSeedGenerator

        llm_seeds = ScenarioSeedGenerator.load_seeds(Path(args.llm_seed_file))
        n_added = _inject_llm_seeds(seed_pool, llm_seeds, args)
        print(f"[LLM] Injected {n_added} seeds from {args.llm_seed_file}")
        if seed_pool:
            _recompute_keep_scores(seed_pool, weights)

    # ── Failure tracking for LLM ──────────────────────────────────────────
    failure_buffer: list[dict] = []
    all_analyses: list = []

    # ── LLM mutation strategy ──────────────────────────────────────────────
    mutation_strategy = None
    strategy_remaining_iters: int = 0
    strategy_generator = None
    failure_buffer_for_strategy: list[dict] = []

    if llm_client is not None:
        try:
            from robostl.llm.mutation_strategy import MutationStrategyGenerator
            strategy_generator = MutationStrategyGenerator(llm_client)
            print("[Strategy] Mutation strategy generator initialized.")
        except Exception as exc:
            print(f"[Strategy] Init failed: {exc}")

    # ── Main fuzzing loop ─────────────────────────────────────────────────
    if args.iterations <= 0:
        iteration_iter = iter(int, 1)
    else:
        iteration_iter = range(args.iterations)

    for iteration in iteration_iter:
        exploit_ratio = min(args.exploit_cap, len(seed_pool) * 0.02)
        _rand = random.random()
        # Three-channel sampling:
        #   [0, random_ratio)                          → pure random (no strategy)
        #   [random_ratio, random_ratio+exploit_ratio) → exploit + LLM strategy
        #   rest                                       → BO / pure random
        use_pure_random = _rand < args.random_exploration_ratio
        use_exploit = (not use_pure_random) and (
            _rand < args.random_exploration_ratio + exploit_ratio
        )
        active_strategy = (
            mutation_strategy
            if (use_exploit and strategy_remaining_iters > 0)
            else None
        )

        # ── 1. L1 parameter sampling ──────────────────────────────────────
        attempts = 0
        skip_iteration = False

        while True:
            attempts += 1

            if use_exploit and seed_pool:
                seed = _select_seed(seed_pool)
                push, friction, cmd, terrain = _mutate_l1(
                    seed, seed_pool, push_min, push_max, fric_min, fric_max,
                    cmd_min, cmd_max, args,
                    strategy=active_strategy,
                )
                phase_mode = random.choice(["util", "random", "crossover"])
                mate_phase = None
                if phase_mode == "crossover" and len(seed_pool) > 1:
                    mate = random.choice([s for s in seed_pool if s is not seed] or [seed])
                    mate_phase = mate.phase
                phases_l2 = _mutate_phase(
                    seed.phase, phase_mode, args.phase_mutation_samples,
                    args.phase_mutation_range, mate_phase=mate_phase,
                )
            elif use_surrogate and surrogate.is_ready and not use_pure_random:
                # Bayesian Optimization suggests L1 candidate
                bo_cands = bo.suggest(surrogate, n_suggestions=1)
                x_l1_suggest = bo_cands[0] if bo_cands else None
                if x_l1_suggest is not None:
                    # Decode suggest back to push/friction (terrain kept random)
                    push = np.clip(x_l1_suggest[:3].astype(np.float32), push_min, push_max)
                    friction = np.clip(x_l1_suggest[3:6].astype(np.float32), fric_min, fric_max)
                    cmd = np.random.uniform(cmd_min, cmd_max).astype(np.float32)
                    terrain = _sample_terrain(args)
                else:
                    push = np.random.uniform(push_min, push_max).astype(np.float32)
                    friction = np.random.uniform(fric_min, fric_max).astype(np.float32)
                    cmd = np.random.uniform(cmd_min, cmd_max).astype(np.float32)
                    terrain = _sample_terrain(args)
                phase_mode = "grid"
                phases_l2 = [i * args.phase_step for i in range(int(1.0 / args.phase_step) + 1)]
            else:
                push = np.random.uniform(push_min, push_max).astype(np.float32)
                friction = np.random.uniform(fric_min, fric_max).astype(np.float32)
                cmd = np.random.uniform(cmd_min, cmd_max).astype(np.float32)
                terrain = _sample_terrain(args)
                phase_mode = "grid"
                phases_l2 = [i * args.phase_step for i in range(int(1.0 / args.phase_step) + 1)]

            # ── 2. L1 surrogate pre-filter ────────────────────────────────
            x_l1 = encode_l1(push, friction, terrain)
            if use_surrogate and surrogate._gpr_fitted:
                mu, sigma = surrogate.gpr_predict(x_l1)
                if mu - 2.0 * sigma > args.surrogate_safe_threshold:
                    # Surrogate predicts this config is almost certainly safe — skip
                    print(
                        f"[{iteration+1}] Skipping L1 (GPR: μ={mu:.3f}, σ={sigma:.3f})"
                    )
                    skip_iteration = True
                    break  # move to next iteration

            # ── 3. Phase probing ──────────────────────────────────────────
            runner.env.cmd = cmd.copy()
            period, cycle_start = _probe_phase(
                runner,
                friction=friction,
                terrain=terrain,
                duration_s=args.probe_duration,
                min_step_duration=args.min_step_duration,
                min_period=args.min_period,
                default_period=args.default_period,
            )

            # Quick phase-scan to check if this L1 config immediately causes falls
            runner.env.set_terrain(terrain)
            best_phase_quick = phases_l2[0]
            best_rho_quick = float("inf")
            best_fallen_quick = False
            for ph in phases_l2[:5]:  # check just a few phases for fall detection
                ps = _phase_to_time(ph, period, cycle_start, args.settle_time)
                atks = _build_attacks(push, friction, ps, args.push_duration, args.push_body)
                runner.env.attacks = atks
                try:
                    result_q = runner.run_episode()
                    rho_q = float(result_q.metrics.get("stl_robustness", 0.0))
                    fallen_q = bool(result_q.metrics.get("fallen", False))
                except Exception:
                    rho_q = 0.0
                    fallen_q = False
                if rho_q < best_rho_quick:
                    best_rho_quick = rho_q
                    best_phase_quick = ph
                    best_fallen_quick = fallen_q

            if not best_fallen_quick:
                break
            if attempts >= args.l1_max_resample:
                print(f"[Warning] Iter {iteration+1}: L1 causes fall after {attempts} resamples; skipping.")
                skip_iteration = True
                break

        if skip_iteration:
            continue

        # ── 4. L2-L3 combined search ──────────────────────────────────────
        x_l1 = encode_l1(push, friction, terrain)  # recompute after possible terrain change

        if use_surrogate:
            refined_rho, best_l3, best_episode, best_phase = _l2l3_surrogate_search(
                runner=runner,
                surrogate=surrogate,
                buffer=buffer,
                push=push,
                friction=friction,
                terrain=terrain,
                x_l1=x_l1,
                period=period,
                cycle_start=cycle_start,
                args=args,
                analyzer=analyzer,
                config=config,
                deviation_config=deviation_config,
            )
        else:
            refined_rho, best_l3, best_episode, best_phase = _l2l3_coldstart_search(
                runner=runner,
                push=push,
                friction=friction,
                terrain=terrain,
                x_l1=x_l1,
                period=period,
                cycle_start=cycle_start,
                buffer=buffer,
                args=args,
                config=config,
                deviation_config=deviation_config,
            )

        push_start = _phase_to_time(best_phase, period, cycle_start, args.settle_time)
        bo.update_best(refined_rho)

        # ── 5. Seed pool update ───────────────────────────────────────────
        vec = np.concatenate([push, friction, np.array([best_phase], dtype=np.float32)])
        distance = _weighted_distance(vec, seed_pool, weights)
        rob_vals = [s.robustness for s in seed_pool] + [refined_rho]
        dist_vals = [_min_distance(s, seed_pool, weights) for s in seed_pool] + [distance]
        keep_score = _compute_keep_score(
            refined_rho,
            distance,
            float(min(rob_vals)),
            float(max(rob_vals)),
            float(min(dist_vals)),
            float(max(dist_vals)),
        )

        num_joints = config.num_actions
        new_seed = SeedEntry(
            push=push,
            friction=friction,
            cmd=cmd,
            terrain=terrain,
            phase=best_phase,
            push_start=push_start,
            robustness=refined_rho,
            keep_score=keep_score,
            joint_pos_offset=best_l3[:num_joints],
            joint_vel_offset=best_l3[num_joints : num_joints * 2],
        )

        if (
            len(seed_pool) < args.seed_pool_size
            or keep_score >= min(s.keep_score for s in seed_pool)
        ):
            seed_pool.append(new_seed)
            _recompute_keep_scores(seed_pool, weights)
            seed_pool.sort(key=lambda s: s.keep_score, reverse=True)
            if len(seed_pool) > args.seed_pool_size:
                seed_pool.pop()

        # ── 6. Failure handling + LLM analysis ───────────────────────────
        stl_details = best_episode.stl if best_episode is not None else None
        failure_payload = {
            "iteration": iteration,
            "mode": "exploit" if use_exploit else ("bo" if use_surrogate else "random"),
            "push": push.tolist(),
            "friction": friction.tolist(),
            "cmd": cmd.tolist(),
            "terrain": terrain,
            "phase": best_phase,
            "push_start": push_start,
            "push_duration": args.push_duration,
            "push_body": args.push_body,
            "period_s": period,
            "cycle_start_s": cycle_start,
            "settle_time": args.settle_time,
            "joint_pos_offset": new_seed.joint_pos_offset.tolist()
            if new_seed.joint_pos_offset is not None else None,
            "joint_vel_offset": new_seed.joint_vel_offset.tolist()
            if new_seed.joint_vel_offset is not None else None,
            "robustness": refined_rho,
            "keep_score": keep_score,
            "stl_details": stl_details,
            "use_surrogate": use_surrogate,
            "config": str(args.config),
        }

        if refined_rho < 0:
            _save_failure(failure_dir, iteration, failure_payload)
            failure_buffer_for_strategy.append(failure_payload)

            if failure_analyzer is not None and failure_analyzer.should_analyze(
                failure_payload, all_analyses
            ):
                failure_buffer.append(failure_payload)

            if (
                failure_analyzer is not None
                and len(failure_buffer) >= args.llm_analysis_batch
            ):
                analyses = failure_analyzer.analyze_batch(failure_buffer)
                all_analyses.extend(analyses)
                pattern = failure_analyzer.summarize_patterns(all_analyses)

                # (Seed injection replaced by LLM mutation strategy; see step 6b below)
                if seed_generator is not None and pattern.new_scenario_descriptions:
                    try:
                        new_seeds = seed_generator.generate_from_descriptions(
                            pattern.new_scenario_descriptions
                        )
                        n_added = _inject_llm_seeds(seed_pool, new_seeds, args)
                        print(f"[LLM] Injected {n_added} new seeds from failure analysis.")
                        if seed_pool:
                            _recompute_keep_scores(seed_pool, weights)
                    except Exception as exc:
                        print(f"[LLM] Seed generation error: {exc}")

                failure_buffer.clear()

        # ── 6b. LLM mutation strategy update ─────────────────────────────
        if strategy_remaining_iters > 0:
            strategy_remaining_iters -= 1

        if (
            strategy_generator is not None
            and (iteration + 1) % args.strategy_update_interval == 0
            and len(failure_buffer_for_strategy) >= args.strategy_failure_min
        ):
            try:
                from robostl.llm.mutation_strategy import compute_pool_stats
                pool_stats = compute_pool_stats(seed_pool)
                new_strategy = strategy_generator.generate_strategy(
                    failure_buffer_for_strategy,
                    pool_stats=pool_stats,
                    valid_for_iterations=args.strategy_valid_iters,
                )
                mutation_strategy = new_strategy
                strategy_remaining_iters = new_strategy.valid_for_iterations
                print(
                    f"[Strategy] Updated at iter {iteration+1}: "
                    f"conf={new_strategy.confidence:.2f}, "
                    f"valid={new_strategy.valid_for_iterations} iters. "
                    f"Reason: {new_strategy.reasoning[:80]}"
                )
            except Exception as exc:
                print(f"[Strategy] Generation error: {exc}")

        # ── 7. Periodic surrogate maintenance ─────────────────────────────
        retrain_every = 10
        if (iteration + 1) % retrain_every == 0 and buffer.size >= args.surrogate_cold_start:
            surrogate.retrain(buffer)
            buffer.save(buffer_path)
            if not use_surrogate:
                use_surrogate = True
                print(
                    f"[Surrogate] Activated after {buffer.size} data points "
                    f"at iteration {iteration+1}."
                )

        # ── 8. Save progress ──────────────────────────────────────────────
        _save_seed_pool(seed_dir, seed_pool)

        print(
            f"[{iteration+1}/{args.iterations}] "
            f"mode={'exploit' if use_exploit else ('bo' if use_surrogate else 'random')} "
            f"rho={refined_rho:.4f} keep={keep_score:.4f} "
            f"seeds={len(seed_pool)} buffer={buffer.size} "
            f"surrogate={'on' if use_surrogate else 'off'}"
        )

    # Final surrogate save
    buffer.save(buffer_path)


if __name__ == "__main__":
    main()
