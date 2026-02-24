"""偏差型 Walking STL 规约评估器。

核心思路：
  attack_trace - shadow_trace = delta_trace
STL 谓词基于 delta_trace 的偏差信号进行判定，
阈值为与 cmd 无关的常数（由 DeviationConfig 提供）。

下游接口与 CompositeWalkingSpec 保持一致：
  evaluate(shadow_trace, attack_trace) -> ExtendedSTLResult
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from robostl.specs.deviation_config import DeviationConfig
from robostl.specs.extended_result import Diagnostics, ExtendedSTLResult, LayerResult
from robostl.specs.stl_evaluator import STLTrace


def _safe_min(arr: np.ndarray) -> float:
    """有限值最小值（忽略 NaN/Inf）。"""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _align_and_filter(
    shadow: STLTrace,
    attack: STLTrace,
    start_time: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """对齐两条 trace 并截取评估窗口。

    Returns:
        (eval_time, shadow_signals_filtered, attack_signals_filtered)
        三者索引一一对应。
    """
    n = min(len(shadow.time), len(attack.time))
    time = shadow.time[:n]
    mask = time >= start_time
    eval_time = time[mask]

    shadow_filtered: Dict[str, np.ndarray] = {}
    attack_filtered: Dict[str, np.ndarray] = {}
    for key in shadow.signals:
        if key in attack.signals:
            s_arr = shadow.signals[key][:n][mask]
            a_arr = attack.signals[key][:n][mask]
            shadow_filtered[key] = s_arr
            attack_filtered[key] = a_arr

    return eval_time, shadow_filtered, attack_filtered


class DeviationWalkingSpec:
    """基于偏差信号的 Walking STL 评估器。

    比较攻击仿真与影子仿真的偏差，用与 cmd 无关的阈值判定是否失效。
    输出格式与 CompositeWalkingSpec 完全一致（ExtendedSTLResult）。
    """

    def __init__(self, config: DeviationConfig) -> None:
        self.config = config

    def evaluate(
        self,
        shadow_trace: STLTrace,
        attack_trace: STLTrace,
    ) -> ExtendedSTLResult:
        """计算偏差鲁棒性并输出扩展结果。

        Args:
            shadow_trace: 零扰动影子仿真的 STLTrace
            attack_trace: 施加攻击的仿真 STLTrace
        """
        start_time = float(self.config.evaluation_start_time)
        eval_time, shadow_sig, attack_sig = _align_and_filter(
            shadow_trace, attack_trace, start_time
        )

        # 空窗口：直接返回安全结果
        if eval_time.size == 0:
            empty_diag = Diagnostics(
                first_violation_time=None,
                most_violated_predicate=None,
                robustness_trajectories={},
                violation_sequence=[],
            )
            inf_layer = LayerResult(ok=True, robustness=float("inf"), details={})
            return ExtendedSTLResult(
                ok=True,
                robustness=float("inf"),
                safety=inf_layer,
                stability=inf_layer,
                performance=None,
                diagnostics=empty_diag,
            )

        # ------------------------------------------------------------------
        # Safety 层谓词
        # ------------------------------------------------------------------
        safety_robs: list[float] = []
        safety_details: Dict[str, dict] = {}
        trajectories: Dict[str, list[float]] = {}
        traj_times: Dict[str, np.ndarray] = {}

        # 1. 绝对高度下界：height_attack > absolute_min_height
        if "height" in attack_sig:
            h_attack = attack_sig["height"]
            robs_series = h_attack - self.config.absolute_min_height
            rob = _safe_min(robs_series)
            safety_robs.append(rob)
            safety_details["absolute_height"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["absolute_height"] = robs_series.tolist()
            traj_times["absolute_height"] = eval_time

        # 2. 绝对倾角上界：tilt_attack < absolute_max_tilt
        if "tilt" in attack_sig:
            t_attack = attack_sig["tilt"]
            robs_series = self.config.absolute_max_tilt - t_attack
            rob = _safe_min(robs_series)
            safety_robs.append(rob)
            safety_details["absolute_tilt"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["absolute_tilt"] = robs_series.tolist()
            traj_times["absolute_tilt"] = eval_time

        # 3. 高度偏差约束：|delta_height| < max_delta_height
        if "height" in shadow_sig and "height" in attack_sig:
            dh = np.abs(attack_sig["height"] - shadow_sig["height"])
            robs_series = self.config.max_delta_height - dh
            rob = _safe_min(robs_series)
            safety_robs.append(rob)
            safety_details["delta_height"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["delta_height"] = robs_series.tolist()
            traj_times["delta_height"] = eval_time

        # 4. 倾角偏差约束：|delta_tilt| < max_delta_tilt
        if "tilt" in shadow_sig and "tilt" in attack_sig:
            dt = np.abs(attack_sig["tilt"] - shadow_sig["tilt"])
            robs_series = self.config.max_delta_tilt - dt
            rob = _safe_min(robs_series)
            safety_robs.append(rob)
            safety_details["delta_tilt"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["delta_tilt"] = robs_series.tolist()
            traj_times["delta_tilt"] = eval_time

        # ------------------------------------------------------------------
        # Stability 层谓词
        # ------------------------------------------------------------------
        stability_robs: list[float] = []
        stability_details: Dict[str, dict] = {}

        # 5. 角速度偏差约束：|delta_angular_velocity| < max_delta_angular_velocity
        if "angular_velocity" in shadow_sig and "angular_velocity" in attack_sig:
            dav = np.abs(attack_sig["angular_velocity"] - shadow_sig["angular_velocity"])
            robs_series = self.config.max_delta_angular_velocity - dav
            rob = _safe_min(robs_series)
            stability_robs.append(rob)
            stability_details["delta_angular_velocity"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["delta_angular_velocity"] = robs_series.tolist()
            traj_times["delta_angular_velocity"] = eval_time

        # 6. 速度误差偏差约束：|delta_velocity_error| < max_delta_velocity_error
        if "velocity_error" in shadow_sig and "velocity_error" in attack_sig:
            dve = np.abs(attack_sig["velocity_error"] - shadow_sig["velocity_error"])
            robs_series = self.config.max_delta_velocity_error - dve
            rob = _safe_min(robs_series)
            stability_robs.append(rob)
            stability_details["delta_velocity_error"] = {"ok": rob >= 0.0, "robustness": rob}
            trajectories["delta_velocity_error"] = robs_series.tolist()
            traj_times["delta_velocity_error"] = eval_time

        # ------------------------------------------------------------------
        # 聚合
        # ------------------------------------------------------------------
        safety_robustness = _safe_min(
            np.array(safety_robs, dtype=np.float32)
        ) if safety_robs else float("nan")
        stability_robustness = _safe_min(
            np.array(stability_robs, dtype=np.float32)
        ) if stability_robs else float("nan")

        safety_ok = bool(np.isfinite(safety_robustness) and safety_robustness >= 0.0)
        stability_ok = bool(np.isfinite(stability_robustness) and stability_robustness >= 0.0)

        safety_result = LayerResult(ok=safety_ok, robustness=safety_robustness, details=safety_details)
        stability_result = LayerResult(ok=stability_ok, robustness=stability_robustness, details=stability_details)

        enabled = [r for r in [safety_robustness, stability_robustness] if np.isfinite(r)]
        overall_robustness = _safe_min(np.array(enabled, dtype=np.float32)) if enabled else float("nan")
        overall_ok = bool(np.isfinite(overall_robustness) and overall_robustness >= 0.0)

        # ------------------------------------------------------------------
        # 诊断
        # ------------------------------------------------------------------
        first_violation_time: Optional[float] = None
        most_violated_pred: Optional[str] = None
        worst_value = float("inf")
        violation_sequence: list[tuple[float, str]] = []

        for name, series in trajectories.items():
            values = np.array(series, dtype=np.float32)
            if np.isfinite(values).any():
                min_val = float(np.nanmin(values))
                if min_val < worst_value:
                    worst_value = min_val
                    most_violated_pred = name
                viol_mask = values < 0.0
                if np.any(viol_mask):
                    idx = int(np.where(viol_mask)[0][0])
                    t_arr = traj_times.get(name, eval_time)
                    t_val = float(t_arr[idx]) if idx < len(t_arr) else float("nan")
                    if first_violation_time is None or t_val < first_violation_time:
                        first_violation_time = t_val
                    violation_sequence.append((t_val, name))

        violation_sequence.sort(key=lambda item: item[0])
        diagnostics = Diagnostics(
            first_violation_time=first_violation_time,
            most_violated_predicate=most_violated_pred,
            robustness_trajectories={k: v for k, v in trajectories.items()},
            violation_sequence=[
                {"time": float(t), "predicate": p} for t, p in violation_sequence
            ],
        )

        return ExtendedSTLResult(
            ok=overall_ok,
            robustness=overall_robustness,
            safety=safety_result,
            stability=stability_result,
            performance=None,
            diagnostics=diagnostics,
        )
