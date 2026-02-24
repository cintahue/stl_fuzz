from __future__ import annotations
"""组合式 Walking STL 规约实现。

核心职责：
1) 按配置构造 safety/stability/performance 三层谓词；
2) 在指定时间窗内计算每个谓词与每层的鲁棒性；
3) 聚合全局鲁棒性并生成诊断信息。
"""

from typing import Dict, List, Optional

import numpy as np

from robostl.specs.extended_result import Diagnostics, ExtendedSTLResult, LayerResult
from robostl.specs.predicates import PredicateSpec, greater_than, less_than
from robostl.specs.spec_config import SpecConfig
from robostl.specs.stl_evaluator import STLTrace


def _safe_min(values: np.ndarray) -> float:
    """有限值最小值（忽略 NaN/Inf）。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _safe_max(values: np.ndarray) -> float:
    """有限值最大值（忽略 NaN/Inf）。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _filter_trace(trace: STLTrace, mask: Optional[np.ndarray]) -> STLTrace:
    """按时间掩码过滤 trace。

    若掩码为空或全 False，返回空轨迹（保持字段完整）。
    """
    if mask is None:
        return trace
    if not np.any(mask):
        return STLTrace(
            time=trace.time[:0],
            signals={key: value[:0] for key, value in trace.signals.items()},
        )
    return STLTrace(
        time=trace.time[mask],
        signals={key: value[mask] for key, value in trace.signals.items()},
    )


class CompositeWalkingSpec:
    """扩展 walking 规约评估器。"""
    def __init__(self, config: SpecConfig) -> None:
        self.config = config

    def _build_predicates(self) -> List[PredicateSpec]:
        """根据配置构造分层谓词集合。"""
        safety = [
            greater_than("height", "height", self.config.h_min, "safety"),
            less_than("tilt", "tilt", self.config.max_tilt_deg, "safety"),
            less_than("max_torque", "max_torque", self.config.max_torque, "safety"),
        ]
        stability = [
            less_than(
                "angular_velocity",
                "angular_velocity",
                self.config.max_angular_velocity,
                "stability",
            ),
        ]
        if self.config.use_zmp:
            stability.append(
                greater_than("zmp_margin", "zmp_margin", self.config.zmp_margin, "stability")
            )
        else:
            stability.append(
                greater_than(
                    "stability_margin",
                    "stability_margin",
                    self.config.stability_margin,
                    "stability",
                )
            )

        performance = [
            less_than(
                "velocity_error",
                "velocity_error",
                self.config.max_velocity_error,
                "performance",
            ),
            greater_than(
                "foot_clearance",
                "foot_clearance",
                self.config.min_foot_clearance,
                "performance",
            ),
            less_than(
                "action_delta",
                "action_delta",
                self.config.max_action_delta,
                "performance",
            ),
        ]
        return safety + stability + performance

    def evaluate(self, trace: STLTrace) -> ExtendedSTLResult:
        """执行分层规约评估并输出扩展结果。"""
        predicates = self._build_predicates()

        time = trace.time
        # 全局评估窗口：跳过初始化阶段（例如落地瞬态）。
        global_mask = time >= float(self.config.evaluation_start_time)
        global_trace = _filter_trace(trace, global_mask)
        perf_mask = None
        if self.config.enable_performance:
            # 性能层可在全局窗口基础上再延迟启动。
            perf_start = float(self.config.evaluation_start_time) + float(
                self.config.performance_start_time
            )
            perf_mask = time >= perf_start
        perf_trace = _filter_trace(trace, perf_mask) if self.config.enable_performance else None

        safety_details: Dict[str, Dict[str, float]] = {}
        stability_details: Dict[str, Dict[str, float]] = {}
        performance_details: Dict[str, Dict[str, float]] = {}
        trajectories: Dict[str, list[float]] = {}
        trajectory_times: Dict[str, np.ndarray] = {}

        layer_robustness: Dict[str, List[float]] = {
            "safety": [],
            "stability": [],
            "performance": [],
        }

        for pred in predicates:
            # 按层级开关过滤谓词。
            if pred.level == "performance" and not self.config.enable_performance:
                continue
            if pred.level == "stability" and not self.config.enable_stability:
                continue
            if pred.level == "safety" and not self.config.enable_safety:
                continue

            if pred.level == "performance" and perf_trace is not None:
                trace_used = perf_trace
            else:
                trace_used = global_trace

            # 逐时刻鲁棒性序列（后续用于 min 聚合和诊断轨迹）。
            series = pred.expr.robustness(trace_used)
            trajectory_times[pred.name] = trace_used.time

            robustness = _safe_min(series)
            ok = bool(np.isfinite(robustness) and robustness >= 0.0)

            trajectories[pred.name] = [float(x) if np.isfinite(x) else float("nan") for x in series]
            detail = {"ok": ok, "robustness": robustness}
            if pred.level == "safety":
                safety_details[pred.name] = detail
            elif pred.level == "stability":
                stability_details[pred.name] = detail
            else:
                performance_details[pred.name] = detail
            layer_robustness[pred.level].append(robustness)

        # 各层鲁棒性采用“层内 min”聚合（G 算子语义）。
        safety_robustness = _safe_min(np.array(layer_robustness["safety"], dtype=np.float32))
        stability_robustness = _safe_min(
            np.array(layer_robustness["stability"], dtype=np.float32)
        )
        performance_robustness = _safe_min(
            np.array(layer_robustness["performance"], dtype=np.float32)
        )

        safety_ok = bool(np.isfinite(safety_robustness) and safety_robustness >= 0.0)
        stability_ok = bool(
            np.isfinite(stability_robustness) and stability_robustness >= 0.0
        )
        performance_ok = bool(
            np.isfinite(performance_robustness) and performance_robustness >= 0.0
        )

        safety_result = LayerResult(
            ok=safety_ok,
            robustness=safety_robustness,
            details=safety_details,
        )
        stability_result = LayerResult(
            ok=stability_ok,
            robustness=stability_robustness,
            details=stability_details,
        )
        performance_result = None
        if self.config.enable_performance:
            performance_result = LayerResult(
                ok=performance_ok,
                robustness=performance_robustness,
                details=performance_details,
            )

        # 全局鲁棒性采用“已启用层的 min”。
        enabled_layers = []
        if self.config.enable_safety:
            enabled_layers.append(safety_robustness)
        if self.config.enable_stability:
            enabled_layers.append(stability_robustness)
        if self.config.enable_performance:
            enabled_layers.append(performance_robustness)
        overall_robustness = _safe_min(np.array(enabled_layers, dtype=np.float32))
        overall_ok = bool(np.isfinite(overall_robustness) and overall_robustness >= 0.0)

        first_violation_time = None
        most_violated_pred = None
        worst_value = float("inf")
        violation_sequence: list[tuple[float, str]] = []

        # 诊断构建：
        # - first_violation_time: 最早违例时刻
        # - most_violated_predicate: 最严重谓词
        # - violation_sequence: 按时间排序的违例序列
        for name, series in trajectories.items():
            values = np.array(series, dtype=np.float32)
            if np.isfinite(values).any():
                min_val = float(np.nanmin(values))
                if min_val < worst_value:
                    worst_value = min_val
                    most_violated_pred = name
                violation_mask = values < 0.0
                if np.any(violation_mask):
                    idx = int(np.where(violation_mask)[0][0])
                    time_series = trajectory_times.get(name, time)
                    if idx < len(time_series):
                        t_val = float(time_series[idx])
                    else:
                        t_val = float("nan")
                    if first_violation_time is None or t_val < first_violation_time:
                        first_violation_time = t_val
                    violation_sequence.append((t_val, name))

        violation_sequence.sort(key=lambda item: item[0])
        diagnostics = Diagnostics(
            first_violation_time=first_violation_time,
            most_violated_predicate=most_violated_pred,
            robustness_trajectories=trajectories,
            violation_sequence=[
                {"time": float(t_val), "predicate": name}
                for t_val, name in violation_sequence
            ],
        )

        return ExtendedSTLResult(
            ok=overall_ok,
            robustness=overall_robustness,
            safety=safety_result,
            stability=stability_result,
            performance=performance_result,
            diagnostics=diagnostics,
        )
