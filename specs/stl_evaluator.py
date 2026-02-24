from __future__ import annotations
"""基础 STL 表达式实现。

该文件提供：
1) Trace 数据结构；
2) 原子谓词与逻辑算子（Not/And/Or）；
3) 时序算子（Always/Eventually）；
4) 旧版 walking 规约 `WalkingSTLSpec`（兼容保留）。
"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass
class STLTrace:
    """离散时间 trace。

    要求：
    - `time` 与 `signals[*]` 按索引对齐；
    - 每个信号数组长度一致。
    """

    time: np.ndarray
    signals: Dict[str, np.ndarray]


class STLExpr:
    """STL 表达式抽象基类。"""
    def eval(self, trace: STLTrace) -> np.ndarray:
        raise NotImplementedError

    def robustness(self, trace: STLTrace) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Predicate(STLExpr):
    """原子谓词：`signal op threshold`。"""
    signal: str
    op: str
    threshold: float

    def eval(self, trace: STLTrace) -> np.ndarray:
        """返回逐时刻布尔值。"""
        values = trace.signals[self.signal]
        if self.op == ">":
            return values > self.threshold
        if self.op == ">=":
            return values >= self.threshold
        if self.op == "<":
            return values < self.threshold
        if self.op == "<=":
            return values <= self.threshold
        raise ValueError(f"Unsupported operator: {self.op}")

    def robustness(self, trace: STLTrace) -> np.ndarray:
        """返回逐时刻鲁棒性值。

        定义：
        - `x > c` / `x >= c` 的鲁棒性为 `x-c`
        - `x < c` / `x <= c` 的鲁棒性为 `c-x`
        """
        values = trace.signals[self.signal]
        if self.op in (">", ">="):
            return values - self.threshold
        if self.op in ("<", "<="):
            return self.threshold - values
        raise ValueError(f"Unsupported operator: {self.op}")


@dataclass
class Not(STLExpr):
    """逻辑非。"""
    expr: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return ~self.expr.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return -self.expr.robustness(trace)


@dataclass
class And(STLExpr):
    """逻辑与。"""
    left: STLExpr
    right: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return self.left.eval(trace) & self.right.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return np.minimum(self.left.robustness(trace), self.right.robustness(trace))


@dataclass
class Or(STLExpr):
    """逻辑或。"""
    left: STLExpr
    right: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return self.left.eval(trace) | self.right.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return np.maximum(self.left.robustness(trace), self.right.robustness(trace))


@dataclass
class Always:
    """时序算子 G（全程满足）。"""
    expr: STLExpr

    def eval(self, trace: STLTrace) -> bool:
        return bool(np.all(self.expr.eval(trace)))

    def robustness(self, trace: STLTrace) -> float:
        return float(np.min(self.expr.robustness(trace)))


@dataclass
class Eventually:
    """时序算子 F（最终满足）。"""
    expr: STLExpr

    def eval(self, trace: STLTrace) -> bool:
        return bool(np.any(self.expr.eval(trace)))

    def robustness(self, trace: STLTrace) -> float:
        return float(np.max(self.expr.robustness(trace)))


@dataclass
class STLResult:
    """简版规约输出结构（兼容旧接口）。"""
    ok: bool
    robustness: float
    details: Dict[str, Union[float, bool]]


@dataclass
class WalkingSTLSpec:
    """旧版 walking 规约（仅高度+倾斜）。"""

    height_threshold: float
    max_tilt_deg: float

    def evaluate(self, trace: STLTrace) -> STLResult:
        """执行旧版规约评估。"""
        height_expr = Predicate("height", ">", self.height_threshold)
        tilt_expr = Predicate("tilt", "<", self.max_tilt_deg)

        height_ok = Always(height_expr).eval(trace)
        tilt_ok = Always(tilt_expr).eval(trace)
        height_rob = Always(height_expr).robustness(trace)
        tilt_rob = Always(tilt_expr).robustness(trace)
        robustness = float(min(height_rob, tilt_rob))

        return STLResult(
            ok=bool(height_ok and tilt_ok),
            robustness=robustness,
            details={
                "height_ok": bool(height_ok),
                "tilt_ok": bool(tilt_ok),
                "height_robustness": float(height_rob),
                "tilt_robustness": float(tilt_rob),
            },
        )
