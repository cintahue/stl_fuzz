from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass
class STLTrace:
    """Discrete-time STL trace with aligned timestamps."""

    time: np.ndarray
    signals: Dict[str, np.ndarray]


class STLExpr:
    def eval(self, trace: STLTrace) -> np.ndarray:
        raise NotImplementedError

    def robustness(self, trace: STLTrace) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Predicate(STLExpr):
    signal: str
    op: str
    threshold: float

    def eval(self, trace: STLTrace) -> np.ndarray:
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
        values = trace.signals[self.signal]
        if self.op in (">", ">="):
            return values - self.threshold
        if self.op in ("<", "<="):
            return self.threshold - values
        raise ValueError(f"Unsupported operator: {self.op}")


@dataclass
class Not(STLExpr):
    expr: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return ~self.expr.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return -self.expr.robustness(trace)


@dataclass
class And(STLExpr):
    left: STLExpr
    right: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return self.left.eval(trace) & self.right.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return np.minimum(self.left.robustness(trace), self.right.robustness(trace))


@dataclass
class Or(STLExpr):
    left: STLExpr
    right: STLExpr

    def eval(self, trace: STLTrace) -> np.ndarray:
        return self.left.eval(trace) | self.right.eval(trace)

    def robustness(self, trace: STLTrace) -> np.ndarray:
        return np.maximum(self.left.robustness(trace), self.right.robustness(trace))


@dataclass
class Always:
    expr: STLExpr

    def eval(self, trace: STLTrace) -> bool:
        return bool(np.all(self.expr.eval(trace)))

    def robustness(self, trace: STLTrace) -> float:
        return float(np.min(self.expr.robustness(trace)))


@dataclass
class Eventually:
    expr: STLExpr

    def eval(self, trace: STLTrace) -> bool:
        return bool(np.any(self.expr.eval(trace)))

    def robustness(self, trace: STLTrace) -> float:
        return float(np.max(self.expr.robustness(trace)))


@dataclass
class STLResult:
    ok: bool
    robustness: float
    details: Dict[str, Union[float, bool]]


@dataclass
class WalkingSTLSpec:
    """Default walking STL spec: always height > threshold and tilt < max."""

    height_threshold: float
    max_tilt_deg: float

    def evaluate(self, trace: STLTrace) -> STLResult:
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
