from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from robostl.specs.stl_evaluator import Predicate


@dataclass
class PredicateSpec:
    name: str
    expr: Predicate
    level: str
    signal: str
    op: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "signal": self.signal,
            "op": self.op,
            "threshold": self.threshold,
        }


def greater_than(name: str, signal: str, threshold: float, level: str) -> PredicateSpec:
    return PredicateSpec(
        name=name,
        expr=Predicate(signal=signal, op=">", threshold=threshold),
        level=level,
        signal=signal,
        op=">",
        threshold=threshold,
    )


def less_than(name: str, signal: str, threshold: float, level: str) -> PredicateSpec:
    return PredicateSpec(
        name=name,
        expr=Predicate(signal=signal, op="<", threshold=threshold),
        level=level,
        signal=signal,
        op="<",
        threshold=threshold,
    )
