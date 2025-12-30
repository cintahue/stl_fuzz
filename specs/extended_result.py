from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LayerResult:
    ok: bool
    robustness: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "robustness": self.robustness,
            "details": self.details,
        }


@dataclass
class Diagnostics:
    first_violation_time: Optional[float]
    most_violated_predicate: Optional[str]
    robustness_trajectories: Dict[str, list[float]]
    violation_sequence: list[dict]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_violation_time": self.first_violation_time,
            "most_violated_predicate": self.most_violated_predicate,
            "robustness_trajectories": self.robustness_trajectories,
            "violation_sequence": self.violation_sequence,
        }


@dataclass
class ExtendedSTLResult:
    ok: bool
    robustness: float
    safety: LayerResult
    stability: LayerResult
    performance: Optional[LayerResult]
    diagnostics: Diagnostics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "robustness": self.robustness,
            "safety": self.safety.to_dict(),
            "stability": self.stability.to_dict(),
            "performance": self.performance.to_dict() if self.performance else None,
            "diagnostics": self.diagnostics.to_dict(),
        }
