from __future__ import annotations
"""扩展 STL 评估结果结构。

结果分为三层：
1) 全局结果：`ok`, `robustness`
2) 分层结果：`safety/stability/performance`
3) 诊断信息：首次违例时刻、最严重谓词、轨迹等
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LayerResult:
    """单层规约结果。"""
    ok: bool
    robustness: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "ok": self.ok,
            "robustness": self.robustness,
            "details": self.details,
        }


@dataclass
class Diagnostics:
    """诊断信息集合。"""
    first_violation_time: Optional[float]
    most_violated_predicate: Optional[str]
    robustness_trajectories: Dict[str, list[float]]
    violation_sequence: list[dict]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "first_violation_time": self.first_violation_time,
            "most_violated_predicate": self.most_violated_predicate,
            "robustness_trajectories": self.robustness_trajectories,
            "violation_sequence": self.violation_sequence,
        }


@dataclass
class ExtendedSTLResult:
    """完整 STL 评估结果（供 runner/fuzz/analysis 统一消费）。"""
    ok: bool
    robustness: float
    safety: LayerResult
    stability: LayerResult
    performance: Optional[LayerResult]
    diagnostics: Diagnostics

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "ok": self.ok,
            "robustness": self.robustness,
            "safety": self.safety.to_dict(),
            "stability": self.stability.to_dict(),
            "performance": self.performance.to_dict() if self.performance else None,
            "diagnostics": self.diagnostics.to_dict(),
        }
