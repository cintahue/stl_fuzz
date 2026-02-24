from __future__ import annotations
"""谓词工厂与谓词元数据定义。

本文件将 STL 原子谓词包装为 `PredicateSpec`，便于：
1) 统一记录谓词所属层级（safety/stability/performance）；
2) 在结果里保留信号名、操作符、阈值等诊断信息。
"""

from dataclasses import dataclass
from typing import Any, Dict

from robostl.specs.stl_evaluator import Predicate


@dataclass
class PredicateSpec:
    """带元信息的谓词对象。

    `expr` 是可执行的 `Predicate`；
    其余字段用于日志、可视化、问题定位。
    """
    name: str
    expr: Predicate
    level: str
    signal: str
    op: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """导出轻量描述，便于写入结果文件。"""
        return {
            "name": self.name,
            "level": self.level,
            "signal": self.signal,
            "op": self.op,
            "threshold": self.threshold,
        }


def greater_than(name: str, signal: str, threshold: float, level: str) -> PredicateSpec:
    """构造 `signal > threshold` 类型谓词。"""
    return PredicateSpec(
        name=name,
        expr=Predicate(signal=signal, op=">", threshold=threshold),
        level=level,
        signal=signal,
        op=">",
        threshold=threshold,
    )


def less_than(name: str, signal: str, threshold: float, level: str) -> PredicateSpec:
    """构造 `signal < threshold` 类型谓词。"""
    return PredicateSpec(
        name=name,
        expr=Predicate(signal=signal, op="<", threshold=threshold),
        level=level,
        signal=signal,
        op="<",
        threshold=threshold,
    )
