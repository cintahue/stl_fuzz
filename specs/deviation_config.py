"""Deviation STL 配置 — 与 cmd 无关的偏差阈值。

影子仿真模式下，评估标准从"绝对值是否超阈值"变为
"攻击仿真 与 零扰动影子仿真 之间的偏差是否超阈值"。
这使得阈值不再依赖速度指令 cmd，可以静态预设。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DeviationConfig:
    """偏差型 STL 评估配置。

    字段分两类：
    - max_delta_* : 偏差约束，衡量 attack - shadow 的差值绝对值上限
    - absolute_*  : 绝对硬约束（无论偏差如何都不可超过的极限值）
    """

    # 偏差约束
    max_delta_height: float = 0.08             # 高度偏差 > 8 cm 视为异常
    max_delta_tilt: float = 8.0                # 倾角偏差 > 8° 视为异常
    max_delta_angular_velocity: float = 1.0    # 角速度偏差上限 (rad/s)
    max_delta_velocity_error: float = 0.3      # 速度误差偏差上限 (m/s)

    # 绝对硬约束（保留，不走偏差逻辑）
    absolute_max_tilt: float = 70.0            # 无论如何不可超过 70°（倒地判定）
    absolute_min_height: float = 0.25          # 无论如何不可低于 0.25 m

    # 评估窗口
    evaluation_start_time: float = 1.0         # 跳过初始化阶段

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeviationConfig":
        """从 YAML 字典构造配置对象。"""
        if not d:
            return cls()
        return cls(
            max_delta_height=float(d.get("max_delta_height", 0.08)),
            max_delta_tilt=float(d.get("max_delta_tilt", 8.0)),
            max_delta_angular_velocity=float(d.get("max_delta_angular_velocity", 1.0)),
            max_delta_velocity_error=float(d.get("max_delta_velocity_error", 0.3)),
            absolute_max_tilt=float(d.get("absolute_max_tilt", 70.0)),
            absolute_min_height=float(d.get("absolute_min_height", 0.25)),
            evaluation_start_time=float(d.get("evaluation_start_time", 1.0)),
        )
