from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SpecConfig:
    h_min: float = 0.25
    max_tilt_deg: float = 45.0
    max_torque: float = 50.0
    zmp_margin: float = 0.02
    stability_margin: float = 0.02
    max_angular_velocity: float = 3.0
    max_velocity_error: float = 0.3
    min_foot_clearance: float = 0.02
    max_action_delta: float = 0.2

    enable_safety: bool = True
    enable_stability: bool = True
    enable_performance: bool = False
    use_zmp: bool = True
    evaluation_start_time: float = 1.0
    performance_start_time: float = 1.0

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "SpecConfig":
        if not raw:
            return cls()
        data = dict(raw)
        return cls(
            h_min=float(data.get("h_min", cls.h_min)),
            max_tilt_deg=float(data.get("max_tilt_deg", cls.max_tilt_deg)),
            max_torque=float(data.get("max_torque", cls.max_torque)),
            zmp_margin=float(data.get("zmp_margin", cls.zmp_margin)),
            stability_margin=float(data.get("stability_margin", cls.stability_margin)),
            max_angular_velocity=float(
                data.get("max_angular_velocity", cls.max_angular_velocity)
            ),
            max_velocity_error=float(
                data.get("max_velocity_error", cls.max_velocity_error)
            ),
            min_foot_clearance=float(
                data.get("min_foot_clearance", cls.min_foot_clearance)
            ),
            max_action_delta=float(data.get("max_action_delta", cls.max_action_delta)),
            enable_safety=bool(data.get("enable_safety", cls.enable_safety)),
            enable_stability=bool(data.get("enable_stability", cls.enable_stability)),
            enable_performance=bool(
                data.get("enable_performance", cls.enable_performance)
            ),
            use_zmp=bool(data.get("use_zmp", cls.use_zmp)),
            evaluation_start_time=float(
                data.get("evaluation_start_time", cls.evaluation_start_time)
            ),
            performance_start_time=float(
                data.get("performance_start_time", cls.performance_start_time)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "h_min": self.h_min,
            "max_tilt_deg": self.max_tilt_deg,
            "max_torque": self.max_torque,
            "zmp_margin": self.zmp_margin,
            "stability_margin": self.stability_margin,
            "max_angular_velocity": self.max_angular_velocity,
            "max_velocity_error": self.max_velocity_error,
            "min_foot_clearance": self.min_foot_clearance,
            "max_action_delta": self.max_action_delta,
            "enable_safety": self.enable_safety,
            "enable_stability": self.enable_stability,
            "enable_performance": self.enable_performance,
            "use_zmp": self.use_zmp,
            "evaluation_start_time": self.evaluation_start_time,
            "performance_start_time": self.performance_start_time,
        }
