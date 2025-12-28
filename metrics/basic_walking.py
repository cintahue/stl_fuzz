from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from robostl.core.state import SimState
from robostl.specs.stl_evaluator import STLTrace


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def tilt_angle_deg(quat: np.ndarray) -> float:
    rot = quat_to_matrix(quat)
    z_axis = rot[:, 2]
    alignment = float(np.clip(z_axis.dot(np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    return float(np.degrees(np.arccos(alignment)))


@dataclass
class WalkingMetrics:
    min_height_ratio: float = 0.5
    min_height_abs: float = 0.25
    max_tilt_deg: float = 45.0

    start_state: Optional[SimState] = None
    steps: int = 0
    min_height: float = field(default_factory=lambda: float("inf"))
    max_tilt: float = 0.0
    min_stability_margin: float = field(default_factory=lambda: float("inf"))
    fallen: bool = False
    fall_time: Optional[float] = None
    time_series: list[float] = field(default_factory=list)
    height_series: list[float] = field(default_factory=list)
    tilt_series: list[float] = field(default_factory=list)
    stability_series: list[float] = field(default_factory=list)

    def reset(self, state: SimState) -> None:
        self.start_state = state
        self.steps = 0
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.min_stability_margin = float("inf")
        self.fallen = False
        self.fall_time = None
        self.time_series = []
        self.height_series = []
        self.tilt_series = []
        self.stability_series = []

    def update(self, state: SimState) -> None:
        self.steps += 1
        height = float(state.base_pos[2])
        self.min_height = min(self.min_height, height)

        tilt = tilt_angle_deg(state.base_quat)
        self.max_tilt = max(self.max_tilt, tilt)
        self.time_series.append(state.time)
        self.height_series.append(height)
        self.tilt_series.append(tilt)

        stability_margin = state.stability_margin
        if stability_margin is None or not math.isfinite(stability_margin):
            stability_margin = float("nan")
        else:
            self.min_stability_margin = min(self.min_stability_margin, stability_margin)
        self.stability_series.append(float(stability_margin))

        if not self.fallen and self._is_fall(height, tilt):
            self.fallen = True
            self.fall_time = state.time

    def finalize(self, state: SimState) -> dict:
        if self.start_state is None:
            raise RuntimeError("Metrics not reset before finalize.")

        duration = state.time - self.start_state.time
        distance = float(state.base_pos[0] - self.start_state.base_pos[0])
        mean_speed = distance / duration if duration > 0 else 0.0

        metrics = {
            "steps": self.steps,
            "duration_s": duration,
            "distance_x_m": distance,
            "mean_speed_mps": mean_speed,
            "min_height_m": self.min_height,
            "max_tilt_deg": self.max_tilt,
            "fallen": self.fallen,
            "fall_time_s": self.fall_time,
        }
        if self.stability_series:
            stability_array = np.array(self.stability_series, dtype=np.float32)
            if np.isfinite(stability_array).any():
                metrics["min_stability_margin"] = float(np.nanmin(stability_array))
                metrics["mean_stability_margin"] = float(np.nanmean(stability_array))
        return metrics

    def build_trace(self) -> STLTrace:
        if not self.time_series:
            raise RuntimeError("No time series recorded for STL evaluation.")
        return STLTrace(
            time=np.array(self.time_series, dtype=np.float32),
            signals={
                "height": np.array(self.height_series, dtype=np.float32),
                "tilt": np.array(self.tilt_series, dtype=np.float32),
                "stability_margin": np.array(self.stability_series, dtype=np.float32),
            },
        )

    def _is_fall(self, height: float, tilt: float) -> bool:
        if self.start_state is None:
            return False
        ref_height = float(self.start_state.base_pos[2])
        height_threshold = max(self.min_height_abs, ref_height * self.min_height_ratio)
        return height < height_threshold or tilt > self.max_tilt_deg
