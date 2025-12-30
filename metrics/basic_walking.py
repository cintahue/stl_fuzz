from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from robostl.core.state import SimState
from robostl.specs.stl_evaluator import STLTrace
from robostl.metrics.stability import (
    calculate_zmp_margin,
    compute_foot_clearance,
    compute_zmp_from_contacts,
    get_support_polygon,
)


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
    min_zmp_margin: float = field(default_factory=lambda: float("inf"))
    max_torque: float = 0.0
    max_angular_velocity: float = 0.0
    max_velocity_error: float = 0.0
    min_foot_clearance: float = field(default_factory=lambda: float("inf"))
    max_action_delta: float = 0.0
    fallen: bool = False
    fall_time: Optional[float] = None
    time_series: list[float] = field(default_factory=list)
    height_series: list[float] = field(default_factory=list)
    tilt_series: list[float] = field(default_factory=list)
    stability_series: list[float] = field(default_factory=list)
    zmp_margin_series: list[float] = field(default_factory=list)
    max_torque_series: list[float] = field(default_factory=list)
    angular_velocity_series: list[float] = field(default_factory=list)
    velocity_error_series: list[float] = field(default_factory=list)
    foot_clearance_series: list[float] = field(default_factory=list)
    action_delta_series: list[float] = field(default_factory=list)

    def reset(self, state: SimState) -> None:
        self.start_state = state
        self.steps = 0
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.min_stability_margin = float("inf")
        self.min_zmp_margin = float("inf")
        self.max_torque = 0.0
        self.max_angular_velocity = 0.0
        self.max_velocity_error = 0.0
        self.min_foot_clearance = float("inf")
        self.max_action_delta = 0.0
        self.fallen = False
        self.fall_time = None
        self.time_series = []
        self.height_series = []
        self.tilt_series = []
        self.stability_series = []
        self.zmp_margin_series = []
        self.max_torque_series = []
        self.angular_velocity_series = []
        self.velocity_error_series = []
        self.foot_clearance_series = []
        self.action_delta_series = []

    def update(
        self,
        state: SimState,
        model,
        data,
        ctrl: np.ndarray,
        cmd: np.ndarray,
        prev_action: Optional[np.ndarray],
        ground_geom_ids: Optional[list[int]] = None,
    ) -> None:
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

        zmp_margin = float("nan")
        if model is not None and data is not None and ground_geom_ids:
            support_poly = get_support_polygon(model, data, ground_geom_ids)
            if support_poly:
                zmp = compute_zmp_from_contacts(model, data, ground_geom_ids)
                zmp_margin = calculate_zmp_margin(zmp, support_poly)
        if math.isfinite(zmp_margin):
            self.min_zmp_margin = min(self.min_zmp_margin, zmp_margin)
        self.zmp_margin_series.append(float(zmp_margin))

        max_torque = float(np.max(np.abs(ctrl))) if ctrl.size else float("nan")
        if math.isfinite(max_torque):
            self.max_torque = max(self.max_torque, max_torque)
        self.max_torque_series.append(float(max_torque))

        angular_velocity = float(np.linalg.norm(state.qvel[3:6]))
        self.max_angular_velocity = max(self.max_angular_velocity, angular_velocity)
        self.angular_velocity_series.append(float(angular_velocity))

        velocity_error = float(np.linalg.norm(state.qvel[:2] - cmd[:2]))
        self.max_velocity_error = max(self.max_velocity_error, velocity_error)
        self.velocity_error_series.append(float(velocity_error))

        foot_clearance = float("nan")
        if model is not None and data is not None:
            foot_clearance = compute_foot_clearance(model, data, ground_geom_ids)
        if math.isfinite(foot_clearance):
            self.min_foot_clearance = min(self.min_foot_clearance, foot_clearance)
        self.foot_clearance_series.append(float(foot_clearance))

        if prev_action is None:
            action_delta = float("nan")
        else:
            action_delta = float(np.linalg.norm(state.action - prev_action))
            self.max_action_delta = max(self.max_action_delta, action_delta)
        self.action_delta_series.append(float(action_delta))

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
            "min_zmp_margin": self.min_zmp_margin if math.isfinite(self.min_zmp_margin) else None,
            "max_torque": self.max_torque if math.isfinite(self.max_torque) else None,
            "max_angular_velocity": self.max_angular_velocity,
            "max_velocity_error": self.max_velocity_error,
            "min_foot_clearance": self.min_foot_clearance
            if math.isfinite(self.min_foot_clearance)
            else None,
            "max_action_delta": self.max_action_delta if math.isfinite(self.max_action_delta) else None,
            "fallen": self.fallen,
            "fall_time_s": self.fall_time,
        }
        if self.stability_series:
            stability_array = np.array(self.stability_series, dtype=np.float32)
            if np.isfinite(stability_array).any():
                metrics["min_stability_margin"] = float(np.nanmin(stability_array))
                metrics["mean_stability_margin"] = float(np.nanmean(stability_array))
        if self.zmp_margin_series:
            zmp_array = np.array(self.zmp_margin_series, dtype=np.float32)
            if np.isfinite(zmp_array).any():
                metrics["mean_zmp_margin"] = float(np.nanmean(zmp_array))
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
                "zmp_margin": np.array(self.zmp_margin_series, dtype=np.float32),
                "max_torque": np.array(self.max_torque_series, dtype=np.float32),
                "angular_velocity": np.array(self.angular_velocity_series, dtype=np.float32),
                "velocity_error": np.array(self.velocity_error_series, dtype=np.float32),
                "foot_clearance": np.array(self.foot_clearance_series, dtype=np.float32),
                "action_delta": np.array(self.action_delta_series, dtype=np.float32),
            },
        )

    def _is_fall(self, height: float, tilt: float) -> bool:
        if self.start_state is None:
            return False
        ref_height = float(self.start_state.base_pos[2])
        height_threshold = max(self.min_height_abs, ref_height * self.min_height_ratio)
        return height < height_threshold or tilt > self.max_tilt_deg
