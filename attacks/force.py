from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ForcePerturbation:
    body_name: str
    force: np.ndarray
    torque: np.ndarray
    start_time: float
    duration: float

    def __init__(
        self,
        body_name: str = "pelvis",
        force: np.ndarray | None = None,
        torque: np.ndarray | None = None,
        start_time: float = 1.0,
        duration: float = 0.2,
    ) -> None:
        self.body_name = body_name
        self.force = (
            np.array(force, dtype=np.float32)
            if force is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.torque = (
            np.array(torque, dtype=np.float32)
            if torque is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.start_time = float(start_time)
        self.duration = float(duration)
        self._body_id: int | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.body_name
        )
        if self._body_id < 0:
            raise ValueError(f"Body not found: {self.body_name}")
        data.xfrc_applied[self._body_id, :] = 0.0

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
        step_count: int,
    ) -> None:
        if self._body_id is None:
            return
        active = self.start_time <= sim_time <= self.start_time + self.duration
        if active:
            data.xfrc_applied[self._body_id, :3] = self.force
            data.xfrc_applied[self._body_id, 3:] = self.torque
        else:
            data.xfrc_applied[self._body_id, :] = 0.0
