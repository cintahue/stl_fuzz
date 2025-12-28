from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SimState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    cmd: np.ndarray
    stability_margin: Optional[float] = None

    @property
    def base_pos(self) -> np.ndarray:
        return self.qpos[:3]

    @property
    def base_quat(self) -> np.ndarray:
        return self.qpos[3:7]

    @property
    def joint_pos(self) -> np.ndarray:
        return self.qpos[7:]

    @property
    def joint_vel(self) -> np.ndarray:
        return self.qvel[6:]
