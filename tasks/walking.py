from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from robostl.core.config import DeployConfig


@dataclass
class WalkingTask:
    command: np.ndarray

    @classmethod
    def from_config(cls, config: DeployConfig) -> "WalkingTask":
        return cls(command=config.cmd_init.copy())
