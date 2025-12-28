from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


class Attack(Protocol):
    """Base interface for runtime perturbations."""

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Apply one-time changes after reset."""

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
        step_count: int,
    ) -> None:
        """Apply per-step perturbations before stepping physics."""


@dataclass
class ObservationContext:
    """Metadata for observation perturbations."""

    slices: dict[str, slice]


class ObservationAttack(Protocol):
    """Base interface for observation perturbations."""

    def reset(self, context: ObservationContext) -> None:
        """Optional initialization hook."""

    def apply(
        self,
        obs: np.ndarray,
        context: ObservationContext,
        sim_time: float,
        step_count: int,
    ) -> None:
        """Apply in-place perturbations to the observation vector."""
