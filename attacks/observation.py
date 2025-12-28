from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robostl.attacks.base import ObservationContext


@dataclass
class GaussianObservationNoise:
    targets: tuple[str, ...]
    std: float

    def __init__(self, std: float = 0.01, targets: tuple[str, ...] | None = None) -> None:
        self.std = float(std)
        self.targets = targets or ("omega", "gravity", "qj", "dqj")

    def reset(self, context: ObservationContext) -> None:
        return

    def apply(
        self,
        obs: np.ndarray,
        context: ObservationContext,
        sim_time: float,
        step_count: int,
    ) -> None:
        if self.std <= 0:
            return
        for key in self.targets:
            sl = context.slices.get(key)
            if sl is None:
                continue
            obs[sl] = obs[sl] + np.random.normal(0.0, self.std, obs[sl].shape)
