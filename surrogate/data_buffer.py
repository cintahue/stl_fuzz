"""DataBuffer: stores (x_l1, phase, x_l3, rho) training data for the surrogate model.

Layout of the 36-dim full feature vector:
  [0:11]   x_l1  - L1 encoding (push 3D + friction 3D + terrain 5D)
  [11]     phase  - attack phase in [0, 1]
  [12:36]  x_l3  - L3 state perturbation (12 joint pos + 12 joint vel offsets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DataPoint:
    x_l1: np.ndarray   # 11D L1 encoding
    phase: float        # attack phase in [0, 1]
    x_l3: np.ndarray   # 24D L3 state perturbation
    rho: float          # STL robustness (negative = violation)
    metadata: dict = field(default_factory=dict)


class DataBuffer:
    """Ring buffer for surrogate model training data."""

    L1_DIM = 11
    L3_DIM = 24
    FULL_DIM = 36  # L1_DIM + 1 (phase) + L3_DIM

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.entries: list[DataPoint] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(
        self,
        x_l1: np.ndarray,
        phase: float,
        x_l3: np.ndarray,
        rho: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a data point; pads x_l3 to L3_DIM if shorter."""
        x_l1 = np.asarray(x_l1, dtype=np.float32)
        x_l3 = np.asarray(x_l3, dtype=np.float32)

        if x_l3.shape[0] < self.L3_DIM:
            padded = np.zeros(self.L3_DIM, dtype=np.float32)
            padded[: x_l3.shape[0]] = x_l3
            x_l3 = padded
        else:
            x_l3 = x_l3[: self.L3_DIM]

        self.entries.append(
            DataPoint(
                x_l1=x_l1,
                phase=float(phase),
                x_l3=x_l3,
                rho=float(rho),
                metadata=metadata or {},
            )
        )
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size :]

    def get_training_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (X_l1 [N,11], X_full [N,36], y_rho [N])."""
        if not self.entries:
            return (
                np.zeros((0, self.L1_DIM), dtype=np.float32),
                np.zeros((0, self.FULL_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        X_l1 = np.stack([e.x_l1 for e in self.entries])
        X_full = np.stack(
            [
                np.concatenate([e.x_l1, [e.phase], e.x_l3])
                for e in self.entries
            ]
        )
        y = np.array([e.rho for e in self.entries], dtype=np.float32)
        return X_l1, X_full, y

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        X_l1, X_full, y = self.get_training_data()
        np.savez_compressed(path, X_l1=X_l1, X_full=X_full, y=y)
        print(f"[DataBuffer] Saved {self.size} entries to {path}")

    @classmethod
    def load_or_create(cls, path: Path, max_size: int = 10000) -> "DataBuffer":
        path = Path(path)
        buf = cls(max_size=max_size)
        if path.exists():
            try:
                data = np.load(path)
                X_full = data["X_full"]
                y = data["y"]
                for i in range(len(y)):
                    xf = X_full[i]
                    x_l1 = xf[: cls.L1_DIM]
                    phase = float(xf[cls.L1_DIM])
                    x_l3 = xf[cls.L1_DIM + 1 :]
                    buf.entries.append(
                        DataPoint(
                            x_l1=x_l1.astype(np.float32),
                            phase=phase,
                            x_l3=x_l3.astype(np.float32),
                            rho=float(y[i]),
                        )
                    )
                print(f"[DataBuffer] Loaded {buf.size} entries from {path}")
            except Exception as exc:
                print(f"[DataBuffer] Load failed ({exc}); starting fresh.")
        return buf


def encode_l1(
    push: np.ndarray,
    friction: np.ndarray,
    terrain: dict,
) -> np.ndarray:
    """Encode L1 parameters into an 11-dim feature vector.

    Layout:
      [0:3]   push force [fx, fy, fz]
      [3:6]   friction [mu_slide, mu_spin, mu_roll]
      [6:9]   terrain encoding:
                flat  → [0, 0, 0]
                pit   → [depth, radius, 0]
                bump  → [0, 0, height]
      [9:11]  terrain center [cx, cy]  (zeros for flat)
    """
    mode = terrain.get("mode", "flat")
    if mode == "pit":
        depth = float(terrain.get("depth", 0.0))
        radius = float(terrain.get("radius", 0.1))
        terrain_enc = np.array([depth, radius, 0.0], dtype=np.float32)
        c = terrain.get("center", [0.0, 0.0])
        center = np.array([float(c[0]), float(c[1])], dtype=np.float32)
    elif mode == "bump":
        height = float(terrain.get("height", 0.0))
        radius = float(terrain.get("radius", 0.1))
        terrain_enc = np.array([0.0, 0.0, height], dtype=np.float32)
        c = terrain.get("center", [0.0, 0.0])
        center = np.array([float(c[0]), float(c[1])], dtype=np.float32)
    else:  # flat
        terrain_enc = np.zeros(3, dtype=np.float32)
        center = np.zeros(2, dtype=np.float32)

    return np.concatenate(
        [
            np.asarray(push, dtype=np.float32),
            np.asarray(friction, dtype=np.float32),
            terrain_enc,
            center,
        ]
    )
