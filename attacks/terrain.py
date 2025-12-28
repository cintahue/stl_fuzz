from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"Geom not found: {name}")
    return geom_id


def align_heightfield_baseline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name: str = "terrain",
    baseline: float = 0.5,
    flatten: bool = True,
) -> bool:
    """Align heightfield baseline to z=0 and optionally flatten to baseline."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return False

    hfield_id = int(model.geom_dataid[geom_id])
    if hfield_id < 0:
        return False

    size = model.hfield_size[hfield_id]
    size_z = float(size[2])
    base = float(size[3])
    ground_height = base + baseline * size_z
    model.geom_pos[geom_id][2] = -ground_height

    if flatten:
        adr = int(model.hfield_adr[hfield_id])
        nrow = int(model.hfield_nrow[hfield_id])
        ncol = int(model.hfield_ncol[hfield_id])
        total = nrow * ncol
        model.hfield_data[adr : adr + total] = float(baseline)

    mujoco.mj_forward(model, data)
    return True


@dataclass
class FloorFrictionModifier:
    """地面摩擦扰动：一次性修改地面摩擦系数，用于模拟打滑/粗糙地面。"""
    geom_name: str
    friction: np.ndarray

    def __init__(self, geom_name: str = "terrain", friction: np.ndarray | None = None) -> None:
        self.geom_name = geom_name
        self.friction = (
            np.array(friction, dtype=np.float32)
            if friction is not None
            else np.array([1.0, 0.005, 0.0001], dtype=np.float32)
        )
        self._geom_id: int | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # 直接修改地面geom的摩擦系数，模型结构不变。
        self._geom_id = _geom_id(model, self.geom_name)
        model.geom_friction[self._geom_id] = self.friction
        mujoco.mj_forward(model, data)

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
        step_count: int,
    ) -> None:
        return


@dataclass
class HeightfieldPit:
    """静态挖坑：在heightfield上一次性挖出一个固定的坑。"""
    geom_name: str
    center_xy: np.ndarray
    radius: float
    depth: float
    baseline: float

    def __init__(
        self,
        geom_name: str = "terrain",
        center_xy: np.ndarray | None = None,
        radius: float = 0.15,
        depth: float = 0.03,
        baseline: float = 0.5,
    ) -> None:
        self.geom_name = geom_name
        self.center_xy = (
            np.array(center_xy, dtype=np.float32)
            if center_xy is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        )
        self.radius = float(radius)
        self.depth = float(depth)
        self.baseline = float(baseline)
        self._geom_id: int | None = None
        self._hfield_id: int | None = None
        self._adr: int | None = None
        self._nrow: int | None = None
        self._ncol: int | None = None
        self._size: np.ndarray | None = None
        self._pos_xy: np.ndarray | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # heightfield的高度是归一化值[0,1]，映射到世界高度：
        # height = base + data * size_z，并叠加geom的z偏移。
        self._geom_id = _geom_id(model, self.geom_name)
        self._hfield_id = int(model.geom_dataid[self._geom_id])
        if self._hfield_id < 0:
            raise ValueError(f"Geom is not hfield: {self.geom_name}")

        self._adr = int(model.hfield_adr[self._hfield_id])
        self._nrow = int(model.hfield_nrow[self._hfield_id])
        self._ncol = int(model.hfield_ncol[self._hfield_id])
        self._size = model.hfield_size[self._hfield_id].copy()
        self._pos_xy = model.geom_pos[self._geom_id][:2].copy()

        if self._size is not None:
            size_z = self._size[2]
            base = self._size[3]
            # 对齐地面：保证baseline对应的地面高度在z=0附近。
            ground_height = base + self.baseline * size_z
            model.geom_pos[self._geom_id][2] = -ground_height

        total = self._nrow * self._ncol
        # 先铺平基准地面，再在其上挖坑。
        model.hfield_data[self._adr : self._adr + total] = self.baseline

        self._apply_pit(model)
        mujoco.mj_forward(model, data)

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
        step_count: int,
    ) -> None:
        return

    def _apply_pit(self, model: mujoco.MjModel) -> None:
        if (
            self._adr is None
            or self._nrow is None
            or self._ncol is None
            or self._size is None
            or self._pos_xy is None
        ):
            return

        size_x, size_y, size_z, _ = self._size
        if size_z <= 0:
            return

        center_x, center_y = self.center_xy
        rel_x = (center_x - self._pos_xy[0]) / (2 * size_x) + 0.5
        rel_y = (center_y - self._pos_xy[1]) / (2 * size_y) + 0.5

        col_center = int(np.clip(rel_x * (self._ncol - 1), 0, self._ncol - 1))
        row_center = int(np.clip(rel_y * (self._nrow - 1), 0, self._nrow - 1))

        radius_x = self.radius / (2 * size_x) * (self._ncol - 1)
        radius_y = self.radius / (2 * size_y) * (self._nrow - 1)

        min_row = int(max(0, row_center - radius_y - 1))
        max_row = int(min(self._nrow - 1, row_center + radius_y + 1))
        min_col = int(max(0, col_center - radius_x - 1))
        max_col = int(min(self._ncol - 1, col_center + radius_x + 1))

        data = model.hfield_data
        # 按半径生成一个平滑的“碗状”坑，depth按size_z归一化。
        depth_norm = self.depth / size_z

        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                dr = (r - row_center) / max(radius_y, 1e-6)
                dc = (c - col_center) / max(radius_x, 1e-6)
                dist = np.sqrt(dr * dr + dc * dc)
                if dist > 1.0:
                    continue
                delta = (1.0 - dist) * depth_norm
                idx = self._adr + r * self._ncol + c
                data[idx] = np.clip(self.baseline - delta, 0.0, 1.0)


@dataclass
class HeightfieldBump:
    """静态凸起：在heightfield上一次性抬升一个固定的凸起。"""
    geom_name: str
    center_xy: np.ndarray
    radius: float
    height: float
    baseline: float

    def __init__(
        self,
        geom_name: str = "terrain",
        center_xy: np.ndarray | None = None,
        radius: float = 0.15,
        height: float = 0.03,
        baseline: float = 0.5,
    ) -> None:
        self.geom_name = geom_name
        self.center_xy = (
            np.array(center_xy, dtype=np.float32)
            if center_xy is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        )
        self.radius = float(radius)
        self.height = float(height)
        self.baseline = float(baseline)
        self._geom_id: int | None = None
        self._hfield_id: int | None = None
        self._adr: int | None = None
        self._nrow: int | None = None
        self._ncol: int | None = None
        self._size: np.ndarray | None = None
        self._pos_xy: np.ndarray | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # heightfield的高度是归一化值[0,1]，映射到世界高度：
        # height = base + data * size_z，并叠加geom的z偏移。
        self._geom_id = _geom_id(model, self.geom_name)
        self._hfield_id = int(model.geom_dataid[self._geom_id])
        if self._hfield_id < 0:
            raise ValueError(f"Geom is not hfield: {self.geom_name}")

        self._adr = int(model.hfield_adr[self._hfield_id])
        self._nrow = int(model.hfield_nrow[self._hfield_id])
        self._ncol = int(model.hfield_ncol[self._hfield_id])
        self._size = model.hfield_size[self._hfield_id].copy()
        self._pos_xy = model.geom_pos[self._geom_id][:2].copy()

        if self._size is not None:
            size_z = self._size[2]
            base = self._size[3]
            # 对齐地面：保证baseline对应的地面高度在z=0附近。
            ground_height = base + self.baseline * size_z
            model.geom_pos[self._geom_id][2] = -ground_height

        total = self._nrow * self._ncol
        # 先铺平基准地面，再在其上生成凸起。
        model.hfield_data[self._adr : self._adr + total] = self.baseline

        self._apply_bump(model)
        mujoco.mj_forward(model, data)

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
        step_count: int,
    ) -> None:
        return

    def _apply_bump(self, model: mujoco.MjModel) -> None:
        if (
            self._adr is None
            or self._nrow is None
            or self._ncol is None
            or self._size is None
            or self._pos_xy is None
        ):
            return

        size_x, size_y, size_z, _ = self._size
        if size_z <= 0:
            return

        center_x, center_y = self.center_xy
        rel_x = (center_x - self._pos_xy[0]) / (2 * size_x) + 0.5
        rel_y = (center_y - self._pos_xy[1]) / (2 * size_y) + 0.5

        col_center = int(np.clip(rel_x * (self._ncol - 1), 0, self._ncol - 1))
        row_center = int(np.clip(rel_y * (self._nrow - 1), 0, self._nrow - 1))

        radius_x = self.radius / (2 * size_x) * (self._ncol - 1)
        radius_y = self.radius / (2 * size_y) * (self._nrow - 1)

        min_row = int(max(0, row_center - radius_y - 1))
        max_row = int(min(self._nrow - 1, row_center + radius_y + 1))
        min_col = int(max(0, col_center - radius_x - 1))
        max_col = int(min(self._ncol - 1, col_center + radius_x + 1))

        data = model.hfield_data
        # 按半径生成一个平滑的“圆包”凸起，height按size_z归一化。
        height_norm = self.height / size_z

        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                dr = (r - row_center) / max(radius_y, 1e-6)
                dc = (c - col_center) / max(radius_x, 1e-6)
                dist = np.sqrt(dr * dr + dc * dc)
                if dist > 1.0:
                    continue
                delta = (1.0 - dist) * height_norm
                idx = self._adr + r * self._ncol + c
                data[idx] = np.clip(self.baseline + delta, 0.0, 1.0)

