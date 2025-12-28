from __future__ import annotations

from typing import Iterable, Sequence

import mujoco
import numpy as np


def resolve_ground_geom_ids(
    model: mujoco.MjModel, ground_geom_names: Sequence[str]
) -> list[int]:
    ids: list[int] = []
    for name in ground_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            ids.append(int(geom_id))
    return ids


def get_support_polygon(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: Iterable[int],
) -> list[list[float]]:
    """Collect contact points between the ground and foot-related geoms."""
    contact_points: list[np.ndarray] = []
    ground_ids = set(int(gid) for gid in ground_geom_ids)
    if not ground_ids:
        return []

    contact_keywords = ("foot", "ankle", "toe", "sole", "heel")

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 < 0 or geom2 < 0:
            continue

        candidate_geom = None
        if geom1 in ground_ids:
            candidate_geom = geom2
        elif geom2 in ground_ids:
            candidate_geom = geom1
        else:
            continue

        body_id = int(model.geom_bodyid[candidate_geom])
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, candidate_geom) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        names = (geom_name.lower(), body_name.lower())
        if not any(keyword in name for name in names for keyword in contact_keywords):
            continue

        contact_points.append(contact.pos[:2].copy())

    if not contact_points:
        return []
    return [point.tolist() for point in contact_points]


def _convex_hull(points: Sequence[Sequence[float]]) -> list[np.ndarray]:
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 1:
        return [np.array(p, dtype=np.float32) for p in unique]

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [np.array(p, dtype=np.float32) for p in hull]


def _point_in_convex_polygon(point: np.ndarray, polygon: Sequence[np.ndarray]) -> bool:
    if len(polygon) < 3:
        return False

    sign = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
        if cross == 0:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


def _distance_point_to_segment(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    line_vec = p2 - p1
    point_vec = point - p1
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq == 0.0:
        return float(np.linalg.norm(point_vec))
    t = float(np.clip(np.dot(point_vec, line_vec) / line_len_sq, 0.0, 1.0))
    closest = p1 + t * line_vec
    return float(np.linalg.norm(point - closest))


def calculate_stability_margin(
    com_proj: Sequence[float],
    support_poly_points: Sequence[Sequence[float]],
) -> float:
    """Return the minimum distance from COM projection to the support polygon boundary."""
    if len(support_poly_points) == 0:
        return 0.0
    if len(support_poly_points) == 1:
        return float(np.linalg.norm(np.array(com_proj) - np.array(support_poly_points[0])))

    if len(support_poly_points) == 2:
        p1 = np.array(support_poly_points[0], dtype=np.float32)
        p2 = np.array(support_poly_points[1], dtype=np.float32)
        return _distance_point_to_segment(np.array(com_proj, dtype=np.float32), p1, p2)

    hull = _convex_hull(support_poly_points)
    com = np.array(com_proj, dtype=np.float32)
    if len(hull) < 3:
        return _distance_point_to_segment(com, hull[0], hull[-1])
    if not _point_in_convex_polygon(com, hull):
        return 0.0

    min_dist = float("inf")
    for i in range(len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1) % len(hull)]
        min_dist = min(min_dist, _distance_point_to_segment(com, p1, p2))

    return float(min_dist) if np.isfinite(min_dist) else 0.0
