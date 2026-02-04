from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LasCloud:
    center: np.ndarray  # (3,)
    bbox: Tuple[float, float, float, float, float, float]  # x_min,x_max,y_min,y_max,z_min,z_max
    points: np.ndarray  # (N,3) float64


def read_las_points(las_path: Path) -> np.ndarray:
    try:
        import laspy
    except ImportError as e:
        raise ImportError("Missing dependency: laspy. Install via `pip install laspy`") from e

    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def load_las_cloud(las_path: Path) -> LasCloud:
    pts = read_las_points(las_path)
    if pts.size == 0:
        raise ValueError(f"LAS contains no points: {las_path}")
    center = np.mean(pts, axis=0)
    x_min, y_min, z_min = np.min(pts, axis=0)
    x_max, y_max, z_max = np.max(pts, axis=0)
    bbox = (float(x_min), float(x_max), float(y_min), float(y_max), float(z_min), float(z_max))
    return LasCloud(center=center.astype(np.float64), bbox=bbox, points=pts)


def downsample_points_step(points: np.ndarray, step: int) -> np.ndarray:
    step = int(step)
    if step <= 1:
        return points
    return points[::step].copy()


def downsample_points_ratio(points: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    ratio = float(ratio)
    if ratio >= 1.0:
        return points
    if ratio <= 0.0:
        return points[:0].copy()
    rng = np.random.default_rng(int(seed))
    n = len(points)
    k = int(round(n * ratio))
    k = max(0, min(n, k))
    if k == n:
        return points
    idx = rng.choice(n, size=k, replace=False)
    return points[idx].copy()

