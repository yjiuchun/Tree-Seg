from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .quaternion import quat_conj, quat_rotate_vector


def world_point_to_uv_equirect(
    point_xyz: np.ndarray,
    cam_xyz: np.ndarray,
    q_c2w: np.ndarray,
    width: int,
    height: int,
    *,
    flip_v: bool = True,
    require_in_front: bool = True,
) -> Optional[Tuple[int, int]]:
    """
    Project a world point to equirectangular image pixel coordinates.

    Coordinate convention:
    - Camera frame: +Z forward, +X right, +Y up.
    - q_c2w is camera-to-world quaternion (qw,qx,qy,qz).

    Returns (u,v) as int pixels, or None if invalid / behind camera.
    """
    point_xyz = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
    q_c2w = np.asarray(q_c2w, dtype=np.float64).reshape(4)

    d_world = point_xyz - cam_xyz
    n = float(np.linalg.norm(d_world))
    if n < 1e-12:
        return None
    d_world /= n

    q_w2c = quat_conj(q_c2w)
    d_cam = quat_rotate_vector(q_w2c, d_world)
    if require_in_front and float(d_cam[2]) <= 0.0:
        return None

    dx, dy, dz = float(d_cam[0]), float(d_cam[1]), float(d_cam[2])
    theta = np.arctan2(dx, dz)  # [-pi, pi]
    phi = np.arcsin(np.clip(dy, -1.0, 1.0))  # [-pi/2, pi/2]

    u = (theta + np.pi) / (2.0 * np.pi) * float(width)
    v = (np.pi / 2.0 - phi) / np.pi * float(height)

    u_i = int(round(float(np.clip(u, 0.0, float(width - 1)))))
    v_i = int(round(float(np.clip(v, 0.0, float(height - 1)))))
    if flip_v:
        v_i = int(height - 1 - v_i)
    return (u_i, v_i)


def world_dir_to_cam_dir(
    d_world: np.ndarray,
    q_c2w: np.ndarray,
) -> np.ndarray:
    """Rotate unit direction from world into camera frame."""
    d_world = np.asarray(d_world, dtype=np.float64).reshape(3)
    q_c2w = np.asarray(q_c2w, dtype=np.float64).reshape(4)
    q_w2c = quat_conj(q_c2w)
    return quat_rotate_vector(q_w2c, d_world)

