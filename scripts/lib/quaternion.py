from __future__ import annotations

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate. q is (qw, qx, qy, qz)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q.

    q: (qw, qx, qy, qz)
    v: (3,)
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    # v' = v + 2 * r × (r × v + qw * v), where r=(qx,qy,qz)
    tx = qy * vz - qz * vy + qw * vx
    ty = qz * vx - qx * vz + qw * vy
    tz = qx * vy - qy * vx + qw * vz
    return np.array(
        [
            vx + 2.0 * (qy * tz - qz * ty),
            vy + 2.0 * (qz * tx - qx * tz),
            vz + 2.0 * (qx * ty - qy * tx),
        ],
        dtype=np.float64,
    )

