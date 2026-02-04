from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MorphParams:
    kernel: int = 9
    dilate_iter: int = 2
    close_iter: int = 2


def points_to_mask(
    uv_points: np.ndarray,
    height: int,
    width: int,
    *,
    morph: MorphParams = MorphParams(),
) -> np.ndarray:
    """
    Build a binary mask from uv points (N,2) int pixels.
    Returns uint8 mask with values {0,255}.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    if uv_points is None or len(uv_points) == 0:
        return mask

    pts = np.asarray(uv_points, dtype=np.int32).reshape(-1, 2)
    u = np.clip(pts[:, 0], 0, width - 1)
    v = np.clip(pts[:, 1], 0, height - 1)
    mask[v, u] = 255

    k = int(max(1, morph.kernel))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    di = int(max(0, morph.dilate_iter))
    ci = int(max(0, morph.close_iter))
    if di > 0:
        mask = cv2.dilate(mask, kernel, iterations=di)
    if ci > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ci)
    return mask


RefineMode = Literal["largest_contour", "convex_hull", "approx"]


def refine_mask(
    mask: np.ndarray,
    *,
    mode: RefineMode = "largest_contour",
    approx_epsilon_ratio: float = 0.01,
) -> np.ndarray:
    """
    Convert a (possibly noisy) binary mask into a single smooth-ish region mask.
    Returns uint8 mask {0,255}.
    """
    import cv2

    if mask is None:
        raise ValueError("mask is None")
    m = (mask > 0).astype(np.uint8) * 255
    if m.max() == 0:
        return m

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)
    # pick largest area
    c = max(contours, key=cv2.contourArea)
    if mode == "convex_hull":
        c2 = cv2.convexHull(c)
    elif mode == "approx":
        peri = cv2.arcLength(c, True)
        eps = float(max(1e-6, approx_epsilon_ratio) * peri)
        c2 = cv2.approxPolyDP(c, eps, True)
    else:
        c2 = c

    out = np.zeros_like(m)
    cv2.fillPoly(out, [c2], 255)
    return out


def apply_mask_bgr(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to BGR image; background set to black."""
    m = (mask > 0).astype(np.uint8)
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be HxWx3 BGR")
    out = img_bgr.copy()
    out[m == 0] = 0
    return out


def apply_mask_rgba(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create RGBA image where alpha=mask."""
    import cv2

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be HxWx3 BGR")
    alpha = (mask > 0).astype(np.uint8) * 255
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra

