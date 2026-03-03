"""Render each tree point cloud from 4 orthographic viewpoints using RGB colors.

Reads .las files from tree-las/, rotates around Z axis for front/right/back/left
views, performs orthographic projection, and saves RGB images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import laspy
import numpy as np


VIEW_ANGLES = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}


def load_las_with_rgb(las_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a LAS file and return (points_xyz (N,3), colors_rgb (N,3) uint8)."""
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    points = np.stack([x, y, z], axis=1)

    r = np.asarray(las.red, dtype=np.float64)
    g = np.asarray(las.green, dtype=np.float64)
    b = np.asarray(las.blue, dtype=np.float64)
    rgb = np.stack([r, g, b], axis=1)

    if rgb.max() > 255:
        rgb = rgb / 257.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return points, rgb


def rotate_z(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points around the Z axis by angle_deg degrees."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return points @ R.T


def render_orthographic(
    points: np.ndarray,
    colors: np.ndarray,
    angle_deg: float,
    pixels_per_meter: float,
    point_radius: int,
    padding: int,
) -> np.ndarray:
    """Render an orthographic projection of the point cloud.

    After rotating by angle_deg around Z, projects onto the XZ plane.
    The frontmost point (smallest Y) determines pixel color.
    """
    rotated = rotate_z(points, angle_deg)

    x = rotated[:, 0]
    y = rotated[:, 1]
    z = rotated[:, 2]

    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    w = int(np.ceil((x_max - x_min) * pixels_per_meter)) + 2 * padding
    h = int(np.ceil((z_max - z_min) * pixels_per_meter)) + 2 * padding
    w = max(w, 1)
    h = max(h, 1)

    col = ((x - x_min) * pixels_per_meter + padding).astype(np.int32)
    row = ((z_max - z) * pixels_per_meter + padding).astype(np.int32)
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)

    order = np.argsort(-y)

    img = np.zeros((h, w, 3), dtype=np.uint8)

    for idx in order:
        c_bgr = (int(colors[idx, 2]), int(colors[idx, 1]), int(colors[idx, 0]))
        cv2.circle(img, (int(col[idx]), int(row[idx])), point_radius, c_bgr, -1)

    return img


def process_one_tree(
    las_path: Path,
    out_dir: Path,
    pixels_per_meter: float,
    point_radius: int,
    padding: int,
) -> None:
    """Load one LAS file and render 4 views."""
    points, colors = load_las_with_rgb(las_path)
    center = points.mean(axis=0)
    points_centered = points - center

    tree_id = las_path.stem
    tree_out = out_dir / tree_id
    tree_out.mkdir(parents=True, exist_ok=True)

    for view_name, angle in VIEW_ANGLES.items():
        img = render_orthographic(
            points_centered, colors, angle,
            pixels_per_meter=pixels_per_meter,
            point_radius=point_radius,
            padding=padding,
        )
        out_path = tree_out / f"{view_name}.png"
        cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tree point clouds to RGB images")
    parser.add_argument(
        "--las-dir",
        type=Path,
        default=Path("/root/Tree-Seg/tree-las"),
        help="Directory containing .las files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/Tree-Seg/tree-las/img"),
        help="Output image directory",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=80.0,
        help="Rendering scale (pixels per meter)",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=3,
        help="Rendered point radius in pixels",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Image border padding in pixels",
    )
    parser.add_argument(
        "--tree-ids",
        type=str,
        default=None,
        help="Comma-separated tree IDs to process (e.g. '1,2,3'). If not set, process all.",
    )
    args = parser.parse_args()

    las_dir = args.las_dir
    if not las_dir.exists():
        print(f"LAS directory not found: {las_dir}")
        sys.exit(1)

    las_paths = sorted(las_dir.glob("*.las"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if args.tree_ids:
        wanted = set(args.tree_ids.split(","))
        las_paths = [p for p in las_paths if p.stem in wanted]

    if not las_paths:
        print("No LAS files found.")
        sys.exit(0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(las_paths)} trees -> {args.out_dir}")
    print(f"Scale: {args.pixels_per_meter} px/m, point radius: {args.point_radius}, padding: {args.padding}")

    for i, las_path in enumerate(las_paths):
        print(f"[{i+1}/{len(las_paths)}] {las_path.stem}")
        process_one_tree(
            las_path, args.out_dir,
            pixels_per_meter=args.pixels_per_meter,
            point_radius=args.point_radius,
            padding=args.padding,
        )

    print("Done.")


if __name__ == "__main__":
    main()
