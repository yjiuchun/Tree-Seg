# Copyright 2023 WolkenVision AG. All rights reserved.
"""Image filtering utilities for Tree-Seg.

This module provides an `ImageFilter` class to select "reasonable" panorama
images for a single tree point cloud.

Currently implemented:
    - Filter panorama images by camera-to-tree distance.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Allow importing `scripts/lib/*` as `lib.*` from the repository root.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.las_utils import load_las_cloud  # noqa: E402
from lib.poses import build_pose_index_by_basename, resolve_pano_image_path  # noqa: E402


class ImageFilter:
    """Filter panorama images using simple geometric/point-cloud heuristics.

    Args:
        pano_image_dir: Directory containing panorama images.
        pano_poses_csv: Path to `panoramicPoses.csv`.
        map_las: Optional global map LAS path (reserved for occlusion filtering).
        max_dist_m: Distance threshold (meters). Keep frames with
            `||cam_xyz - tree_center|| <= max_dist_m`.
        occl_ratio_thr: Occlusion ratio threshold (reserved for occlusion filtering).
    """

    def __init__(
        self,
        *,
        pano_image_dir: Path | str,
        pano_poses_csv: Path | str,
        map_las: Path | str | None,
        max_dist_m: float,
        occl_ratio_thr: float,
    ) -> None:
        self.pano_image_dir = Path(pano_image_dir)
        self.pano_poses_csv = Path(pano_poses_csv)
        self.map_las = Path(map_las) if map_las is not None else None
        self.max_dist_m = float(max_dist_m)
        self.occl_ratio_thr = float(occl_ratio_thr)

        self.pose_index = build_pose_index_by_basename(self.pano_poses_csv)

        self._map_points: Optional[np.ndarray] = None
        if self.map_las is not None:
            self._map_points = load_las_cloud(self.map_las).points

        self.selected_paths: list[Path] = []
        self.selected_names: list[str] = []

    @staticmethod
    def _voxelize_unique_count(points_xyz: np.ndarray, voxel_size: float) -> int:
        """Voxelize points in 3D and return the number of unique occupied voxels."""
        voxel_size = float(voxel_size)
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")

        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return 0

        ijk = np.floor(pts / voxel_size).astype(np.int64)  # (N,3)
        v = np.ascontiguousarray(ijk).view(
            [("i", np.int64), ("j", np.int64), ("k", np.int64)]
        )
        return int(np.unique(v).shape[0])

    def filter_by_distance(
        self,
        *,
        tree_las: Path | str,
        output_dir: Optional[Path | str] = None,
    ) -> list[str]:
        """Filter images by distance threshold.

        Args:
            tree_las: Path to the LAS file containing the tree point cloud. The
                tree center will be computed from this file for filtering.
            output_dir: Output root directory. If provided, selected images will
                be copied into: `output_dir/<tree_name>/image_filter/`, where
                `tree_name` is derived from `Path(tree_las).stem`.

        Returns:
            A list of selected **image basenames** (including extension).
        """
        tree_las_path = Path(tree_las)
        cloud = load_las_cloud(tree_las_path)
        tree_center = cloud.center.astype(np.float64).reshape(3)

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            tree_name = tree_las_path.stem
            out_dir = out_dir / tree_name / "image_filter"
            out_dir.mkdir(parents=True, exist_ok=True)

        selected_paths: list[Path] = []
        selected_names: list[str] = []

        for basename, pose in self.pose_index.items():
            img_path = resolve_pano_image_path(self.pano_image_dir, basename)
            if img_path is None:
                continue

            dist = float(np.linalg.norm(tree_center - pose.cam_xyz))
            if dist > self.max_dist_m:
                continue

            selected_paths.append(img_path)
            selected_names.append(img_path.name)

            if out_dir is not None:
                shutil.copy2(img_path, out_dir / img_path.name)

        self.selected_paths = selected_paths
        self.selected_names = selected_names
        return list(self.selected_names)

    def filter_by_occlusion(
        self,
        image_names: Sequence[str],
        *,
        tree_las: Path | str,
        output_dir: Optional[Path | str] = None,
        voxel_size: float = 0.05,
        occl_ratio_thr: Optional[float] = None,
        tube_radius_m: Optional[float] = None,
        tree_clearance_m: float = 0.0,
    ) -> list[str]:
        """Filter images by occlusion ratio using `map.las` and `tree.las`.

        Occluders are defined as map points that lie close to the camera->tree_center
        ray segment (a 3D tube). We voxelize both occluders and tree points using
        the same `voxel_size` and compute:

            occl_ratio = occluder_voxels / tree_voxels

        If `occl_ratio > occl_ratio_thr`, the image is filtered out.

        Args:
            image_names: Output list from `filter_by_distance()` (image basenames).
            tree_las: Path to the LAS file containing the tree point cloud.
            output_dir: Output root directory. If provided, kept images will be
                copied into: `output_dir/<tree_name>/image_filter_occl/`.
            voxel_size: Voxel size (meters) used for 3D voxelization.
            occl_ratio_thr: Threshold for filtering. If None, use `self.occl_ratio_thr`.
            tube_radius_m: Ray tube radius (meters). If None, a radius derived
                from the tree bounding box is used.
            tree_clearance_m: Ignore occluders within this distance to the tree
                along the ray (meters).

        Returns:
            A list of kept **image basenames** (including extension).
        """
        if self._map_points is None:
            raise ValueError("map_las is required for occlusion filtering, but is None.")

        tree_las_path = Path(tree_las)
        tree_cloud = load_las_cloud(tree_las_path)
        tree_center = tree_cloud.center.astype(np.float64).reshape(3)
        tree_points = tree_cloud.points

        tree_vox = self._voxelize_unique_count(tree_points, voxel_size)
        if tree_vox <= 0:
            return []

        thr = float(self.occl_ratio_thr if occl_ratio_thr is None else occl_ratio_thr)

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            tree_name = tree_las_path.stem
            out_dir = out_dir / tree_name / "image_filter_occl"
            out_dir.mkdir(parents=True, exist_ok=True)

        if tube_radius_m is None:
            x_min, x_max, y_min, y_max, z_min, z_max = tree_cloud.bbox
            ex = float(x_max - x_min)
            ey = float(y_max - y_min)
            ez = float(z_max - z_min)
            tube_radius_m = max(0.5 * max(ex, ey, ez), float(voxel_size))

        tube_r = float(tube_radius_m)
        clearance = max(0.0, float(tree_clearance_m))

        kept_names: list[str] = []

        for name in image_names:
            key = Path(name).name
            pose = self.pose_index.get(key)
            if pose is None:
                continue

            cam_xyz = np.asarray(pose.cam_xyz, dtype=np.float64).reshape(3)
            v = tree_center - cam_xyz
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                continue
            d = v / dist

            rel = self._map_points - cam_xyz
            t = rel @ d
            t_max = max(0.0, dist - clearance)
            m_t = (t > 0.0) & (t < t_max)
            if not np.any(m_t):
                kept_names.append(key)
                continue

            rel_t = rel[m_t]
            t_sel = t[m_t].reshape(-1, 1)
            perp = rel_t - t_sel * d.reshape(1, 3)
            perp_dist = np.linalg.norm(perp, axis=1)
            m = perp_dist <= tube_r
            occ_pts = rel_t[m] + cam_xyz

            occl_vox = self._voxelize_unique_count(occ_pts, voxel_size)
            ratio = float(occl_vox) / float(tree_vox)
            if ratio > thr:
                continue

            kept_names.append(key)
            if out_dir is not None:
                img_path = resolve_pano_image_path(self.pano_image_dir, key)
                if img_path is not None:
                    shutil.copy2(img_path, out_dir / img_path.name)

        return kept_names

__all__ = ["ImageFilter"]

if __name__ == "__main__":
    import time
    image_filter = ImageFilter(
        pano_image_dir="/home/yjc/Project/plant_classfication/222_2025-11-25-155532/panoramicImage",
        pano_poses_csv="/home/yjc/Project/plant_classfication/222_2025-11-25-155532/panoramicPoses.csv",
        map_las="/home/yjc/Project/plant_classfication/222_2025-11-25-155532/map2.las",
        max_dist_m=10,
        occl_ratio_thr=1,
    )
    start_time = time.time()
    image_list = image_filter.filter_by_distance(
        tree_las="/home/yjc/Project/plant_classfication/222_2025-11-25-155532/tree2.las",
    )
    end_time = time.time()
    print(f"filter_by_distance time: {end_time - start_time} seconds")
    kept = image_filter.filter_by_occlusion(
        image_list,
        tree_las="/home/yjc/Project/plant_classfication/222_2025-11-25-155532/tree2.las",
        output_dir="/home/yjc/Project/plant_classfication/Tree-Seg/output-dir",
        voxel_size=1,
    )
    print(f"kept after occlusion: {len(kept)}/{len(image_list)}")
