# Copyright 2023 WolkenVision AG. All rights reserved.
"""Image filtering utilities for Tree-Seg.

This module provides an `ImageFilter` class to select "reasonable" panorama
images for a single tree point cloud. Only distance-based filtering is
implemented.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow importing `scripts/lib/*` as `lib.*` from the repository root.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.las_utils import load_las_cloud  # noqa: E402
from lib.poses import build_pose_index_by_basename, resolve_pano_image_path  # noqa: E402


class ImageFilter:
    """Filter panorama images by camera-to-tree distance.

    Args:
        pano_image_dir: Directory containing panorama images.
        pano_poses_csv: Path to `panoramicPoses.csv`.
        max_dist_m: Distance threshold (meters). Keep frames with
            `||cam_xyz - tree_center|| <= max_dist_m`.
    """

    def __init__(
        self,
        *,
        pano_image_dir: Path | str,
        pano_poses_csv: Path | str,
        max_dist_m: float,
    ) -> None:
        self.pano_image_dir = Path(pano_image_dir)
        self.pano_poses_csv = Path(pano_poses_csv)
        self.max_dist_m = float(max_dist_m)

        self.pose_index = build_pose_index_by_basename(self.pano_poses_csv)

        self.selected_paths: list[Path] = []
        self.selected_names: list[str] = []

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


__all__ = ["ImageFilter"]

if __name__ == "__main__":
    image_filter = ImageFilter(
        pano_image_dir="/root/autodl-fs/222-pcimg-data/panoramicImage",
        pano_poses_csv="/root/autodl-fs/222-pcimg-data/panoramicPoses.csv",
        max_dist_m=10,
    )
    start_time = time.time()
    image_list = image_filter.filter_by_distance(
        tree_las="/root/autodl-fs/222-pcimg-data/tree2.las",
    )
    end_time = time.time()
    print(f"filter_by_distance time: {end_time - start_time} seconds")
    print(f"image_list_length: {len(image_list)}")

