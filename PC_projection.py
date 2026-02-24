# Copyright 2023 WolkenVision AG. All rights reserved.
"""Point-cloud to panorama projection with optional obstacle-based frame discard.

After distance-based image selection (ImageFilter), projects tree point cloud
onto each selected panorama to produce binary masks. Optionally uses map.las
to discard frames where obstacle projection area is large relative to tree
projection area.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# Allow importing `scripts/lib/*` as `lib.*` from the repository root.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.las_utils import load_las_cloud, downsample_points_step  # noqa: E402
from lib.mask_ops import MorphParams, points_to_mask  # noqa: E402
from lib.poses import build_pose_index_by_basename, resolve_pano_image_path  # noqa: E402
from lib.projection_equirect import world_point_to_uv_equirect  # noqa: E402

# ImageFilter used only by run_pc_projection to produce selected_basenames.
from ImageFilter import ImageFilter  # noqa: E402

__all__ = ["PCProjection", "run_pc_projection"]


def _obstacle_points_in_ray_tube(
    map_points: np.ndarray,
    cam_xyz: np.ndarray,
    tree_center: np.ndarray,
    tree_bbox: Tuple[float, float, float, float, float, float],
    tube_radius_m: float,
    tree_clearance_m: float,
    tree_bbox_margin_m: float = 0.5,
) -> np.ndarray:
    """Return map points inside the camera-to-tree ray tube (potential occluders).

    Keeps points with 0 < t <= dist - tree_clearance_m and perpendicular
    distance <= tube_radius_m. Optionally excludes points inside tree bbox
    (XY) expanded by tree_bbox_margin_m.
    """
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
    tree_center = np.asarray(tree_center, dtype=np.float64).reshape(3)
    map_pts = np.asarray(map_points, dtype=np.float64).reshape(-1, 3)
    if map_pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    v = tree_center - cam_xyz
    dist = float(np.linalg.norm(v))
    if dist < 1e-9:
        return np.zeros((0, 3), dtype=np.float64)
    d = v / dist

    rel = map_pts - cam_xyz
    t = rel @ d
    t_max = max(0.0, dist - tree_clearance_m)
    m_t = (t > 0.0) & (t <= t_max)
    if not np.any(m_t):
        return np.zeros((0, 3), dtype=np.float64)

    rel_t = rel[m_t]
    t_sel = t[m_t].reshape(-1, 1)
    perp = rel_t - t_sel * d.reshape(1, 3)
    perp_dist = np.linalg.norm(perp, axis=1)
    m_tube = perp_dist <= tube_radius_m
    if not np.any(m_tube):
        return np.zeros((0, 3), dtype=np.float64)

    occ_pts = (rel_t[m_tube] + cam_xyz).astype(np.float64)

    x0, x1, y0, y1 = (
        tree_bbox[0] - tree_bbox_margin_m,
        tree_bbox[1] + tree_bbox_margin_m,
        tree_bbox[2] - tree_bbox_margin_m,
        tree_bbox[3] + tree_bbox_margin_m,
    )
    px = occ_pts[:, 0]
    py = occ_pts[:, 1]
    m_not_tree = ~((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1))
    return occ_pts[m_not_tree]


def _project_points_to_mask(
    points_xyz: np.ndarray,
    cam_xyz: np.ndarray,
    q_c2w: np.ndarray,
    width: int,
    height: int,
    morph: MorphParams,
    flip_v: bool = True,
) -> np.ndarray:
    """Project world points to equirectangular image and build binary mask."""
    uv_list: list[Tuple[int, int]] = []
    for p in points_xyz:
        uv = world_point_to_uv_equirect(
            p, cam_xyz, q_c2w, width, height,
            flip_v=flip_v, require_in_front=False,
        )
        if uv is not None:
            uv_list.append(uv)
    uv_points = (
        np.asarray(uv_list, dtype=np.int32)
        if uv_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    return points_to_mask(uv_points, height, width, morph=morph)


class PCProjection:
    """Point-cloud to panorama projection with optional obstacle-based discard.

    Independent of image filtering: input is a list of selected image basenames
    (e.g. from ImageFilter.filter_by_distance). Config and pose index are fixed
    at construction; call project(tree_las, output_dir, selected_basenames) to
    write masks and optionally discard occluded frames.
    """

    def __init__(
        self,
        *,
        pano_image_dir: Path | str,
        pano_poses_csv: Path | str,
        downsample_step: int = 50,
        flip_v: bool = True,
        morph_kernel: int = 9,
        dilate_iter: int = 2,
        close_iter: int = 2,
        map_las: Optional[Path | str] = None,
        occl_area_ratio_thr: float = 0.4,
        tube_radius_m: float = 1.0,
        tree_clearance_m: float = 0.5,
        tree_bbox_margin_m: float = 0.5,
    ) -> None:
        self.pano_image_dir = Path(pano_image_dir)
        self.pano_poses_csv = Path(pano_poses_csv)
        self.downsample_step = int(downsample_step)
        self.flip_v = bool(flip_v)
        self.morph = MorphParams(
            kernel=morph_kernel, dilate_iter=dilate_iter, close_iter=close_iter,
        )
        self.map_las = Path(map_las) if map_las is not None else None
        self.occl_area_ratio_thr = float(occl_area_ratio_thr)
        self.tube_radius_m = float(tube_radius_m)
        self.tree_clearance_m = float(tree_clearance_m)
        self.tree_bbox_margin_m = float(tree_bbox_margin_m)

        self.pose_index = build_pose_index_by_basename(self.pano_poses_csv)
        self._map_points: Optional[np.ndarray] = None
        if self.map_las is not None:
            self._map_points = load_las_cloud(self.map_las).points

        self.written: list[str] = []
        self.discarded: list[str] = []

    def project(
        self,
        *,
        tree_las: Path | str,
        output_dir: Path | str,
        selected_basenames: Sequence[str],
        write_params_json: bool = True,
    ) -> Tuple[list[str], list[str]]:
        """Project one tree point cloud to panorama masks.

        Args:
            tree_las: Path to the tree LAS file.
            output_dir: Output root directory for this tree (masks_raw under it).
            selected_basenames: List of image basenames to process (e.g. output
                of ImageFilter.filter_by_distance). Each element is filename with
                extension.
            write_params_json: If True, write params.json under output_dir.

        Returns:
            (written_basenames, discarded_basenames).
        """
        tree_las = Path(tree_las)
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        selected_names = list(selected_basenames)

        params = {
            "tree_las": str(tree_las),
            "pano_image_dir": str(self.pano_image_dir),
            "pano_poses_csv": str(self.pano_poses_csv),
            "output_dir": str(output_dir),
            "selected_basenames_count": len(selected_names),
            "downsample_step": self.downsample_step,
            "flip_v": self.flip_v,
            "morph_kernel": self.morph.kernel,
            "dilate_iter": self.morph.dilate_iter,
            "close_iter": self.morph.close_iter,
            "map_las": str(self.map_las) if self.map_las is not None else None,
            "occl_area_ratio_thr": self.occl_area_ratio_thr,
            "tube_radius_m": self.tube_radius_m,
            "tree_clearance_m": self.tree_clearance_m,
            "tree_bbox_margin_m": self.tree_bbox_margin_m,
        }
        print("Parameter paths:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        if write_params_json:
            params_path = output_dir / "params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"Wrote {params_path}")

        if not selected_names:
            print("No selected images to project.")
            self.written = []
            self.discarded = []
            return [], []

        tree_cloud = load_las_cloud(tree_las)
        tree_points = tree_cloud.points
        if self.downsample_step > 1:
            tree_points = downsample_points_step(tree_points, self.downsample_step)
        tree_center = tree_cloud.center.astype(np.float64).reshape(3)
        tree_bbox = tree_cloud.bbox

        masks_raw_dir = output_dir / "masks_raw"
        if not masks_raw_dir.exists():
            masks_raw_dir.mkdir(parents=True)

        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "opencv-python required for reading images. pip install opencv-python",
            ) from e

        written: list[str] = []
        discarded: list[str] = []

        for idx, basename in enumerate(selected_names):
            key = Path(basename).name
            pose = self.pose_index.get(key)
            if pose is None:
                continue
            img_path = resolve_pano_image_path(self.pano_image_dir, key)
            if img_path is None or not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            height, width = img.shape[:2]

            cam_xyz = np.asarray(pose.cam_xyz, dtype=np.float64)
            q_c2w = np.asarray(pose.q_c2w, dtype=np.float64)

            tree_mask = _project_points_to_mask(
                tree_points, cam_xyz, q_c2w, width, height,
                self.morph, flip_v=self.flip_v,
            )
            area_tree = int((tree_mask > 0).sum())

            if self._map_points is not None and self._map_points.size > 0:
                occ_pts = _obstacle_points_in_ray_tube(
                    self._map_points, cam_xyz, tree_center, tree_bbox,
                    tube_radius_m=self.tube_radius_m,
                    tree_clearance_m=self.tree_clearance_m,
                    tree_bbox_margin_m=self.tree_bbox_margin_m,
                )
                if occ_pts.shape[0] > 0:
                    obst_mask = _project_points_to_mask(
                        occ_pts, cam_xyz, q_c2w, width, height,
                        self.morph, flip_v=self.flip_v,
                    )
                    area_obstacle = int((obst_mask > 0).sum())
                    ratio = area_obstacle / max(area_tree, 1)
                    if ratio > self.occl_area_ratio_thr:
                        discarded.append(key)
                        if (idx + 1) % 50 == 0 or idx == 0:
                            print(
                                f"[project] {idx+1}/{len(selected_names)} discard "
                                f"(occl ratio {ratio:.3f}) {key}",
                            )
                        continue

            out_path = masks_raw_dir / f"{img_path.stem}.png"
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True)
            cv2.imwrite(str(out_path), tree_mask)
            written.append(key)
            if (idx + 1) % 50 == 0 or idx == 0:
                print(
                    f"[project] {idx+1}/{len(selected_names)} wrote {out_path.name} "
                    f"(points={len(tree_points)})",
                )

        print(
            f"Wrote {len(written)} masks to {masks_raw_dir}; "
            f"discarded {len(discarded)} by occlusion.",
        )
        self.written = written
        self.discarded = discarded
        return written, discarded


def run_pc_projection(
    *,
    tree_las: Path | str,
    pano_image_dir: Path | str,
    pano_poses_csv: Path | str,
    output_dir: Path | str,
    max_dist_m: float = 10.0,
    downsample_step: int = 50,
    flip_v: bool = True,
    morph_kernel: int = 9,
    dilate_iter: int = 2,
    close_iter: int = 2,
    map_las: Optional[Path | str] = None,
    occl_area_ratio_thr: float = 0.4,
    tube_radius_m: float = 1.0,
    tree_clearance_m: float = 0.5,
    tree_bbox_margin_m: float = 0.5,
    write_params_json: bool = True,
) -> Tuple[list[str], list[str]]:
    """Run distance filter, then project tree point cloud to panorama masks.

    Uses ImageFilter to get selected image basenames, then PCProjection to
    project. Projection class is independent of ImageFilter; this function
    only wires the two together.

    Returns:
        (written_basenames, discarded_basenames).
    """
    image_filter = ImageFilter(
        pano_image_dir=pano_image_dir,
        pano_poses_csv=pano_poses_csv,
        max_dist_m=max_dist_m,
    )
    selected_basenames = image_filter.filter_by_distance(
        tree_las=tree_las, output_dir=None,
    )
    proj = PCProjection(
        pano_image_dir=pano_image_dir,
        pano_poses_csv=pano_poses_csv,
        downsample_step=downsample_step,
        flip_v=flip_v,
        morph_kernel=morph_kernel,
        dilate_iter=dilate_iter,
        close_iter=close_iter,
        map_las=map_las,
        occl_area_ratio_thr=occl_area_ratio_thr,
        tube_radius_m=tube_radius_m,
        tree_clearance_m=tree_clearance_m,
        tree_bbox_margin_m=tree_bbox_margin_m,
    )
    return proj.project(
        tree_las=tree_las,
        output_dir=output_dir,
        selected_basenames=selected_basenames,
        write_params_json=write_params_json,
    )


def main() -> None:

    Tree_las = "/root/autodl-fs/222-pcimg-data/tree2.las"
    Panorama_image_dir = "/root/autodl-fs/222-pcimg-data/panoramicImage"
    Panorama_poses_csv = "/root/autodl-fs/222-pcimg-data/panoramicPoses.csv"
    Output_dir = "/root/autodl-fs/222-pcimg-data/output"
    Max_dist_m = 10.0
    Downsample_step = 50
    Flip_v = True
    Morph_kernel = 9
    Dilate_iter = 2
    Close_iter = 2
    Map_las = "/root/autodl-fs/222-pcimg-data/map2.las"
    Occl_area_ratio_thr = 0.4
    Tube_radius_m = 1.0
    Tree_clearance_m = 0.5
    Tree_bbox_margin_m = 0.5
    Write_params_json = True

    run_pc_projection(tree_las=Tree_las, pano_image_dir=Panorama_image_dir, pano_poses_csv=Panorama_poses_csv, output_dir=Output_dir, max_dist_m=Max_dist_m, downsample_step=Downsample_step, flip_v=Flip_v, morph_kernel=Morph_kernel, dilate_iter=Dilate_iter, close_iter=Close_iter, map_las=Map_las, occl_area_ratio_thr=Occl_area_ratio_thr, tube_radius_m=Tube_radius_m, tree_clearance_m=Tree_clearance_m, tree_bbox_margin_m=Tree_bbox_margin_m, write_params_json=Write_params_json)



if __name__ == "__main__":
    main()
