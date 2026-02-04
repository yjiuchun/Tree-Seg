#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lib.las_utils import LasCloud, downsample_points_ratio, downsample_points_step, load_las_cloud
from lib.mask_ops import MorphParams, apply_mask_bgr, apply_mask_rgba, points_to_mask, refine_mask
from lib.poses import PanoPose, build_pose_index_by_basename, resolve_pano_image_path
from lib.projection_equirect import world_dir_to_cam_dir, world_point_to_uv_equirect


@dataclass(frozen=True)
class SelectedFrame:
    img_path: str
    ts: float
    cam_xyz: List[float]
    q_c2w: List[float]  # (qw,qx,qy,qz)
    dist_m: float
    angle_deg: float
    # optional occlusion metrics (available when --map-las is provided)
    occl_vox_xy_count: int = 0
    tree_vox_xy_total: int = 0
    occl_ratio: float = 0.0


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _angle_deg_from_dcam_z(dz: float) -> float:
    dz = float(np.clip(dz, -1.0, 1.0))
    return float(np.degrees(np.arccos(dz)))


def select_frames(
    *,
    tree_center: np.ndarray,
    pose_index: Dict[str, PanoPose],
    pano_image_dir: Path,
    max_dist: float,
    max_angle_deg: float,
) -> List[SelectedFrame]:
    cos_thr = math.cos(math.radians(float(max_angle_deg)))
    selected: List[SelectedFrame] = []
    for basename, pose in pose_index.items():
        img_path = resolve_pano_image_path(pano_image_dir, basename)
        if img_path is None:
            continue
        cam_xyz = pose.cam_xyz
        d_world = tree_center - cam_xyz
        dist = float(np.linalg.norm(d_world))
        if dist > float(max_dist):
            continue
        n = float(np.linalg.norm(d_world))
        if n < 1e-12:
            continue
        d_world /= n
        d_cam = world_dir_to_cam_dir(d_world, pose.q_c2w)
        dz = float(d_cam[2])
        if dz < cos_thr:
            continue
        angle = _angle_deg_from_dcam_z(dz)
        selected.append(
            SelectedFrame(
                img_path=str(img_path),
                ts=float(pose.ts),
                cam_xyz=[float(x) for x in cam_xyz.tolist()],
                q_c2w=[float(x) for x in pose.q_c2w.tolist()],
                dist_m=dist,
                angle_deg=angle,
            )
        )
    selected.sort(key=lambda x: x.ts)
    return selected


def write_selected_jsonl(selected: Sequence[SelectedFrame], out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for fr in selected:
            f.write(json.dumps(asdict(fr), ensure_ascii=False) + "\n")


def read_selected_jsonl(path: Path) -> List[SelectedFrame]:
    out: List[SelectedFrame] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                SelectedFrame(
                    img_path=str(d["img_path"]),
                    ts=float(d["ts"]),
                    cam_xyz=[float(x) for x in d["cam_xyz"]],
                    q_c2w=[float(x) for x in d["q_c2w"]],
                    dist_m=float(d["dist_m"]),
                    angle_deg=float(d["angle_deg"]),
                    occl_vox_xy_count=int(d.get("occl_vox_xy_count", 0)),
                    tree_vox_xy_total=int(d.get("tree_vox_xy_total", 0)),
                    occl_ratio=float(d.get("occl_ratio", 0.0)),
                )
            )
    return out


def _voxelize_xy_unique_count(points_xyz: np.ndarray, voxel_size: float) -> int:
    """
    Quantize XY to a 2D voxel grid and return unique occupied voxel count.
    points_xyz: (N,3) or (N,2)
    """
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return 0
    xy = pts[:, :2]
    ij = np.floor(xy / voxel_size).astype(np.int64)  # (N,2)
    # fast unique rows via structured view
    v = np.ascontiguousarray(ij).view([("i", np.int64), ("j", np.int64)])
    return int(np.unique(v).shape[0])


def _filter_frames_by_occlusion_xy_voxel(
    *,
    selected: Sequence[SelectedFrame],
    tree_cloud: LasCloud,
    map_points: np.ndarray,
    occl_fov_deg: float,
    occl_radius_m: float,
    occl_voxel_size: float,
    occl_thr: float,
    occl_tube_radius_m: float,
    occl_tree_clearance_m: float,
    occl_tree_bbox_margin_m: float,
) -> List[SelectedFrame]:
    """
    Occlusion test in XY plane:
    - take map points within a horizontal sector (FOV+radius) around the camera->tree direction
    - keep points that lie between camera and tree (along ray) and close to ray (tube)
    - exclude points inside tree bbox (XY) to avoid counting tree itself
    - voxelize XY occupancy and compare against tree's XY voxel occupancy
    """
    tree_center = np.asarray(tree_cloud.center, dtype=np.float64).reshape(3)
    tree_bbox = tree_cloud.bbox  # x_min,x_max,y_min,y_max,z_min,z_max
    x_min, x_max, y_min, y_max = tree_bbox[0], tree_bbox[1], tree_bbox[2], tree_bbox[3]
    margin = float(max(0.0, occl_tree_bbox_margin_m))
    x0, x1 = x_min - margin, x_max + margin
    y0, y1 = y_min - margin, y_max + margin

    # denominator: tree xy voxel occupancy
    tree_vox_xy_total = _voxelize_xy_unique_count(tree_cloud.points, occl_voxel_size)
    if tree_vox_xy_total <= 0:
        # Degenerate; keep all frames unchanged.
        return list(selected)

    map_pts = np.asarray(map_points, dtype=np.float64).reshape(-1, 3)
    if map_pts.size == 0:
        return [
            SelectedFrame(**{**asdict(fr), "tree_vox_xy_total": int(tree_vox_xy_total)})  # type: ignore[arg-type]
            for fr in selected
        ]

    occl_radius_m = float(occl_radius_m)
    occl_fov_deg = float(occl_fov_deg)
    occl_thr = float(occl_thr)
    tube_r = float(occl_tube_radius_m)
    tree_clearance = float(occl_tree_clearance_m)

    if occl_radius_m <= 0 or tube_r <= 0:
        return [
            SelectedFrame(**{**asdict(fr), "tree_vox_xy_total": int(tree_vox_xy_total)})  # type: ignore[arg-type]
            for fr in selected
        ]

    # precompute cosine threshold for FOV/2
    half_fov = max(0.0, occl_fov_deg) * 0.5
    cos_thr = math.cos(math.radians(half_fov))
    cos_thr = float(np.clip(cos_thr, -1.0, 1.0))

    tree_xy = tree_center[:2]
    map_xy = map_pts[:, :2]
    map_x = map_xy[:, 0]
    map_y = map_xy[:, 1]

    out: List[SelectedFrame] = []
    filtered = 0

    for fr in selected:
        cam_xyz = np.asarray(fr.cam_xyz, dtype=np.float64).reshape(3)
        cam_xy = cam_xyz[:2]
        v_tree = tree_xy - cam_xy
        dist_xy = float(np.linalg.norm(v_tree))
        if dist_xy < 1e-8:
            out.append(SelectedFrame(**{**asdict(fr), "tree_vox_xy_total": int(tree_vox_xy_total)}))  # type: ignore[arg-type]
            continue
        d_xy = v_tree / dist_xy  # unit

        rel = map_xy - cam_xy  # (N,2)
        rel_x = rel[:, 0]
        rel_y = rel[:, 1]
        r2 = rel_x * rel_x + rel_y * rel_y
        r = np.sqrt(r2, dtype=np.float64)

        # radius filter (and avoid zero)
        m = (r <= occl_radius_m) & (r > 1e-9)
        if not np.any(m):
            out.append(
                SelectedFrame(
                    **{
                        **asdict(fr),
                        "occl_vox_xy_count": 0,
                        "tree_vox_xy_total": int(tree_vox_xy_total),
                        "occl_ratio": 0.0,
                    }
                )
            )  # type: ignore[arg-type]
            continue

        rel_xm = rel_x[m]
        rel_ym = rel_y[m]
        rm = r[m]

        # FOV filter by dot(rel, d) >= ||rel|| * cos_thr
        dot = rel_xm * float(d_xy[0]) + rel_ym * float(d_xy[1])
        m_fov = dot >= (rm * cos_thr)
        if not np.any(m_fov):
            out.append(
                SelectedFrame(
                    **{
                        **asdict(fr),
                        "occl_vox_xy_count": 0,
                        "tree_vox_xy_total": int(tree_vox_xy_total),
                        "occl_ratio": 0.0,
                    }
                )
            )  # type: ignore[arg-type]
            continue

        rel_xf = rel_xm[m_fov]
        rel_yf = rel_ym[m_fov]
        dotf = dot[m_fov]  # this is t along ray because d_xy is unit

        # between camera and tree (leave a clearance near tree)
        t_max = max(0.0, dist_xy - tree_clearance)
        m_seg = (dotf > 0.0) & (dotf < t_max)
        if not np.any(m_seg):
            out.append(
                SelectedFrame(
                    **{
                        **asdict(fr),
                        "occl_vox_xy_count": 0,
                        "tree_vox_xy_total": int(tree_vox_xy_total),
                        "occl_ratio": 0.0,
                    }
                )
            )  # type: ignore[arg-type]
            continue

        rel_xs = rel_xf[m_seg]
        rel_ys = rel_yf[m_seg]
        ts = dotf[m_seg]

        # perpendicular distance to ray
        perp_x = rel_xs - ts * float(d_xy[0])
        perp_y = rel_ys - ts * float(d_xy[1])
        perp2 = perp_x * perp_x + perp_y * perp_y
        m_tube = perp2 <= (tube_r * tube_r)
        if not np.any(m_tube):
            out.append(
                SelectedFrame(
                    **{
                        **asdict(fr),
                        "occl_vox_xy_count": 0,
                        "tree_vox_xy_total": int(tree_vox_xy_total),
                        "occl_ratio": 0.0,
                    }
                )
            )  # type: ignore[arg-type]
            continue

        # reconstruct map XYZ subset indices to apply bbox filter without materializing all intermediate points
        idx_m = np.flatnonzero(m)
        idx_f = idx_m[m_fov]
        idx_s = idx_f[m_seg]
        idx_t = idx_s[m_tube]
        pts_t = map_pts[idx_t]

        # exclude points inside tree bbox (XY, with margin)
        px = pts_t[:, 0]
        py = pts_t[:, 1]
        m_not_tree = ~((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1))
        pts_occ = pts_t[m_not_tree]

        occl_vox = _voxelize_xy_unique_count(pts_occ, occl_voxel_size)
        ratio = float(occl_vox) / float(tree_vox_xy_total)

        if ratio > occl_thr:
            filtered += 1
            continue

        out.append(
            SelectedFrame(
                **{
                    **asdict(fr),
                    "occl_vox_xy_count": int(occl_vox),
                    "tree_vox_xy_total": int(tree_vox_xy_total),
                    "occl_ratio": float(ratio),
                }
            )
        )  # type: ignore[arg-type]

    print(
        f"[occlusion] kept {len(out)}/{len(selected)} frames "
        f"(filtered={filtered}, thr={occl_thr}, voxel={occl_voxel_size}, fov={occl_fov_deg}, radius={occl_radius_m})"
    )
    return out


def copy_selected_images(selected: Sequence[SelectedFrame], selected_dir: Path) -> None:
    _ensure_dir(selected_dir)
    for fr in selected:
        src = Path(fr.img_path)
        if not src.exists():
            continue
        shutil.copy2(src, selected_dir / src.name)


def project_masks(
    *,
    selected: Sequence[SelectedFrame],
    points: np.ndarray,
    out_masks_raw_dir: Path,
    flip_v: bool,
    morph: MorphParams,
) -> None:
    import cv2

    _ensure_dir(out_masks_raw_dir)
    for idx, fr in enumerate(selected):
        img_path = Path(fr.img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cam_xyz = np.asarray(fr.cam_xyz, dtype=np.float64)
        q_c2w = np.asarray(fr.q_c2w, dtype=np.float64)

        uv_list: List[Tuple[int, int]] = []
        for p in points:
            uv = world_point_to_uv_equirect(
                p, cam_xyz, q_c2w, w, h, flip_v=flip_v, require_in_front=False
            )
            if uv is not None:
                uv_list.append(uv)
        uv_points = np.asarray(uv_list, dtype=np.int32) if uv_list else np.zeros((0, 2), dtype=np.int32)
        mask = points_to_mask(uv_points, h, w, morph=morph)
        out_path = out_masks_raw_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), mask)
        if idx == 0 or (idx + 1) % 50 == 0:
            print(f"[project] {idx+1}/{len(selected)} wrote {out_path.name} (points={len(points)}, hits={len(uv_list)})")


def refine_and_segment(
    *,
    selected: Sequence[SelectedFrame],
    masks_raw_dir: Path,
    masks_refined_dir: Path,
    segmented_dir: Path,
    refine_mode: str,
    write_rgba: bool,
) -> None:
    import cv2

    _ensure_dir(masks_refined_dir)
    _ensure_dir(segmented_dir)
    for idx, fr in enumerate(selected):
        img_path = Path(fr.img_path)
        raw_mask_path = masks_raw_dir / f"{img_path.stem}.png"
        if not raw_mask_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        raw = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        refined = refine_mask(raw, mode=refine_mode)  # type: ignore[arg-type]
        out_refined = masks_refined_dir / raw_mask_path.name
        cv2.imwrite(str(out_refined), refined)

        seg = apply_mask_bgr(img, refined)
        out_seg = segmented_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_seg), seg)

        if write_rgba:
            rgba = apply_mask_rgba(img, refined)
            out_rgba = segmented_dir / f"{img_path.stem}_rgba.png"
            cv2.imwrite(str(out_rgba), rgba)

        if idx == 0 or (idx + 1) % 50 == 0:
            print(f"[refine] {idx+1}/{len(selected)} wrote {out_seg.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tree pano pipeline (self-contained in Tree_Segment)")
    p.add_argument("--stage", choices=["select", "project", "refine", "all"], default="all")
    p.add_argument("--tree-las", type=str, required=True)
    p.add_argument("--pano-image-dir", type=str, required=True)
    p.add_argument("--pano-poses", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--max-dist", type=float, default=10.0)
    p.add_argument("--max-angle-deg", type=float, default=90.0)
    p.add_argument("--copy-selected", action="store_true")
    p.add_argument("--selected-dir", type=str, default=None)

    # optional occlusion filtering using a global map point cloud
    p.add_argument("--map-las", type=str, default=None, help="Optional global map LAS for occlusion filtering")
    p.add_argument("--map-downsample-step", type=int, default=1, help="Downsample step for map LAS points")
    p.add_argument("--occl-fov-deg", type=float, default=100.0, help="Horizontal sector FOV (degrees) for occlusion search")
    p.add_argument("--occl-radius-m", type=float, default=10.0, help="Search radius (meters) from camera in XY")
    p.add_argument("--occl-voxel-size", type=float, default=0.20, help="XY voxel size (meters) for occupancy ratio")
    p.add_argument("--occl-thr", type=float, default=0.40, help="Filter frame if occlusion_ratio > thr")
    p.add_argument("--occl-tube-radius-m", type=float, default=1.0, help="Ray tube radius in XY (meters)")
    p.add_argument("--occl-tree-clearance-m", type=float, default=0.5, help="Clearance near tree end along the ray (meters)")
    p.add_argument("--occl-tree-bbox-margin-m", type=float, default=0.5, help="XY margin around tree bbox to exclude points from occluders")

    p.add_argument("--downsample-step", type=int, default=1)
    p.add_argument("--downsample-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--flip-v", action="store_true", help="Flip v coordinate (dataset-specific)")
    p.add_argument("--morph-kernel", type=int, default=9)
    p.add_argument("--dilate-iter", type=int, default=2)
    p.add_argument("--close-iter", type=int, default=2)

    p.add_argument("--refine", choices=["largest_contour", "convex_hull", "approx"], default="convex_hull")
    p.add_argument("--write-rgba", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    tree_las = Path(args.tree_las)
    pano_image_dir = Path(args.pano_image_dir)
    pano_poses = Path(args.pano_poses)
    out_dir = Path(args.out_dir)

    _ensure_dir(out_dir)
    selected_jsonl = out_dir / "selected_frames.jsonl"
    masks_raw_dir = out_dir / "masks_raw"
    masks_refined_dir = out_dir / "masks_refined"
    segmented_dir = out_dir / "segmented"

    if args.stage in ("select", "all"):
        cloud = load_las_cloud(tree_las)
        pose_index = build_pose_index_by_basename(pano_poses)
        selected = select_frames(
            tree_center=cloud.center,
            pose_index=pose_index,
            pano_image_dir=pano_image_dir,
            max_dist=args.max_dist,
            max_angle_deg=args.max_angle_deg,
        )
        print(f"[select] selected {len(selected)} frames")

        # Optional occlusion filtering using a global map point cloud.
        if args.map_las:
            map_las = Path(args.map_las)
            map_cloud = load_las_cloud(map_las)
            map_pts = downsample_points_step(map_cloud.points, int(args.map_downsample_step))
            # Pre-crop around tree in XY to reduce per-frame work.
            tree_xy = cloud.center[:2].astype(np.float64)
            map_xy = map_pts[:, :2]
            r_keep = float(args.max_dist) + float(args.occl_radius_m)
            if r_keep > 0:
                dxy = map_xy - tree_xy
                keep = (dxy[:, 0] * dxy[:, 0] + dxy[:, 1] * dxy[:, 1]) <= (r_keep * r_keep)
                map_pts = map_pts[keep]
            print(f"[occlusion] map points (downsampled+cropped): {len(map_cloud.points)} -> {len(map_pts)}")
            selected = _filter_frames_by_occlusion_xy_voxel(
                selected=selected,
                tree_cloud=cloud,
                map_points=map_pts,
                occl_fov_deg=float(args.occl_fov_deg),
                occl_radius_m=float(args.occl_radius_m),
                occl_voxel_size=float(args.occl_voxel_size),
                occl_thr=float(args.occl_thr),
                occl_tube_radius_m=float(args.occl_tube_radius_m),
                occl_tree_clearance_m=float(args.occl_tree_clearance_m),
                occl_tree_bbox_margin_m=float(args.occl_tree_bbox_margin_m),
            )

        write_selected_jsonl(selected, selected_jsonl)
        if args.copy_selected:
            sd = Path(args.selected_dir) if args.selected_dir else (out_dir / "selected_images")
            copy_selected_images(selected, sd)
            print(f"[select] copied images -> {sd}")

    if args.stage in ("project", "all"):
        selected = read_selected_jsonl(selected_jsonl)
        cloud = load_las_cloud(tree_las)
        pts = cloud.points
        if args.downsample_ratio is not None:
            pts = downsample_points_ratio(pts, args.downsample_ratio, seed=args.seed)
        else:
            pts = downsample_points_step(pts, args.downsample_step)
        print(f"[project] points: {len(cloud.points)} -> {len(pts)}")
        morph = MorphParams(kernel=args.morph_kernel, dilate_iter=args.dilate_iter, close_iter=args.close_iter)
        project_masks(
            selected=selected,
            points=pts,
            out_masks_raw_dir=masks_raw_dir,
            flip_v=bool(args.flip_v),
            morph=morph,
        )

    if args.stage in ("refine", "all"):
        selected = read_selected_jsonl(selected_jsonl)
        refine_and_segment(
            selected=selected,
            masks_raw_dir=masks_raw_dir,
            masks_refined_dir=masks_refined_dir,
            segmented_dir=segmented_dir,
            refine_mode=str(args.refine),
            write_rgba=bool(args.write_rgba),
        )

    print("[done]")


if __name__ == "__main__":
    main()

