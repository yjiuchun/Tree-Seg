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
                )
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
                p, cam_xyz, q_c2w, w, h, flip_v=flip_v, require_in_front=True
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

