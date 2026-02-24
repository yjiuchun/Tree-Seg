# Copyright 2023 WolkenVision AG. All rights reserved.
"""Batch script: run ImageFilter + PCProjection for 50 trees, save only 1080x720
cropped images: segment tree (background black), then crop a fixed window centered
on the mask (no scaling). Does not modify ImageFilter.py or PC_projection.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Allow importing scripts/lib and repo-root ImageFilter / PC_projection.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.mask_ops import apply_mask_bgr, refine_mask  # noqa: E402
from lib.poses import resolve_pano_image_path  # noqa: E402
from ImageFilter import ImageFilter  # noqa: E402
from PC_projection import PCProjection  # noqa: E402


# Default paths (aligned with PC_projection.py)
DEFAULT_PANO_IMAGE_DIR = "/root/autodl-fs/222-pcimg-data/panoramicImage"
DEFAULT_PANO_POSES_CSV = "/root/autodl-fs/222-pcimg-data/panoramicPoses.csv"
DEFAULT_MAP_LAS = "/root/autodl-fs/222-pcimg-data/map2.las"
DEFAULT_TREE_LAS_DIR = "/root/Tree-Seg/tree-origin"
DEFAULT_OUTPUT_DIR = "/root/Tree-Seg/50trees-output"


def _crop_centered_on_mask(
    img: np.ndarray,
    mask: np.ndarray,
    crop_width: int,
    crop_height: int,
) -> Optional[np.ndarray]:
    """Crop a fixed crop_width x crop_height window centered on the mask. No scaling.

    If the window extends outside the image, the output is padded with black so the
    result is always crop_width x crop_height. Returns None if mask is empty.
    """
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    H, W = img.shape[:2]
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0
    x0 = int(cx - half_w)
    x1 = int(cx + half_w)
    y0 = int(cy - half_h)
    y1 = int(cy + half_h)

    # Intersection of window [x0,x1) x [y0,y1) with image [0,W) x [0,H)
    src_x0 = max(0, x0)
    src_x1 = min(W, x1)
    src_y0 = max(0, y0)
    src_y1 = min(H, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        # Window fully outside image
        out = np.zeros((crop_height, crop_width, 3), dtype=img.dtype)
        return out

    # Destination in the crop canvas (same size as window, origin at 0,0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out = np.zeros((crop_height, crop_width, 3), dtype=img.dtype)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def _process_one_tree(
    *,
    tree_las_path: Path,
    tree_name: str,
    pano_image_dir: Path,
    pano_poses_csv: Path,
    output_dir: Path,
    map_las: Optional[Path],
    max_dist_m: float,
    crop_width: int,
    crop_height: int,
    downsample_step: int,
    flip_v: bool,
    morph_kernel: int,
    dilate_iter: int,
    close_iter: int,
    occl_area_ratio_thr: float,
    tube_radius_m: float,
    tree_clearance_m: float,
    tree_bbox_margin_m: float,
    refine_mask_mode: str = "largest_contour",
    approx_epsilon_ratio: float = 0.005,
) -> Tuple[int, int]:
    """Run ImageFilter + PCProjection for one tree, then crop and save images. Returns (saved_count, skipped_count)."""
    import cv2

    # Temporary directory for this tree's masks (will be removed at the end)
    temp_dir = Path(tempfile.mkdtemp(prefix="trees_seg_batch_"))
    try:
        image_filter = ImageFilter(
            pano_image_dir=pano_image_dir,
            pano_poses_csv=pano_poses_csv,
            max_dist_m=max_dist_m,
        )
        selected_basenames = image_filter.filter_by_distance(
            tree_las=tree_las_path,
            output_dir=None,
        )
        if not selected_basenames:
            return 0, 0

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
        written, _discarded = proj.project(
            tree_las=tree_las_path,
            output_dir=temp_dir,
            selected_basenames=selected_basenames,
            write_params_json=False,
        )

        tree_out_dir = output_dir / tree_name
        tree_out_dir.mkdir(parents=True, exist_ok=True)
        masks_raw_dir = temp_dir / "masks_raw"
        saved = 0
        skipped = 0

        for basename in written:
            key = Path(basename).name
            stem = Path(basename).stem
            mask_path = masks_raw_dir / f"{stem}.png"
            if not mask_path.exists():
                skipped += 1
                continue
            img_path = resolve_pano_image_path(pano_image_dir, key)
            if img_path is None or not img_path.exists():
                skipped += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != img.shape[:2]:
                skipped += 1
                continue
            if refine_mask_mode != "none":
                mask = refine_mask(
                    mask,
                    mode=refine_mask_mode,
                    approx_epsilon_ratio=approx_epsilon_ratio,
                )
            segmented = apply_mask_bgr(img, mask)
            cropped = _crop_centered_on_mask(segmented, mask, crop_width, crop_height)
            if cropped is None:
                skipped += 1
                continue
            out_path = tree_out_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), cropped)
            saved += 1

        return saved, skipped
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _collect_las_paths(tree_las_dir: Path, max_trees: int) -> List[Path]:
    """Enumerate *.las in tree_las_dir, sort by name, return first max_trees."""
    paths = sorted(tree_las_dir.glob("*.las"), key=lambda p: p.name)
    return paths[:max_trees]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch: ImageFilter + PCProjection for 50 trees, save 1080x720 cropped images only.",
    )
    parser.add_argument(
        "--tree-las-dir",
        type=Path,
        default=Path(DEFAULT_TREE_LAS_DIR),
        help="Directory containing tree LAS files",
    )
    parser.add_argument(
        "--pano-image-dir",
        type=Path,
        default=Path(DEFAULT_PANO_IMAGE_DIR),
        help="Panorama image directory",
    )
    parser.add_argument(
        "--pano-poses-csv",
        type=Path,
        default=Path(DEFAULT_PANO_POSES_CSV),
        help="Panoramic poses CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output root directory (one subfolder per tree)",
    )
    parser.add_argument(
        "--map-las",
        type=Path,
        default=Path(DEFAULT_MAP_LAS),
        help="Map LAS for occlusion filtering",
    )
    parser.add_argument(
        "--max-trees",
        type=int,
        default=50,
        help="Maximum number of trees to process",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=1080,
        help="Output crop width in pixels",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=720,
        help="Output crop height in pixels",
    )
    parser.add_argument(
        "--max-dist-m",
        type=float,
        default=10.0,
        help="Max camera-to-tree distance for image selection",
    )
    parser.add_argument(
        "--downsample-step",
        type=int,
        default=50,
        help="Point cloud downsample step for projection",
    )
    parser.add_argument(
        "--flip-v",
        action="store_true",
        default=True,
        help="Flip v coordinate in projection",
    )
    parser.add_argument(
        "--no-flip-v",
        action="store_false",
        dest="flip_v",
        help="Disable flip_v",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=9,
        help="Morphology kernel size",
    )
    parser.add_argument(
        "--dilate-iter",
        type=int,
        default=2,
        help="Dilate iterations",
    )
    parser.add_argument(
        "--close-iter",
        type=int,
        default=2,
        help="Close iterations",
    )
    parser.add_argument(
        "--occl-area-ratio-thr",
        type=float,
        default=0.4,
        help="Occlusion area ratio threshold",
    )
    parser.add_argument(
        "--tube-radius-m",
        type=float,
        default=1.0,
        help="Ray tube radius (m)",
    )
    parser.add_argument(
        "--tree-clearance-m",
        type=float,
        default=0.5,
        help="Tree clearance along ray (m)",
    )
    parser.add_argument(
        "--tree-bbox-margin-m",
        type=float,
        default=0.5,
        help="Tree bbox margin (m)",
    )
    parser.add_argument(
        "--refine-mask",
        type=str,
        choices=["none", "largest_contour", "convex_hull", "approx"],
        default="largest_contour",
        help="Refine mask: largest_contour=fill largest only (closest to original), "
             "approx=polygon approx, convex_hull=convex fill (default: largest_contour)",
    )
    parser.add_argument(
        "--approx-epsilon-ratio",
        type=float,
        default=0.005,
        help="When --refine-mask approx: polygon approx strength (smaller=closer to contour, default 0.005)",
    )
    args = parser.parse_args()

    tree_las_dir = args.tree_las_dir
    output_dir = args.output_dir
    if not tree_las_dir.exists():
        print(f"Tree LAS directory does not exist: {tree_las_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    las_paths = _collect_las_paths(tree_las_dir, args.max_trees)
    if not las_paths:
        print(f"No LAS files found in {tree_las_dir}")
        sys.exit(0)

    print(f"Processing {len(las_paths)} trees from {tree_las_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Crop size: {args.crop_width}x{args.crop_height}")

    total_saved = 0
    total_skipped = 0
    for i, las_path in enumerate(las_paths):
        tree_name = las_path.stem
        print(f"[{i+1}/{len(las_paths)}] Tree: {tree_name}")
        saved, skipped = _process_one_tree(
            tree_las_path=las_path,
            tree_name=tree_name,
            pano_image_dir=args.pano_image_dir,
            pano_poses_csv=args.pano_poses_csv,
            output_dir=output_dir,
            map_las=args.map_las if args.map_las.exists() else None,
            max_dist_m=args.max_dist_m,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            downsample_step=args.downsample_step,
            flip_v=args.flip_v,
            morph_kernel=args.morph_kernel,
            dilate_iter=args.dilate_iter,
            close_iter=args.close_iter,
            occl_area_ratio_thr=args.occl_area_ratio_thr,
            tube_radius_m=args.tube_radius_m,
            tree_clearance_m=args.tree_clearance_m,
            tree_bbox_margin_m=args.tree_bbox_margin_m,
            refine_mask_mode=args.refine_mask,
            approx_epsilon_ratio=args.approx_epsilon_ratio,
        )
        total_saved += saved
        total_skipped += skipped
        print(f"  Saved {saved} crops, skipped {skipped}")

    print(f"Done. Total crops saved: {total_saved} (skipped: {total_skipped})")


if __name__ == "__main__":
    main()
