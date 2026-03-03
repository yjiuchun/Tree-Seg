#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""处理剩余未修复的 6 棵树（30, 32, 40, 47, 48, 49）。"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
_BATCH_SCRIPT = _REPO_DIR / "trees-seg-batch.py"
_spec = importlib.util.spec_from_file_location("trees_seg_batch", _BATCH_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_process_one_tree = _mod._process_one_tree

TREE_LAS_DIR = Path(_mod.DEFAULT_TREE_LAS_DIR)
OUTPUT_DIR = Path(_mod.DEFAULT_OUTPUT_DIR)
PANO_IMAGE_DIR = Path(_mod.DEFAULT_PANO_IMAGE_DIR)
PANO_POSES_CSV = Path(_mod.DEFAULT_PANO_POSES_CSV)
MAP_LAS = Path(_mod.DEFAULT_MAP_LAS)
DISCARDED_DIR = _REPO_DIR / "temp"

BASE_DIST_M = 10.0
MAX_EXTRA_DIST_M = 20

COMMON_KWARGS = dict(
    pano_image_dir=PANO_IMAGE_DIR,
    pano_poses_csv=PANO_POSES_CSV,
    discarded_dir=DISCARDED_DIR,
    crop_width=1080,
    crop_height=720,
    downsample_step=50,
    flip_v=True,
    morph_kernel=9,
    dilate_iter=2,
    close_iter=2,
    occl_area_ratio_thr=0.4,
    tube_radius_m=1.0,
    tree_clearance_m=0.5,
    tree_bbox_margin_m=0.5,
    refine_mask_mode="largest_contour",
    approx_epsilon_ratio=0.005,
)

REMAINING_IDS = [30, 32, 40, 47, 48, 49]


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() == ".png")


def main():
    import shutil

    for tid in REMAINING_IDS:
        las_path = TREE_LAS_DIR / f"{tid}.las"
        if not las_path.exists():
            print(f"[跳过] 树 {tid}：LAS 文件不存在")
            continue

        print(f"\n{'='*60}")
        print(f"[距离重试] 树 {tid}")

        success = False
        for delta in range(1, MAX_EXTRA_DIST_M + 1):
            new_dist = BASE_DIST_M + delta
            folder_name = f"{tid}_+{delta}m"
            out_dir = OUTPUT_DIR / folder_name

            saved, skipped = _process_one_tree(
                tree_las_path=las_path,
                tree_name=folder_name,
                output_dir=OUTPUT_DIR,
                map_las=MAP_LAS if MAP_LAS.exists() else None,
                max_dist_m=new_dist,
                **COMMON_KWARGS,
            )

            if saved > 0:
                print(f"  [成功] max_dist={new_dist}m (+{delta}m) -> 保存 {saved} 张")
                success = True
                break
            else:
                if out_dir.exists() and _count_images(out_dir) == 0:
                    shutil.rmtree(out_dir, ignore_errors=True)
                print(f"  [尝试] max_dist={new_dist}m (+{delta}m) -> 无图片")

        if not success:
            print(f"  [失败] 树 {tid}：+20m 仍无法生成图片")

    print("\n完成！")


if __name__ == "__main__":
    main()
