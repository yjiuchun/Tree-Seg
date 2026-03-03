#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理剩余 6 棵树（30, 32, 40, 47, 48, 49）。
策略：先关闭遮挡筛选 + 逐步增大距离阈值，直到产生图片。
"""

from __future__ import annotations

import shutil
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
    for tid in REMAINING_IDS:
        las_path = TREE_LAS_DIR / f"{tid}.las"
        if not las_path.exists():
            print(f"[跳过] 树 {tid}：LAS 文件不存在")
            continue

        print(f"\n{'='*60}")
        print(f"树 {tid}：关闭遮挡 + 距离递增重试")

        success = False
        for delta in range(0, MAX_EXTRA_DIST_M + 1):
            new_dist = BASE_DIST_M + delta
            if delta == 0:
                folder_name = f"{tid}_no_filter"
            else:
                folder_name = f"{tid}_no_filter_+{delta}m"
            out_dir = OUTPUT_DIR / folder_name

            saved, skipped = _process_one_tree(
                tree_las_path=las_path,
                tree_name=folder_name,
                output_dir=OUTPUT_DIR,
                map_las=None,
                max_dist_m=new_dist,
                **COMMON_KWARGS,
            )

            if saved > 0:
                print(f"  [成功] dist={new_dist}m, no_filter -> 保存 {saved} 张, 输出: {out_dir}")
                success = True
                break
            else:
                if out_dir.exists() and _count_images(out_dir) == 0:
                    shutil.rmtree(out_dir, ignore_errors=True)
                print(f"  [尝试] dist={new_dist}m, no_filter -> 无图片")

        if not success:
            print(f"  [失败] 树 {tid}：所有策略均失败")

    print("\n完成！")


if __name__ == "__main__":
    main()
