#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缺失的树木分割输出：
  - 21 棵缺失（无子文件夹）：因距离阈值过小 → 逐步增大距离阈值重试
  - 3 棵空文件夹（24, 46, 50）：因遮挡筛选 → 关闭遮挡筛选重试
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set

# 复用 trees-seg-batch.py 中的处理函数（文件名含连字符，需用 importlib）
import importlib.util

_REPO_DIR = Path(__file__).resolve().parent
_BATCH_SCRIPT = _REPO_DIR / "trees-seg-batch.py"
_spec = importlib.util.spec_from_file_location("trees_seg_batch", _BATCH_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_mod)  # type: ignore

_process_one_tree = _mod._process_one_tree
DEFAULT_PANO_IMAGE_DIR = _mod.DEFAULT_PANO_IMAGE_DIR
DEFAULT_PANO_POSES_CSV = _mod.DEFAULT_PANO_POSES_CSV
DEFAULT_MAP_LAS = _mod.DEFAULT_MAP_LAS
DEFAULT_TREE_LAS_DIR = _mod.DEFAULT_TREE_LAS_DIR
DEFAULT_OUTPUT_DIR = _mod.DEFAULT_OUTPUT_DIR

# ─── 参数配置 ────────────────────────────────────────
TREE_LAS_DIR = Path(DEFAULT_TREE_LAS_DIR)
OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
PANO_IMAGE_DIR = Path(DEFAULT_PANO_IMAGE_DIR)
PANO_POSES_CSV = Path(DEFAULT_PANO_POSES_CSV)
MAP_LAS = Path(DEFAULT_MAP_LAS)
DISCARDED_DIR = _REPO_DIR / "temp"

BASE_DIST_M = 10.0
MAX_EXTRA_DIST_M = 20
CROP_WIDTH = 1080
CROP_HEIGHT = 720

COMMON_KWARGS = dict(
    pano_image_dir=PANO_IMAGE_DIR,
    pano_poses_csv=PANO_POSES_CSV,
    discarded_dir=DISCARDED_DIR,
    crop_width=CROP_WIDTH,
    crop_height=CROP_HEIGHT,
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


def _find_las(tree_id: int) -> Path:
    """根据树 ID 找到对应 .las 文件。"""
    candidates = sorted(TREE_LAS_DIR.glob("*.las"), key=lambda p: p.name)
    for p in candidates:
        if p.stem == str(tree_id):
            return p
    raise FileNotFoundError(f"未找到树 {tree_id} 的 LAS 文件")


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() == ".png")


def diagnose() -> tuple[List[int], List[int]]:
    """诊断问题树，返回 (距离筛选失败列表, 遮挡筛选失败列表)。"""
    all_ids = sorted(int(p.stem) for p in TREE_LAS_DIR.glob("*.las"))
    missing_ids: List[int] = []
    empty_ids: List[int] = []

    for tid in all_ids:
        folder = OUTPUT_DIR / str(tid)
        if not folder.exists():
            missing_ids.append(tid)
        elif _count_images(folder) == 0:
            empty_ids.append(tid)

    return missing_ids, empty_ids


def retry_distance(tree_ids: List[int]) -> None:
    """对距离筛选失败的树逐步增大距离阈值重试。"""
    for tid in tree_ids:
        las_path = _find_las(tid)
        print(f"\n{'='*60}")
        print(f"[距离重试] 树 {tid}：基准 {BASE_DIST_M}m 内无图，开始递增重试")

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
                print(f"  [成功] max_dist={new_dist}m (+{delta}m) → 保存 {saved} 张，输出: {out_dir}")
                success = True
                break
            else:
                # 清理空文件夹
                if out_dir.exists() and _count_images(out_dir) == 0:
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
                print(f"  [尝试] max_dist={new_dist}m (+{delta}m) → 仍无图片")

        if not success:
            print(f"  [失败] 树 {tid}：增大到 {BASE_DIST_M + MAX_EXTRA_DIST_M}m 仍无法生成图片")


def retry_no_filter(tree_ids: List[int]) -> None:
    """对遮挡筛选失败的树关闭遮挡筛选重试。"""
    for tid in tree_ids:
        las_path = _find_las(tid)
        folder_name = f"{tid}_no_filter"
        out_dir = OUTPUT_DIR / folder_name

        print(f"\n{'='*60}")
        print(f"[关闭遮挡] 树 {tid}：禁用遮挡筛选重试")

        saved, skipped = _process_one_tree(
            tree_las_path=las_path,
            tree_name=folder_name,
            output_dir=OUTPUT_DIR,
            map_las=None,
            max_dist_m=BASE_DIST_M,
            **COMMON_KWARGS,
        )

        if saved > 0:
            print(f"  [成功] 关闭遮挡 → 保存 {saved} 张，输出: {out_dir}")
        else:
            print(f"  [仍为空] 关闭遮挡后仍无图片，回退到距离递增策略")
            # 清理空文件夹
            if out_dir.exists() and _count_images(out_dir) == 0:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
            retry_distance([tid])


def main() -> None:
    print("="*60)
    print("树木分割修复脚本")
    print("="*60)

    missing_ids, empty_ids = diagnose()

    print(f"\n诊断结果：")
    print(f"  距离筛选失败（无子文件夹）：{len(missing_ids)} 棵 → {missing_ids}")
    print(f"  遮挡筛选失败（空文件夹）：  {len(empty_ids)} 棵 → {empty_ids}")

    if not missing_ids and not empty_ids:
        print("\n所有树木均已成功分割，无需修复。")
        return

    # 先处理遮挡筛选失败的树（通常较少）
    if empty_ids:
        print(f"\n--- 第一阶段：关闭遮挡筛选重试 ({len(empty_ids)} 棵) ---")
        retry_no_filter(empty_ids)

    # 再处理距离筛选失败的树
    if missing_ids:
        print(f"\n--- 第二阶段：距离阈值递增重试 ({len(missing_ids)} 棵) ---")
        retry_distance(missing_ids)

    # 汇总
    print(f"\n{'='*60}")
    print("修复完成，最终状态：")
    all_folders = sorted(OUTPUT_DIR.iterdir())
    total_with_images = 0
    for f in all_folders:
        if f.is_dir():
            cnt = _count_images(f)
            if cnt > 0:
                total_with_images += 1
    print(f"  {OUTPUT_DIR} 中共 {total_with_images} 个有图片的子文件夹")


if __name__ == "__main__":
    main()
