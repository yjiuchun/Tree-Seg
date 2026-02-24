#!/usr/bin/env python3
"""
按标量场（Scalar Field）分割 LAS 点云，将每个唯一值对应的点保存为单独的 LAS 文件。
默认使用 TreeInstance 字段分割 100 棵树木并保存到指定文件夹。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import laspy


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="按标量场分割 LAS，每个唯一值输出一个 LAS 文件到文件夹"
    )
    p.add_argument(
        "input_las",
        type=Path,
        help="输入 LAS 路径（如 1-100.las）",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("trees_out"),
        help="分割后的 LAS 保存目录（默认: trees_out）",
    )
    p.add_argument(
        "--scalar-field",
        type=str,
        default="TreeInstance",
        help="用于分割的标量字段名（默认: TreeInstance）",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="tree",
        help="输出文件名前缀（默认: tree -> tree_001.las）",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["las", "laz"],
        default="las",
        help="输出格式 las 或 laz（默认: las）",
    )
    return p


def main() -> None:
    args = get_parser().parse_args()
    input_path = args.input_las
    out_dir = args.out_dir
    scalar_name = args.scalar_field
    prefix = args.prefix
    ext = f".{args.format}"

    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    las = laspy.read(str(input_path))
    if not hasattr(las, scalar_name):
        available = [d for d in dir(las) if not d.startswith("_") and d not in ("header", "point_format", "points")]
        raise SystemExit(
            f"标量字段 '{scalar_name}' 不存在。可用字段示例: {available}"
        )

    scalar = getattr(las, scalar_name)
    # 支持整型或浮点型标量，转为可排序唯一值
    unique_vals = sorted(set(scalar))

    print(f"输入: {input_path}, 点数: {len(las.x)}")
    print(f"标量字段: {scalar_name}, 唯一值数量: {len(unique_vals)}")
    print(f"输出目录: {out_dir}, 格式: {ext}")

    for i, val in enumerate(unique_vals):
        mask = scalar == val
        n = int(mask.sum())
        if n == 0:
            continue
        sub = laspy.create(
            point_format=las.header.point_format,
            file_version=las.header.version,
        )
        sub.points = las.points[mask]
        # 输出文件名：按顺序编号，若值为整数则也可用该值
        if isinstance(val, float) and val == int(val):
            val = int(val)
        out_name = f"{prefix}_{i+1:03d}{ext}"
        out_path = out_dir / out_name
        sub.write(str(out_path))
        print(f"  {out_name}: {n} 点 (scalar={val})")

    print(f"完成: 共 {len(unique_vals)} 个文件 -> {out_dir}")


if __name__ == "__main__":
    main()
