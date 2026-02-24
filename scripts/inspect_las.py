#!/usr/bin/env python3
"""Inspect LAS file dimensions and scalar fields."""
import sys
from pathlib import Path

import laspy

def main():
    p = Path("/root/Tree-Seg/1-100.las")
    las = laspy.read(str(p))
    print("Point count:", len(las.x))
    print("Point format id:", las.header.point_format.id)
    for name in ["classification", "user_data", "point_source_id", "gps_time", "red", "green", "blue"]:
        if hasattr(las, name):
            arr = getattr(las, name)
            if hasattr(arr, "__len__") and len(arr) == len(las.x):
                uniq = len(set(arr))
                try:
                    print(f"  {name}: unique={uniq}, min={arr.min()}, max={arr.max()}")
                except Exception:
                    print(f"  {name}: unique={uniq}")
    if hasattr(las.point_format, "extra_dimensions"):
        for d in las.point_format.extra_dimensions:
            print("  extra:", d.name)

if __name__ == "__main__":
    main()
