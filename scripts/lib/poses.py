from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PanoPose:
    ts: float
    imgname: str
    cam_xyz: np.ndarray  # (3,)
    q_c2w: np.ndarray  # (4,) as (qw,qx,qy,qz)

    @property
    def basename(self) -> str:
        return Path(self.imgname).name


def iter_panoramic_poses(csv_path: Path) -> Iterator[PanoPose]:
    """
    Parse panoramicPoses.csv.

    Expected row format (space-delimited):
      timestamp imgname x y z qx qy qz qw
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                ts = float(parts[0])
                imgname = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            except (ValueError, IndexError):
                continue

            cam_xyz = np.array([x, y, z], dtype=np.float64)
            q_c2w = np.array([qw, qx, qy, qz], dtype=np.float64)
            yield PanoPose(ts=ts, imgname=imgname, cam_xyz=cam_xyz, q_c2w=q_c2w)


def build_pose_index_by_basename(csv_path: Path) -> Dict[str, PanoPose]:
    """
    Build an index: pano_image_basename -> pose.

    If duplicates exist, keep the pose with smallest timestamp difference to its parsed
    basename timestamp when possible; otherwise keep the first occurrence.
    """
    index: Dict[str, PanoPose] = {}
    for pose in iter_panoramic_poses(csv_path):
        key = pose.basename
        if key not in index:
            index[key] = pose
            continue
        # Prefer closer timestamp if basename looks like a float timestamp.
        prev = index[key]
        t_name = _try_parse_ts_from_pano_basename(key)
        if t_name is None:
            continue
        if abs(pose.ts - t_name) < abs(prev.ts - t_name):
            index[key] = pose
    return index


def _try_parse_ts_from_pano_basename(name: str) -> Optional[float]:
    # examples: "1764058654.586835.jpg", "1764057659.501007(1).jpg"
    stem = Path(name).stem
    base = stem.split("(")[0].strip()
    try:
        return float(base)
    except ValueError:
        return None


def resolve_pano_image_path(pano_image_dir: Path, imgname_or_basename: str) -> Optional[Path]:
    """
    Resolve pano image path given directory and imgname.
    imgname may include subdir like 'panoramicImage/xxx.jpg' or just 'xxx.jpg'.
    """
    name = Path(imgname_or_basename).name
    cand = pano_image_dir / name
    if cand.exists():
        return cand
    # allow direct path
    p = Path(imgname_or_basename)
    if p.exists():
        return p
    return None

