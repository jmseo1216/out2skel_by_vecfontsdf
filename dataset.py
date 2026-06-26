"""폰트 outline-to-skeleton 학습용 Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from svg_utils import load_svg_polylines, rasterize_outline, rasterize_skeleton, sample_polyline_points


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """골격 마스크에서 각 픽셀까지의 거리 변환을 계산한다."""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(mask <= 0.5).astype(np.float32)
    except Exception:
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) == 0:
            return np.full(mask.shape, max(mask.shape), dtype=np.float32)
        yy, xx = np.indices(mask.shape)
        d2 = (xx[..., None] - xs[None, None, :]) ** 2 + (yy[..., None] - ys[None, None, :]) ** 2
        return np.sqrt(d2.min(axis=-1)).astype(np.float32)


class FontSkeletonDataset(Dataset):
    """33.svg~126.svg 같은 paired SVG 디렉터리를 읽는 Dataset."""

    def __init__(
        self,
        outline_dir: str | Path,
        skeleton_dir: str | Path,
        codepoints: Optional[List[int]] = None,
        image_size: int = 128,
        num_target_points: int = 256,
        sigma: float = 2.0,
        curve_samples: int = 24,
        skeleton_line_width: int = 2,
        point_jitter: float = 0.0,
    ) -> None:
        self.outline_dir = Path(outline_dir)
        self.skeleton_dir = Path(skeleton_dir)
        self.codepoints = codepoints if codepoints is not None else list(range(33, 127))
        self.image_size = image_size
        self.num_target_points = num_target_points
        self.sigma = sigma
        self.curve_samples = curve_samples
        self.skeleton_line_width = skeleton_line_width
        self.point_jitter = point_jitter

    def __len__(self) -> int:
        return len(self.codepoints)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        codepoint = int(self.codepoints[idx])
        outline_polys = load_svg_polylines(self.outline_dir / f"{codepoint}.svg", self.curve_samples, self.image_size)
        skeleton_polys = load_svg_polylines(self.skeleton_dir / f"{codepoint}.svg", self.curve_samples, self.image_size)

        outline_image = rasterize_outline(outline_polys, self.image_size)
        skeleton_mask = rasterize_skeleton(skeleton_polys, self.image_size, self.skeleton_line_width)
        target_dist = compute_distance_transform(skeleton_mask)
        target_heatmap = np.exp(-(target_dist ** 2) / (2.0 * self.sigma ** 2)).astype(np.float32)
        target_points = sample_polyline_points(skeleton_polys, self.num_target_points, self.point_jitter, seed=codepoint)

        return {
            "input_image": torch.from_numpy(outline_image).float(),
            "target_dist": torch.from_numpy(target_dist).float(),
            "target_heatmap": torch.from_numpy(target_heatmap).float(),
            "target_sample_points": torch.from_numpy(target_points).float(),
            "target_skeleton": torch.from_numpy(skeleton_mask).float(),
            "codepoint": torch.tensor(codepoint, dtype=torch.long),
        }
