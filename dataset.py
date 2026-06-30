"""Segment matching 기반 outline-to-skeleton Dataset."""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from svg_utils import (
    extract_line_segments_from_svg,
    load_svg_polylines,
    pad_segments,
    rasterize_outline,
    rasterize_skeleton,
    sample_polyline_points,
    total_segment_length,
)


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


def extract_codepoint_from_filename(filename: str) -> int:
    """파일 이름의 마지막 숫자를 codepoint로 사용한다."""
    numbers = re.findall(r"\d+", Path(filename).stem)
    if not numbers:
        raise RuntimeError(f"파일 이름에서 codepoint 숫자를 찾을 수 없습니다: {filename}")
    return int(numbers[-1])


class FontSkeletonDataset(Dataset):
    """outline_dir와 skeleton_dir에서 같은 파일 이름의 SVG pair를 scan하는 Dataset."""

    def __init__(
        self,
        outline_dir: str | Path,
        skeleton_dir: str | Path,
        max_segments: int = 48,
        codepoints: Optional[List[int]] = None,
        filenames: Optional[List[str]] = None,
        image_size: int = 128,
        svg_size: float = 50.0,
        num_target_points: int = 1000,
        sigma: float = 2.0,
        curve_samples: int = 24,
        skeleton_line_width: int = 2,
        point_jitter: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.outline_dir = Path(outline_dir)
        self.skeleton_dir = Path(skeleton_dir)
        self.max_segments = max_segments
        self.image_size = image_size
        self.svg_size = svg_size
        self.num_target_points = num_target_points
        self.sigma = sigma
        self.curve_samples = curve_samples
        self.skeleton_line_width = skeleton_line_width
        self.point_jitter = point_jitter
        self.samples = self._scan_pairs(codepoints=codepoints, filenames=filenames, verbose=verbose)
        if not self.samples:
            raise RuntimeError(f"paired SVG 샘플이 없습니다: outline_dir={self.outline_dir}, skeleton_dir={self.skeleton_dir}")

    def _scan_pairs(self, codepoints: Optional[List[int]], filenames: Optional[List[str]], verbose: bool) -> List[dict]:
        outline_files = sorted(self.outline_dir.glob("*.svg"))
        filename_filter = set(filenames) if filenames is not None else None
        codepoint_filter = set(codepoints) if codepoints is not None else None
        samples: List[dict] = []
        missing: List[str] = []
        for outline_path in outline_files:
            name = outline_path.name
            if filename_filter is not None and name not in filename_filter:
                continue
            codepoint = extract_codepoint_from_filename(name)
            if codepoint_filter is not None and codepoint not in codepoint_filter:
                continue
            skeleton_path = self.skeleton_dir / name
            if not skeleton_path.exists():
                missing.append(name)
                continue
            samples.append({"outline_path": outline_path, "skeleton_path": skeleton_path, "filename": name, "codepoint": codepoint})
        if verbose:
            print(f"paired SVG samples: {len(samples)}")
            if missing:
                warnings.warn(f"skeleton pair가 없는 outline SVG {len(missing)}개를 건너뜁니다. 예: {missing[:5]}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_target_segment_count(self, idx: int) -> int:
        """학습 시작 전 통계 출력을 위해 target segment 수만 빠르게 계산한다."""
        sample = self.samples[idx]
        segs = extract_line_segments_from_svg(sample["skeleton_path"], self.image_size, self.svg_size)
        return int(len(segs))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        outline_polys = load_svg_polylines(sample["outline_path"], self.curve_samples, self.image_size, self.svg_size)
        skeleton_polys = load_svg_polylines(sample["skeleton_path"], self.curve_samples, self.image_size, self.svg_size)
        target_segments = extract_line_segments_from_svg(sample["skeleton_path"], self.image_size, self.svg_size)
        if len(target_segments) == 0:
            raise RuntimeError(f"target segment가 0개입니다: {sample['filename']}")
        if len(target_segments) > self.max_segments:
            raise RuntimeError(f"target segment 수({len(target_segments)})가 K={self.max_segments}보다 큽니다: {sample['filename']}")
        target_segments_padded, target_segment_mask = pad_segments(target_segments, self.max_segments)
        target_total_length = total_segment_length(target_segments)

        outline_image = rasterize_outline(outline_polys, self.image_size)
        skeleton_mask = rasterize_skeleton(skeleton_polys, self.image_size, self.skeleton_line_width)
        target_dist = compute_distance_transform(skeleton_mask)
        target_heatmap = np.exp(-(target_dist ** 2) / (2.0 * self.sigma ** 2)).astype(np.float32)
        target_points = sample_polyline_points(skeleton_polys, self.num_target_points, self.point_jitter, seed=sample["codepoint"] + idx)

        return {
            "input_image": torch.from_numpy(outline_image).float(),
            "target_segments": torch.from_numpy(target_segments_padded).float(),
            "target_segment_mask": torch.from_numpy(target_segment_mask).bool(),
            "target_num_segments": torch.tensor(float(len(target_segments)), dtype=torch.float32),
            "target_total_length": torch.tensor(float(target_total_length), dtype=torch.float32),
            "target_dist": torch.from_numpy(target_dist).float(),
            "target_heatmap": torch.from_numpy(target_heatmap).float(),
            "target_sample_points": torch.from_numpy(target_points).float(),
            "target_skeleton": torch.from_numpy(skeleton_mask).float(),
            "codepoint": torch.tensor(sample["codepoint"], dtype=torch.long),
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "filename": sample["filename"],
        }
