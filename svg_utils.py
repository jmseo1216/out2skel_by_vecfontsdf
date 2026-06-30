"""SVG 전처리 및 target segment 유틸리티.

좌표 규약:
- 원본 SVG는 svg_size x svg_size y-up 좌표계이다.
- 모델/이미지는 image_size x image_size y-down 좌표계이다.
- 변환식은 x_img = x_svg / svg_size * image_size,
  y_img = (svg_size - y_svg) / svg_size * image_size 이다.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]
Polyline = List[Point]

_COMMAND_RE = re.compile(r"([MmLlHhVvCcQqZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def svg_to_image_points(points: np.ndarray, svg_size: float = 50.0, image_size: int = 128) -> np.ndarray:
    """SVG y-up 좌표를 이미지 y-down 좌표로 변환한다."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[..., 0] = pts[..., 0] / svg_size * image_size
    pts[..., 1] = (svg_size - pts[..., 1]) / svg_size * image_size
    return pts


def _tokenize_path(d: str) -> List[str]:
    return [m.group(0) for m in _COMMAND_RE.finditer(d.replace(",", " "))]


def _is_command(tok: str) -> bool:
    return len(tok) == 1 and tok.isalpha()


def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, n: int) -> Polyline:
    pts: Polyline = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        pts.append((
            u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
        ))
    return pts


def _quadratic(p0: Point, p1: Point, p2: Point, n: int) -> Polyline:
    pts: Polyline = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        pts.append((
            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        ))
    return pts


def parse_svg_path(d: str, curve_samples: int = 24) -> List[Polyline]:
    """SVG path의 M/L/H/V/C/Q/Z 명령을 폴리라인 목록으로 변환한다.

    outline rasterization을 위해 C/Q 곡선은 균일 샘플링한 폴리라인으로 근사한다.
    """
    tokens = _tokenize_path(d)
    i = 0
    cmd = None
    cur: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    polylines: List[Polyline] = []
    current_poly: Polyline = []

    def read_float() -> float:
        nonlocal i
        if i >= len(tokens) or _is_command(tokens[i]):
            raise ValueError(f"숫자 토큰이 필요하지만 {tokens[i] if i < len(tokens) else 'EOF'} 를 만났습니다.")
        v = float(tokens[i])
        i += 1
        return v

    def add_point(p: Point) -> None:
        nonlocal current_poly
        if not current_poly:
            current_poly = [cur]
        current_poly.append(p)

    def finish_poly() -> None:
        nonlocal current_poly
        if len(current_poly) >= 2:
            polylines.append(current_poly)
        current_poly = []

    while i < len(tokens):
        if _is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        if cmd is None:
            raise ValueError("path가 명령 없이 시작되었습니다.")
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            finish_poly()
            x, y = read_float(), read_float()
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            start = cur
            current_poly = [cur]
            cmd = "l" if rel else "L"
        elif c == "L":
            while i < len(tokens) and not _is_command(tokens[i]):
                x, y = read_float(), read_float()
                p = (cur[0] + x, cur[1] + y) if rel else (x, y)
                add_point(p)
                cur = p
        elif c == "H":
            while i < len(tokens) and not _is_command(tokens[i]):
                x = read_float()
                p = (cur[0] + x, cur[1]) if rel else (x, cur[1])
                add_point(p)
                cur = p
        elif c == "V":
            while i < len(tokens) and not _is_command(tokens[i]):
                y = read_float()
                p = (cur[0], cur[1] + y) if rel else (cur[0], y)
                add_point(p)
                cur = p
        elif c == "C":
            while i < len(tokens) and not _is_command(tokens[i]):
                vals = [read_float() for _ in range(6)]
                p1, p2, p3 = (vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])
                if rel:
                    p1 = (cur[0] + p1[0], cur[1] + p1[1])
                    p2 = (cur[0] + p2[0], cur[1] + p2[1])
                    p3 = (cur[0] + p3[0], cur[1] + p3[1])
                for p in _cubic(cur, p1, p2, p3, curve_samples):
                    add_point(p)
                cur = p3
        elif c == "Q":
            while i < len(tokens) and not _is_command(tokens[i]):
                vals = [read_float() for _ in range(4)]
                p1, p2 = (vals[0], vals[1]), (vals[2], vals[3])
                if rel:
                    p1 = (cur[0] + p1[0], cur[1] + p1[1])
                    p2 = (cur[0] + p2[0], cur[1] + p2[1])
                for p in _quadratic(cur, p1, p2, curve_samples):
                    add_point(p)
                cur = p2
        elif c == "Z":
            add_point(start)
            cur = start
            finish_poly()
        else:
            raise ValueError(f"지원하지 않는 SVG path 명령입니다: {cmd}")
    finish_poly()
    return polylines


def load_svg_polylines(svg_path: str | Path, curve_samples: int = 24, image_size: int = 128, svg_size: float = 50.0) -> List[np.ndarray]:
    """SVG 파일에서 path/line/polyline/polygon을 읽고 이미지 좌표 폴리라인으로 반환한다."""
    root = ET.parse(svg_path).getroot()
    polylines: List[np.ndarray] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "path" and elem.get("d"):
            for poly in parse_svg_path(elem.get("d", ""), curve_samples):
                polylines.append(svg_to_image_points(np.asarray(poly, dtype=np.float32), svg_size, image_size))
        elif tag == "line":
            pts = np.array([[float(elem.get("x1", 0)), float(elem.get("y1", 0))], [float(elem.get("x2", 0)), float(elem.get("y2", 0))]], dtype=np.float32)
            polylines.append(svg_to_image_points(pts, svg_size, image_size))
        elif tag in {"polyline", "polygon"} and elem.get("points"):
            nums = [float(v) for v in _NUMBER_RE.findall(elem.get("points", ""))]
            pts = np.asarray(list(zip(nums[0::2], nums[1::2])), dtype=np.float32)
            if tag == "polygon" and len(pts) > 0:
                pts = np.vstack([pts, pts[0]])
            polylines.append(svg_to_image_points(pts, svg_size, image_size))
    return [p for p in polylines if len(p) >= 2]


def _segments_from_polylines(polylines: Sequence[np.ndarray], min_length: float = 1e-6) -> np.ndarray:
    segs: List[np.ndarray] = []
    for poly in polylines:
        for a, b in zip(poly[:-1], poly[1:]):
            seg = np.concatenate([a, b]).astype(np.float32)
            if float(np.linalg.norm(seg[2:4] - seg[0:2])) > min_length:
                segs.append(seg)
    if not segs:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack(segs).astype(np.float32)


def extract_line_segments_from_svg(svg_path: str | Path, image_size: int = 128, svg_size: float = 50.0, min_length: float = 1e-6) -> np.ndarray:
    """skeleton SVG에서 실제 target line segment를 [T,4] 이미지 좌표로 추출한다."""
    polylines = load_svg_polylines(svg_path, curve_samples=1, image_size=image_size, svg_size=svg_size)
    return _segments_from_polylines(polylines, min_length=min_length)


def segment_lengths(segments: np.ndarray) -> np.ndarray:
    """[N,4] segment 배열의 길이를 반환한다."""
    if len(segments) == 0:
        return np.zeros((0,), dtype=np.float32)
    segs = np.asarray(segments, dtype=np.float32)
    return np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1).astype(np.float32)


def total_segment_length(segments: np.ndarray) -> float:
    """segment 전체 길이 합을 float로 반환한다."""
    return float(segment_lengths(segments).sum())


def pad_segments(segments: np.ndarray, max_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """target segment를 [K,4]로 padding하고 실제 segment mask [K]를 반환한다."""
    n = int(len(segments))
    if n == 0:
        raise RuntimeError("target skeleton segment가 0개입니다. skeleton SVG를 확인하세요.")
    if n > max_segments:
        raise RuntimeError(f"target segment 수({n})가 max_segments({max_segments})보다 큽니다.")
    padded = np.zeros((max_segments, 4), dtype=np.float32)
    mask = np.zeros((max_segments,), dtype=bool)
    padded[:n] = np.asarray(segments, dtype=np.float32)
    mask[:n] = True
    return padded, mask


def rasterize_outline(polylines: Sequence[np.ndarray], image_size: int = 128) -> np.ndarray:
    """even-odd fill로 outline을 래스터화하여 [1,H,W] 흰 배경/검은 글리프 이미지를 만든다."""
    fill = np.zeros((image_size, image_size), dtype=np.uint8)
    for poly in polylines:
        pts = [tuple(map(float, p)) for p in poly]
        if len(pts) >= 3:
            mask_img = Image.new("L", (image_size, image_size), 0)
            ImageDraw.Draw(mask_img).polygon(pts, fill=1)
            # even-odd 규칙: 각 contour mask를 XOR하여 내부 hole을 흰색으로 유지한다.
            fill ^= np.asarray(mask_img, dtype=np.uint8)
        elif len(pts) == 2:
            mask_img = Image.new("L", (image_size, image_size), 0)
            ImageDraw.Draw(mask_img).line(pts, fill=1, width=1)
            fill |= np.asarray(mask_img, dtype=np.uint8)
    img = np.where(fill > 0, 0.0, 1.0).astype(np.float32)
    return img[None, ...]


def rasterize_skeleton(polylines: Sequence[np.ndarray], image_size: int = 128, line_width: int = 2) -> np.ndarray:
    """골격 선을 흰 선/검은 배경의 [H,W] 마스크로 래스터화한다."""
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)
    for poly in polylines:
        draw.line([tuple(map(float, p)) for p in poly], fill=255, width=line_width, joint="curve")
    return np.asarray(img, dtype=np.float32) / 255.0


def sample_polyline_points(polylines: Sequence[np.ndarray], num_points: int, jitter: float = 0.0, seed: int | None = None) -> np.ndarray:
    """여러 폴리라인에서 전체 arc length 기준으로 균일 샘플링한 [M,2] 점을 반환한다."""
    segments = []
    lengths = []
    for poly in polylines:
        for a, b in zip(poly[:-1], poly[1:]):
            length = float(np.linalg.norm(b - a))
            if length > 1e-6:
                segments.append((a.astype(np.float32), b.astype(np.float32)))
                lengths.append(length)
    if not segments:
        return np.zeros((num_points, 2), dtype=np.float32)
    cum = np.cumsum(np.asarray(lengths, dtype=np.float32))
    total = float(cum[-1])
    targets = (np.arange(num_points, dtype=np.float32) + 0.5) / num_points * total
    pts = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right"))
        prev = 0.0 if idx == 0 else float(cum[idx - 1])
        a, b = segments[idx]
        alpha = (float(t) - prev) / lengths[idx]
        pts.append(a + alpha * (b - a))
    out = np.stack(pts).astype(np.float32)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        out += rng.normal(0.0, jitter, out.shape).astype(np.float32)
    return out
