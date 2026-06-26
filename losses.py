"""VecFontSDF-inspired segment prediction losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    image_size: int = 128
    render_sigma: float = 2.0
    samples_per_segment: int = 16
    length_min: float = 2.0
    w_render: float = 1.0
    w_pred_to_target: float = 1.0
    w_target_to_pred: float = 1.0
    w_exist: float = 0.01
    w_length: float = 0.1
    inactive_distance: float = 128.0
    eps: float = 1e-6


def point_to_segment_distance(points: torch.Tensor, segments: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """점과 선분 거리. points=[B,M,2], segments=[B,K,4], 반환=[B,M,K]."""
    q = points[:, :, None, :]
    a = segments[:, None, :, 0:2]
    b = segments[:, None, :, 2:4]
    ab = b - a
    aq = q - a
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(eps)
    t = (aq * ab).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = a + t * ab
    return torch.sqrt(((q - closest) ** 2).sum(dim=-1).clamp_min(eps))


def sample_predicted_segments(segments: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
    """예측 선분 위의 균일 샘플 점을 만든다. segments=[B,K,4], 반환=[B,K,S,2]."""
    t = torch.linspace(0.0, 1.0, samples_per_segment, device=segments.device, dtype=segments.dtype)
    a = segments[..., 0:2]
    b = segments[..., 2:4]
    return a[:, :, None, :] * (1.0 - t[None, None, :, None]) + b[:, :, None, :] * t[None, None, :, None]


def pixel_to_grid(points: torch.Tensor, image_size: int) -> torch.Tensor:
    """픽셀 좌표 [0,image_size]를 grid_sample용 [-1,1] 좌표로 변환한다."""
    return points / float(image_size) * 2.0 - 1.0


def loss_pred_to_target(pred_segments: torch.Tensor, exist_prob: torch.Tensor, target_dist: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """예측 선분 샘플 위치에서 target distance를 bilinear sampling한다."""
    bsz, k, _ = pred_segments.shape
    pts = sample_predicted_segments(pred_segments, cfg.samples_per_segment).reshape(bsz, k * cfg.samples_per_segment, 2)
    grid = pixel_to_grid(pts, cfg.image_size).view(bsz, k * cfg.samples_per_segment, 1, 2)
    sampled = F.grid_sample(target_dist[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.view(bsz, k, cfg.samples_per_segment)
    weights = exist_prob[:, :, None]
    return (sampled * weights).sum() / (weights.sum() * cfg.samples_per_segment + cfg.eps)


def loss_target_to_pred(pred_segments: torch.Tensor, exist_prob: torch.Tensor, target_points: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """target sample point가 활성 예측 선분으로 덮이도록 하는 Chamfer 방향 loss."""
    dist = point_to_segment_distance(target_points, pred_segments, cfg.eps)
    # 비활성 선분은 큰 penalty를 더해 min 선택에서 자연스럽게 제외한다.
    inactive_penalty = (1.0 - exist_prob)[:, None, :] * cfg.inactive_distance
    min_dist = (dist + inactive_penalty).min(dim=-1).values
    return min_dist.mean()


def make_pixel_grid(batch_size: int, image_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """픽셀 중심 좌표 grid [B,H*W,2]를 생성한다."""
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=dtype) + 0.5,
        torch.arange(image_size, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1).view(1, image_size * image_size, 2)
    return grid.expand(batch_size, -1, -1)


def soft_render_segments(pred_segments: torch.Tensor, exist_prob: torch.Tensor, image_size: int = 128, sigma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """예측 선분을 differentiable heatmap [B,H,W]로 soft rasterize한다."""
    bsz = pred_segments.shape[0]
    grid = make_pixel_grid(bsz, image_size, pred_segments.device, pred_segments.dtype)
    dist = point_to_segment_distance(grid, pred_segments, eps)
    inactive_penalty = (1.0 - exist_prob)[:, None, :] * float(image_size)
    pred_dist = (dist + inactive_penalty).min(dim=-1).values.view(bsz, image_size, image_size)
    return torch.exp(-(pred_dist ** 2) / (2.0 * sigma ** 2))


def loss_render(pred_segments: torch.Tensor, exist_prob: torch.Tensor, target_heatmap: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    pred_heatmap = soft_render_segments(pred_segments, exist_prob, cfg.image_size, cfg.render_sigma, cfg.eps)
    return F.mse_loss(pred_heatmap, target_heatmap)


def loss_exist_sparse(exist_prob: torch.Tensor) -> torch.Tensor:
    return exist_prob.mean()


def loss_length(pred_segments: torch.Tensor, exist_prob: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    lengths = torch.sqrt(((pred_segments[..., 2:4] - pred_segments[..., 0:2]) ** 2).sum(dim=-1) + cfg.eps)
    return (exist_prob * F.relu(cfg.length_min - lengths)).mean()


def compute_losses(pred_exist_logits: torch.Tensor, pred_segments: torch.Tensor, batch: Dict[str, torch.Tensor], cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """다섯 loss와 가중합 total loss를 dictionary로 반환한다."""
    exist_prob = torch.sigmoid(pred_exist_logits)
    l_pred = loss_pred_to_target(pred_segments, exist_prob, batch["target_dist"], cfg)
    l_tgt = loss_target_to_pred(pred_segments, exist_prob, batch["target_sample_points"], cfg)
    l_r = loss_render(pred_segments, exist_prob, batch["target_heatmap"], cfg)
    l_e = loss_exist_sparse(exist_prob)
    l_len = loss_length(pred_segments, exist_prob, cfg)
    total = cfg.w_render * l_r + cfg.w_pred_to_target * l_pred + cfg.w_target_to_pred * l_tgt + cfg.w_exist * l_e + cfg.w_length * l_len
    return {"total": total, "pred_to_target": l_pred, "target_to_pred": l_tgt, "render": l_r, "exist_sparse": l_e, "length": l_len}
