"""Hungarian segment matching 기반 loss 모듈."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass
class LossConfig:
    image_size: int = 128
    render_sigma: float = 2.0
    samples_per_segment: int = 16
    w_match_segment: float = 5.0
    w_exist_bce: float = 1.0
    w_exist_count: float = 0.1
    w_total_length: float = 0.5
    w_render: float = 5.0
    w_pred_to_target: float = 0.5
    w_target_to_pred: float = 0.5
    render_fg_weight: float = 10.0
    inactive_distance: float = 128.0
    eps: float = 1e-6


def segment_lengths_torch(segments: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """segments=[...,4]의 길이를 계산한다."""
    return torch.sqrt(((segments[..., 2:4] - segments[..., 0:2]) ** 2).sum(dim=-1).clamp_min(eps))


def direction_invariant_endpoint_l1(pred: torch.Tensor, target: torch.Tensor, image_size: int) -> torch.Tensor:
    """방향에 무관한 endpoint L1 cost를 계산한다. pred=[...,4], target=[...,4]."""
    normal = (pred[..., 0:2] - target[..., 0:2]).abs().sum(dim=-1) + (pred[..., 2:4] - target[..., 2:4]).abs().sum(dim=-1)
    flipped = (pred[..., 0:2] - target[..., 2:4]).abs().sum(dim=-1) + (pred[..., 2:4] - target[..., 0:2]).abs().sum(dim=-1)
    return torch.minimum(normal, flipped) / float(image_size)


def pairwise_segment_cost(pred_segments: torch.Tensor, target_segments: torch.Tensor, image_size: int) -> torch.Tensor:
    """K개 예측 segment와 T개 target segment 사이의 방향 불변 cost [K,T]를 만든다."""
    pred = pred_segments[:, None, :]
    target = target_segments[None, :, :]
    return direction_invariant_endpoint_l1(pred, target, image_size)


def hungarian_match_indices(pred_segments: torch.Tensor, target_segments: torch.Tensor, target_mask: torch.Tensor, cfg: LossConfig) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """batch별 Hungarian matching index를 반환한다. matching 자체는 detach된 CPU cost로 수행한다."""
    matches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    bsz = pred_segments.shape[0]
    for b in range(bsz):
        valid_targets = target_segments[b][target_mask[b]]
        t_count = int(valid_targets.shape[0])
        if t_count == 0:
            raise RuntimeError("Hungarian matching 대상 target segment가 0개입니다.")
        if t_count > pred_segments.shape[1]:
            raise RuntimeError(f"target segment 수({t_count})가 predicted K({pred_segments.shape[1]})보다 큽니다.")
        cost = pairwise_segment_cost(pred_segments[b], valid_targets, cfg.image_size).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        matches.append((
            torch.as_tensor(row_ind, dtype=torch.long, device=pred_segments.device),
            torch.as_tensor(col_ind, dtype=torch.long, device=pred_segments.device),
        ))
    return matches


def point_to_segment_distance(points: torch.Tensor, segments: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """점과 선분 거리. points=[B,M,2], segments=[B,K,4], 반환=[B,M,K]."""
    q = points[:, :, None, :]
    a = segments[:, None, :, 0:2]
    b = segments[:, None, :, 2:4]
    ab = b - a
    aq = q - a
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(eps)
    t = ((aq * ab).sum(dim=-1, keepdim=True) / denom).clamp(0.0, 1.0)
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


def make_pixel_grid(batch_size: int, image_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """픽셀 중심 좌표 grid [B,H*W,2]를 생성한다."""
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=dtype) + 0.5,
        torch.arange(image_size, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).view(1, image_size * image_size, 2).expand(batch_size, -1, -1)


def soft_render_segments(pred_segments: torch.Tensor, exist_prob: torch.Tensor, image_size: int = 128, sigma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """예측 선분을 differentiable heatmap [B,H,W]로 soft rasterize한다."""
    bsz = pred_segments.shape[0]
    grid = make_pixel_grid(bsz, image_size, pred_segments.device, pred_segments.dtype)
    dist = point_to_segment_distance(grid, pred_segments, eps)
    inactive_penalty = (1.0 - exist_prob)[:, None, :] * float(image_size)
    pred_dist = (dist + inactive_penalty).min(dim=-1).values.view(bsz, image_size, image_size)
    return torch.exp(-(pred_dist ** 2) / (2.0 * sigma ** 2))


def loss_pred_to_target(pred_segments: torch.Tensor, exist_prob: torch.Tensor, target_dist: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """예측 선분 샘플 위치의 target distance 평균을 image_size로 정규화한다."""
    bsz, k, _ = pred_segments.shape
    pts = sample_predicted_segments(pred_segments, cfg.samples_per_segment).reshape(bsz, k * cfg.samples_per_segment, 2)
    grid = pixel_to_grid(pts, cfg.image_size).view(bsz, k * cfg.samples_per_segment, 1, 2)
    sampled = F.grid_sample(target_dist[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.view(bsz, k, cfg.samples_per_segment) / float(cfg.image_size)
    weights = exist_prob[:, :, None]
    return (sampled * weights).sum() / (weights.sum() * cfg.samples_per_segment + cfg.eps)


def loss_target_to_pred(pred_segments: torch.Tensor, exist_prob: torch.Tensor, target_points: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """target sample point가 활성 예측 선분으로 덮이도록 하는 auxiliary loss."""
    dist = point_to_segment_distance(target_points, pred_segments, cfg.eps) / float(cfg.image_size)
    inactive_penalty = (1.0 - exist_prob)[:, None, :] * cfg.inactive_distance / float(cfg.image_size)
    return (dist + inactive_penalty).min(dim=-1).values.mean()


def compute_matched_losses(pred_exist_logits: torch.Tensor, pred_segments: torch.Tensor, batch: Dict[str, torch.Tensor], cfg: LossConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Hungarian matching으로 match_segment loss와 exist BCE target을 계산한다."""
    target_segments = batch["target_segments"]
    target_mask = batch["target_segment_mask"]
    matches = hungarian_match_indices(pred_segments, target_segments, target_mask, cfg)
    bsz, k = pred_exist_logits.shape
    exist_target = torch.zeros_like(pred_exist_logits)
    match_terms = []
    for b, (pred_idx, tgt_compact_idx) in enumerate(matches):
        valid_targets = target_segments[b][target_mask[b]]
        matched_pred = pred_segments[b, pred_idx]
        matched_tgt = valid_targets[tgt_compact_idx]
        exist_target[b, pred_idx] = 1.0
        match_terms.append(direction_invariant_endpoint_l1(matched_pred, matched_tgt, cfg.image_size).mean())
    match_segment = torch.stack(match_terms).mean()
    exist_bce = F.binary_cross_entropy_with_logits(pred_exist_logits, exist_target)
    return match_segment, exist_bce


def compute_losses(pred_exist_logits: torch.Tensor, pred_segments: torch.Tensor, batch: Dict[str, torch.Tensor], cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """segment matching 중심 loss dictionary를 반환한다."""
    exist_prob = torch.sigmoid(pred_exist_logits)
    match_segment, exist_bce = compute_matched_losses(pred_exist_logits, pred_segments, batch, cfg)

    pred_count = exist_prob.sum(dim=1)
    target_count = batch["target_num_segments"].to(pred_count.dtype)
    exist_count = (((pred_count - target_count) / float(pred_exist_logits.shape[1])) ** 2).mean()

    pred_lengths = segment_lengths_torch(pred_segments, cfg.eps)
    pred_total_length = (exist_prob * pred_lengths).sum(dim=1)
    target_total_length = batch["target_total_length"].to(pred_total_length.dtype)
    total_length = (pred_total_length - target_total_length).abs().div(float(cfg.image_size)).mean()

    pred_heatmap = soft_render_segments(pred_segments, exist_prob, cfg.image_size, cfg.render_sigma, cfg.eps)
    weight = 1.0 + cfg.render_fg_weight * batch["target_heatmap"]
    render = (weight * (pred_heatmap - batch["target_heatmap"]) ** 2).mean()

    pred_to_target = loss_pred_to_target(pred_segments, exist_prob, batch["target_dist"], cfg)
    target_to_pred = loss_target_to_pred(pred_segments, exist_prob, batch["target_sample_points"], cfg)

    total = (
        cfg.w_match_segment * match_segment
        + cfg.w_exist_bce * exist_bce
        + cfg.w_exist_count * exist_count
        + cfg.w_total_length * total_length
        + cfg.w_render * render
        + cfg.w_pred_to_target * pred_to_target
        + cfg.w_target_to_pred * target_to_pred
    )
    return {
        "total": total,
        "match_segment": match_segment,
        "exist_bce": exist_bce,
        "exist_count": exist_count,
        "total_length": total_length,
        "render": render,
        "pred_to_target": pred_to_target,
        "target_to_pred": target_to_pred,
    }
