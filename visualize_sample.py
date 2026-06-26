"""샘플 시각화 및 65.svg sanity test."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from dataset import FontSkeletonDataset
from losses import LossConfig, compute_losses, soft_render_segments
from model import FontSkeletonModel


def plot_sample(sample: dict, save_path: str | None = None) -> None:
    """Dataset 샘플의 핵심 전처리 결과를 한 장에 그린다."""
    pts = sample["target_sample_points"].numpy()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(sample["input_image"][0], cmap="gray", origin="upper"); axes[0].set_title("outline image")
    axes[1].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); axes[1].set_title("target skeleton")
    axes[2].imshow(sample["target_dist"], cmap="magma", origin="upper"); axes[2].set_title("distance transform")
    axes[3].imshow(sample["target_heatmap"], cmap="viridis", origin="upper"); axes[3].set_title("target heatmap")
    axes[4].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); axes[4].scatter(pts[:, 0], pts[:, 1], s=4, c="red"); axes[4].set_title("sample points")
    for ax in axes:
        ax.set_xlim(0, 128); ax.set_ylim(128, 0); ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_prediction(sample: dict, pred_segments: torch.Tensor, pred_exist_logits: torch.Tensor, save_path: str | None = None) -> None:
    """예측 선분과 heatmap을 target과 비교해 그린다."""
    exist = torch.sigmoid(pred_exist_logits[0]).detach().cpu()
    segs = pred_segments[0].detach().cpu()
    pred_hm = soft_render_segments(pred_segments, torch.sigmoid(pred_exist_logits), image_size=128).detach().cpu()[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sample["target_skeleton"], cmap="gray", origin="upper")
    for e, s in zip(exist, segs):
        if e > 0.5:
            axes[0].plot([s[0], s[2]], [s[1], s[3]], "r-", alpha=float(e))
    axes[0].set_title("pred segments")
    axes[1].imshow(pred_hm, cmap="viridis", origin="upper"); axes[1].set_title("pred heatmap")
    axes[2].imshow(sample["target_heatmap"], cmap="viridis", origin="upper"); axes[2].set_title("target heatmap")
    for ax in axes:
        ax.set_xlim(0, 128); ax.set_ylim(128, 0); ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


def run_sanity_test(args: argparse.Namespace) -> None:
    dataset = FontSkeletonDataset(args.outline_dir, args.skeleton_dir, codepoints=[65], num_target_points=args.num_target_points, sigma=args.sigma)
    sample = dataset[0]
    print("input_image", tuple(sample["input_image"].shape))
    print("target_dist", tuple(sample["target_dist"].shape))
    print("target_heatmap", tuple(sample["target_heatmap"].shape))
    print("target_sample_points", tuple(sample["target_sample_points"].shape))
    plot_sample(sample, args.sample_plot)

    device = torch.device(args.device)
    model = FontSkeletonModel(args.num_segments).to(device)
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and k != "codepoint" else v for k, v in sample.items()}
    with torch.set_grad_enabled(True):
        logits, segments = model(batch["input_image"])
        losses = compute_losses(logits, segments, batch, LossConfig(render_sigma=args.sigma))
    for name, value in losses.items():
        print(f"{name}: {float(value.detach().cpu()):.6f}")
    plot_prediction(sample, segments, logits, args.pred_plot)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="65.svg sanity test와 시각화")
    p.add_argument("--outline-dir", required=True)
    p.add_argument("--skeleton-dir", required=True)
    p.add_argument("--num-segments", type=int, default=32)
    p.add_argument("--num-target-points", type=int, default=256)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--sample-plot", default="sanity_65_preprocess.png")
    p.add_argument("--pred-plot", default="sanity_65_prediction.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    run_sanity_test(parse_args())
