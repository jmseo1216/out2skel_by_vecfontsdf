"""샘플 시각화, 65.svg sanity test, checkpoint 기반 예측 시각화."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import FontSkeletonDataset
from losses import LossConfig, compute_losses, soft_render_segments
from model import FontSkeletonModel


def plot_sample(sample: dict, image_size: int = 128, save_path: str | None = None) -> None:
    """Dataset 샘플의 핵심 전처리 결과를 한 장에 그린다."""
    pts = sample["target_sample_points"].numpy()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(sample["input_image"][0], cmap="gray", origin="upper"); axes[0].set_title("outline image")
    axes[1].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); axes[1].set_title("target skeleton")
    axes[2].imshow(sample["target_dist"], cmap="magma", origin="upper"); axes[2].set_title("distance transform")
    axes[3].imshow(sample["target_heatmap"], cmap="viridis", origin="upper"); axes[3].set_title("target heatmap")
    axes[4].imshow(sample["target_skeleton"], cmap="gray", origin="upper"); axes[4].scatter(pts[:, 0], pts[:, 1], s=2, c="red"); axes[4].set_title("sample points")
    for ax in axes:
        ax.set_xlim(0, image_size); ax.set_ylim(image_size, 0); ax.axis("off")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_prediction(sample: dict, pred_segments: torch.Tensor, pred_exist_logits: torch.Tensor, image_size: int = 128, save_path: str | None = None) -> None:
    """예측 선분과 heatmap을 target과 비교해 그린다."""
    exist = torch.sigmoid(pred_exist_logits[0]).detach().cpu()
    segs = pred_segments[0].detach().cpu()
    pred_hm = soft_render_segments(pred_segments, torch.sigmoid(pred_exist_logits), image_size=image_size).detach().cpu()[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sample["target_skeleton"], cmap="gray", origin="upper")
    for e, s in zip(exist, segs):
        if e > 0.5:
            axes[0].plot([s[0], s[2]], [s[1], s[3]], "r-", alpha=float(e))
    axes[0].set_title("pred segments")
    axes[1].imshow(pred_hm, cmap="viridis", origin="upper"); axes[1].set_title("pred heatmap")
    axes[2].imshow(sample["target_heatmap"], cmap="viridis", origin="upper"); axes[2].set_title("target heatmap")
    for ax in axes:
        ax.set_xlim(0, image_size); ax.set_ylim(image_size, 0); ax.axis("off")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def load_model(args: argparse.Namespace, device: torch.device) -> FontSkeletonModel:
    """checkpoint가 있으면 가중치를 불러오고, 없으면 랜덤 초기화 모델을 반환한다."""
    model = FontSkeletonModel(args.k_segments, args.image_size).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"checkpoint 로드 완료: {args.checkpoint}")
    else:
        print("checkpoint가 지정되지 않아 랜덤 초기화 모델로 sanity forward를 수행합니다.")
    return model


def run_sanity_test(args: argparse.Namespace) -> None:
    """codepoint 하나를 로드하여 전처리/forward/loss/시각화를 수행한다."""
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    dataset = FontSkeletonDataset(
        args.outline_dir,
        args.skeleton_dir,
        codepoints=[args.codepoint],
        image_size=args.image_size,
        svg_size=args.svg_size,
        num_target_points=args.num_target_points,
        sigma=args.sigma,
    )
    sample = dataset[0]
    print("input_image", tuple(sample["input_image"].shape))
    print("target_dist", tuple(sample["target_dist"].shape))
    print("target_heatmap", tuple(sample["target_heatmap"].shape))
    print("target_sample_points", tuple(sample["target_sample_points"].shape))

    sample_plot = str(save_dir / f"codepoint_{args.codepoint}_preprocess.png") if save_dir else args.sample_plot
    pred_plot = str(save_dir / f"codepoint_{args.codepoint}_prediction.png") if save_dir else args.pred_plot
    plot_sample(sample, args.image_size, sample_plot)

    device = torch.device(args.device)
    model = load_model(args, device)
    model.eval()
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) and k != "codepoint" else v for k, v in sample.items()}
    with torch.set_grad_enabled(args.checkpoint is None):
        logits, segments = model(batch["input_image"])
        losses = compute_losses(logits, segments, batch, LossConfig(image_size=args.image_size, render_sigma=args.sigma))
    for name, value in losses.items():
        print(f"{name}: {float(value.detach().cpu()):.6f}")
    plot_prediction(sample, segments, logits, args.image_size, pred_plot)
    if save_dir:
        print(f"시각화 저장 완료: {save_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVG 샘플 sanity test와 checkpoint 예측 시각화")
    p.add_argument("--outline_dir", "--outline-dir", required=True)
    p.add_argument("--skeleton_dir", "--skeleton-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--save_dir", "--save-dir", default=None)
    p.add_argument("--codepoint", type=int, default=65)
    p.add_argument("--image_size", "--image-size", type=int, default=128)
    p.add_argument("--svg_size", "--svg-size", type=float, default=50.0)
    p.add_argument("--k_segments", "--num-segments", type=int, default=32)
    p.add_argument("--num_target_points", "--num-target-points", type=int, default=256)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--sample_plot", "--sample-plot", default=None)
    p.add_argument("--pred_plot", "--pred-plot", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    run_sanity_test(parse_args())
