"""간단한 학습 스크립트.

PowerShell 스크립트와 호환되도록 argparse는 --outline_dir 및 --outline-dir 형식을 모두 받는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FontSkeletonDataset
from losses import LossConfig, compute_losses
from model import FontSkeletonModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="outline SVG에서 laser skeleton 선분을 예측하는 첫 학습 루프")
    p.add_argument("--outline_dir", "--outline-dir", required=True)
    p.add_argument("--skeleton_dir", "--skeleton-dir", required=True)
    p.add_argument("--output_dir", "--output-dir", default="runs/font_skeleton")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", "--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k_segments", "--num-segments", type=int, default=32)
    p.add_argument("--num_target_points", "--num-target-points", type=int, default=256)
    p.add_argument("--pred_sample_points", "--samples-per-segment", type=int, default=16)
    p.add_argument("--image_size", "--image-size", type=int, default=128)
    p.add_argument("--svg_size", "--svg-size", type=float, default=50.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--length_min", "--length-min", type=float, default=2.0)
    p.add_argument("--w_render", "--w-render", type=float, default=1.0)
    p.add_argument("--w_pred_to_target", "--w-pred-to-target", type=float, default=1.0)
    p.add_argument("--w_target_to_pred", "--w-target-to-pred", type=float, default=1.0)
    p.add_argument("--w_exist", "--w-exist", type=float, default=0.01)
    p.add_argument("--w_length", "--w-length", type=float, default=0.1)
    p.add_argument("--save_path", "--save-path", default=None)
    p.add_argument("--log_every", "--log-every", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    """batch 안의 tensor를 device로 이동한다."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_path) if args.save_path else checkpoint_dir / "font_skeleton_model.pt"

    device = torch.device(args.device)
    dataset = FontSkeletonDataset(
        args.outline_dir,
        args.skeleton_dir,
        image_size=args.image_size,
        svg_size=args.svg_size,
        num_target_points=args.num_target_points,
        sigma=args.sigma,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = FontSkeletonModel(args.k_segments, args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = LossConfig(
        image_size=args.image_size,
        render_sigma=args.sigma,
        samples_per_segment=args.pred_sample_points,
        length_min=args.length_min,
        w_render=args.w_render,
        w_pred_to_target=args.w_pred_to_target,
        w_target_to_pred=args.w_target_to_pred,
        w_exist=args.w_exist,
        w_length=args.w_length,
    )

    log_path = log_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,total,pred_to_target,target_to_pred,render,exist_sparse,length\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "pred_to_target": 0.0, "target_to_pred": 0.0, "render": 0.0, "exist_sparse": 0.0, "length": 0.0}
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, segments = model(batch["input_image"])
            losses = compute_losses(logits, segments, batch, cfg)
            losses["total"].backward()
            optimizer.step()
            for k in sums:
                sums[k] += float(losses[k].detach())
        means = {k: v / max(len(loader), 1) for k, v in sums.items()}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(",".join([str(epoch)] + [f"{means[k]:.8f}" for k in sums]) + "\n")
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            loss_text = ", ".join(f"{k}={v:.6f}" for k, v in means.items())
            print(f"epoch {epoch:04d}: {loss_text}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_path)
    print(f"checkpoint 저장 완료: {save_path}")
    print(f"loss log 저장 완료: {log_path}")


if __name__ == "__main__":
    main()
