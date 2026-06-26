"""간단한 학습 스크립트."""
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
    p.add_argument("--outline-dir", required=True)
    p.add_argument("--skeleton-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-segments", type=int, default=32)
    p.add_argument("--num-target-points", type=int, default=256)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--length-min", type=float, default=2.0)
    p.add_argument("--w-render", type=float, default=1.0)
    p.add_argument("--w-pred-to-target", type=float, default=1.0)
    p.add_argument("--w-target-to-pred", type=float, default=1.0)
    p.add_argument("--w-exist", type=float, default=0.01)
    p.add_argument("--w-length", type=float, default=0.1)
    p.add_argument("--save-path", default="checkpoints/font_skeleton_model.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = FontSkeletonDataset(args.outline_dir, args.skeleton_dir, image_size=args.image_size, num_target_points=args.num_target_points, sigma=args.sigma)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = FontSkeletonModel(args.num_segments, args.image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = LossConfig(image_size=args.image_size, render_sigma=args.sigma, length_min=args.length_min, w_render=args.w_render, w_pred_to_target=args.w_pred_to_target, w_target_to_pred=args.w_target_to_pred, w_exist=args.w_exist, w_length=args.w_length)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, segments = model(batch["input_image"])
            losses = compute_losses(logits, segments, batch, cfg)
            losses["total"].backward()
            optimizer.step()
            running += float(losses["total"].detach())
        print(f"epoch {epoch:04d} total_loss={running / max(len(loader), 1):.6f}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
    print(f"저장 완료: {args.save_path}")


if __name__ == "__main__":
    main()
