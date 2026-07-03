"""STEPS 3 & 4 — train the deblurring network.

    python train.py --config configs/default.yaml --model baseline   # step 3
    python train.py --config configs/default.yaml --model imu         # step 4

Saves a checkpoint to results/checkpoints/{model}.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_gen import load_config
from dataset import DeblurDataset
from models import build_model
from utils import pick_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["baseline", "imu"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device(cfg["train"]["device"])
    use_imu = args.model == "imu"
    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]

    ds = DeblurDataset(cfg, "train")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=2, drop_last=True)

    model = build_model(cfg, use_imu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    l1 = nn.L1Loss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  use_imu={use_imu}  params={n_params/1e3:.1f}K  device={device}")

    model.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for batch in loader:
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            imu = batch["imu"].to(device) if use_imu else None

            pred = model(blur, imu)
            loss = l1(pred, sharp)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * blur.size(0)
        print(f"  epoch {ep:3d}/{epochs}  L1={running/len(ds):.4f}")

    out_dir = Path("results/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.model}.pt"
    torch.save({"state_dict": model.state_dict(),
                "use_imu": use_imu, "config": cfg}, ckpt)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
