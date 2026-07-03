"""Restore the test set with a trained model and report PSNR vs. ground truth.

    python infer.py --config configs/default.yaml --model baseline
    python infer.py --config configs/default.yaml --model imu

Writes restored images to results/restored/{model}/{id}.png (consumed by
make_table.py for OCR/CER), and prints mean PSNR of restored vs. sharp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data_gen import load_config
from dataset import DeblurDataset
from metrics import psnr
from models import build_model
from utils import pick_device


def save_gray(t: torch.Tensor, path: Path) -> None:
    arr = (t.squeeze().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["baseline", "imu"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg["train"]["device"])

    ckpt_path = Path("results/checkpoints") / f"{args.model}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint {ckpt_path}. Train first: python train.py --model {args.model}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    use_imu = ckpt["use_imu"]
    model = build_model(cfg, use_imu).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = Path("results/restored") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = DeblurDataset(cfg, "test")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    psnrs = []
    with torch.no_grad():
        for batch in loader:
            blur = batch["blur"].to(device)
            imu = batch["imu"].to(device) if use_imu else None
            pred = model(blur, imu).cpu()
            sharp = batch["sharp"]
            for i, sid in enumerate(batch["id"]):
                save_gray(pred[i], out_dir / f"{sid}.png")
                psnrs.append(psnr(pred[i].squeeze().numpy(), sharp[i].squeeze().numpy()))

    print(f"model={args.model}  mean PSNR (restored vs sharp) = {np.mean(psnrs):.2f} dB")
    print(f"Restored images -> {out_dir}")


if __name__ == "__main__":
    main()
