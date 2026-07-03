"""Torch Dataset over a generated split (reads the manifest + image/imu files)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_gen import read_manifest, sample_paths


def _load_gray(path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]  # (1, H, W)


class DeblurDataset(Dataset):
    def __init__(self, cfg: dict, split: str):
        self.cfg = cfg
        self.split = split
        self.rows = read_manifest(cfg, split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        p = sample_paths(self.cfg, self.split, r["id"])
        imu = np.load(p["imu"]).astype(np.float32)   # (T, axes)
        return {
            "id": r["id"],
            "text": r["text"],
            "blur": _load_gray(p["blur"]),
            "sharp": _load_gray(p["sharp"]),
            "imu": torch.from_numpy(imu),
        }
