"""Evaluation metrics — no external dependencies (so step 1 needs nothing heavy).

CER  : Character Error Rate = edit_distance(ref, hyp) / len(ref)
PSNR : Peak Signal-to-Noise Ratio between two images in [0, 1].
"""

from __future__ import annotations

import re

import numpy as np


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace — focuses CER on characters, not case/spacing."""
    return re.sub(r"\s+", " ", s.strip().lower())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute), O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, normalize: bool = True) -> float:
    """Character Error Rate. 0 = perfect; can exceed 1 when hyp is much longer."""
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(ref, hyp) / len(ref)


def mean_cer(pairs: list[tuple[str, str]], normalize: bool = True) -> float:
    if not pairs:
        return float("nan")
    return float(np.mean([cer(r, h, normalize) for r, h in pairs]))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    """PSNR in dB between two arrays in [0, max_val]. Higher = closer."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))
