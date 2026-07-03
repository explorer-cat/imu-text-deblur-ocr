"""EasyOCR wrapper + CER evaluation over a generated split.

EasyOCR (and its torch dependency) is only imported lazily, so importing this
module is cheap and the rest of the pipeline doesn't require it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_gen import read_manifest, sample_paths
from metrics import cer, mean_cer


def ocr_langs(language: str) -> list[str]:
    if language == "en":
        return ["en"]
    # Korean recognizer in EasyOCR must be paired with "en".
    return ["ko", "en"]


def build_reader(cfg: dict):
    """Create an EasyOCR Reader (downloads models on first call)."""
    import easyocr

    return easyocr.Reader(ocr_langs(cfg["language"]), gpu=bool(cfg["ocr"]["gpu"]))


def read_image(reader, path: str | Path) -> str:
    """Run OCR on one image and return the concatenated recognized text."""
    result = reader.readtext(str(path), detail=0, paragraph=True)
    return " ".join(result) if result else ""


def evaluate_split(cfg: dict, reader, kind: str, limit: int | None = None) -> dict:
    """OCR every image of a given `kind` ('sharp' | 'blur' | a restored dir) in the
    test split and return mean CER + per-sample rows.

    `kind` may also be an absolute/relative directory of restored images named
    "{id}.png" (used for baseline/imu restorations).
    """
    rows = read_manifest(cfg, "test")
    if limit:
        rows = rows[:limit]

    restored_dir = None
    if kind not in ("sharp", "blur"):
        restored_dir = Path(kind)

    normalize = bool(cfg["ocr"]["normalize"])
    pairs: list[tuple[str, str]] = []
    detail = []
    for r in rows:
        sid, gt = r["id"], r["text"]
        if restored_dir is not None:
            img_path = restored_dir / f"{sid}.png"
        else:
            img_path = sample_paths(cfg, "test", sid)[kind]
        pred = read_image(reader, img_path)
        pairs.append((gt, pred))
        detail.append({"id": sid, "gt": gt, "pred": pred,
                       "cer": cer(gt, pred, normalize)})

    return {"kind": kind, "mean_cer": mean_cer(pairs, normalize), "detail": detail}
