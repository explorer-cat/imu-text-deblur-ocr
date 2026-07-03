"""STEP 5 — the one table that is the whole point.

Compares, on the test set:
    Blur (input) / Baseline (deblur only) / + IMU (ours) / Sharp (GT)
on PSNR (image quality) and CER (does OCR read it?).

If "+ IMU" beats "Baseline" on CER -> the experiment succeeded.

    python make_table.py --config configs/default.yaml [--limit 100] [--no-ocr]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from data_gen import load_config, read_manifest, sample_paths
from metrics import psnr


def load_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def mean_psnr_vs_sharp(cfg: dict, get_path, limit=None) -> float:
    rows = read_manifest(cfg, "test")
    if limit:
        rows = rows[:limit]
    vals = []
    for r in rows:
        sharp = load_gray(sample_paths(cfg, "test", r["id"])["sharp"])
        cmp_path = get_path(r["id"])
        if cmp_path is None or not Path(cmp_path).exists():
            return float("nan")
        vals.append(psnr(load_gray(cmp_path), sharp))
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")


def fmt(x: float) -> str:
    if x != x:            # NaN
        return "n/a"
    if x == float("inf"):
        return "—"   # em dash
    return f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ocr", action="store_true", help="PSNR only (skip EasyOCR)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    base_dir = Path("results/restored/baseline")
    imu_dir = Path("results/restored/imu")

    # ---- PSNR (vs sharp) ----
    psnr_map = {
        "Blur (input)": mean_psnr_vs_sharp(
            cfg, lambda i: sample_paths(cfg, "test", i)["blur"], args.limit),
        "Baseline (deblur only)": mean_psnr_vs_sharp(
            cfg, lambda i: base_dir / f"{i}.png", args.limit),
        "+ IMU (ours)": mean_psnr_vs_sharp(
            cfg, lambda i: imu_dir / f"{i}.png", args.limit),
        "Sharp (GT / upper bound)": float("inf"),
    }

    # ---- CER (via OCR) ----
    cer_map = {k: float("nan") for k in psnr_map}
    if not args.no_ocr:
        from ocr_eval import build_reader, evaluate_split

        print("Loading EasyOCR (first run downloads models)...")
        reader = build_reader(cfg)
        kinds = {
            "Blur (input)": "blur",
            "Baseline (deblur only)": str(base_dir),
            "+ IMU (ours)": str(imu_dir),
            "Sharp (GT / upper bound)": "sharp",
        }
        for label, kind in kinds.items():
            if kind not in ("sharp", "blur") and not Path(kind).exists():
                continue  # not trained yet
            print(f"  OCR: {label} ...")
            cer_map[label] = evaluate_split(cfg, reader, kind, limit=args.limit)["mean_cer"]

    # ---- render table ----
    order = ["Blur (input)", "Baseline (deblur only)", "+ IMU (ours)",
             "Sharp (GT / upper bound)"]
    header = "| Condition | PSNR (dB) ↑ | CER ↓ |"
    sep = "|---|---|---|"
    lines = [header, sep]
    for k in order:
        lines.append(f"| {k} | {fmt(psnr_map[k])} | {fmt(cer_map[k])} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "final_table.md").write_text(table + "\n", encoding="utf-8")
    with open(out / "final_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "psnr_db", "cer"])
        for k in order:
            w.writerow([k, fmt(psnr_map[k]), fmt(cer_map[k])])
    print(f"Saved: {out/'final_table.md'} and {out/'final_table.csv'}")

    b, im = cer_map["Baseline (deblur only)"], cer_map["+ IMU (ours)"]
    if np.isfinite(b) and np.isfinite(im):
        verdict = "SUCCESS — IMU lowered CER vs baseline." if im < b else \
                  "IMU did not beat baseline yet (try more epochs / stronger blur)."
        print("Verdict:", verdict)


if __name__ == "__main__":
    main()
