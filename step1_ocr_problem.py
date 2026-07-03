"""STEP 1  (start here — no training, can't fail)

Measure the *problem*: run OCR on the sharp (ground-truth) vs blurry test
images and compare CER. Expected result:

    blur CER  >>  sharp CER      ->  "blur breaks OCR", proven with a number.

This also gives us the evaluation tool (CER via OCR) reused in the final table.

Run:
    python data_gen.py --config configs/default.yaml   # if not generated yet
    python step1_ocr_problem.py --config configs/default.yaml [--limit 100]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data_gen import load_config
from ocr_eval import build_reader, evaluate_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="OCR only the first N test images (fast smoke check)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    manifest = Path(cfg["data"]["root"]) / "test_manifest.csv"
    if not manifest.exists():
        raise SystemExit(
            f"No test data at {manifest}. Generate it first:\n"
            f"  python data_gen.py --config {args.config}"
        )

    print("Loading EasyOCR (downloads models on first run)...")
    reader = build_reader(cfg)

    results = {}
    for kind in ("sharp", "blur"):
        print(f"\nOCR on '{kind}' images...")
        res = evaluate_split(cfg, reader, kind, limit=args.limit)
        results[kind] = res["mean_cer"]

    # ---- report ----
    sharp_cer, blur_cer = results["sharp"], results["blur"]
    print("\n" + "=" * 44)
    print("STEP 1 — Does blur break OCR?  (mean CER, lower=better)")
    print("=" * 44)
    print(f"  sharp (ground truth) : {sharp_cer:.3f}")
    print(f"  blur  (degraded)     : {blur_cer:.3f}")
    delta = blur_cer - sharp_cer
    print("-" * 44)
    print(f"  degradation          : +{delta:.3f} CER")
    verdict = "YES — blur clearly hurts OCR." if delta > 0.05 else \
              "Weak effect — consider stronger blur (blur.max_shift_px)."
    print(f"  verdict: {verdict}")
    print("=" * 44)

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "step1_ocr_problem.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_cer"])
        w.writerow(["sharp", f"{sharp_cer:.4f}"])
        w.writerow(["blur", f"{blur_cer:.4f}"])
    print(f"\nSaved: {out / 'step1_ocr_problem.csv'}")


if __name__ == "__main__":
    main()
