"""Synthetic data generator for the IMU-guided text-deblurring experiment.

Each sample is a triplet:
  * sharp : clean rendered text image      (ground truth)
  * blur  : sharp convolved with a motion-blur PSF built from a simulated
            camera-shake trajectory during "exposure"
  * imu   : the gyro (angular-velocity) time series that PRODUCED that shake

The physical story: camera shake during exposure integrates a motion
trajectory; the blur kernel *is* that trajectory. A gyroscope measures the
angular velocity of the same motion, so the IMU signal encodes the blur kernel.
That is exactly why feeding IMU to a deblurring network should help.

Run:
    python data_gen.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# Config / fonts / text
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_font(size: int, language: str) -> ImageFont.FreeTypeFont:
    """Return a TrueType font. Uses matplotlib's bundled DejaVuSans for English
    (portable across Mac/Colab). For Korean, tries common CJK fonts."""
    candidates: list[str] = []
    if language in ("ko", "both"):
        candidates += [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Colab (apt)
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",       # macOS
            "/Library/Fonts/NanumGothic.ttf",
        ]
    # English default: matplotlib always ships DejaVuSans.ttf.
    try:
        import matplotlib.font_manager as fm

        candidates.append(fm.findfont(fm.FontProperties(family="DejaVu Sans")))
    except Exception:
        pass

    for c in candidates:
        if c and Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    # Last resort — bitmap default (ugly but never crashes).
    return ImageFont.load_default()


_KO_SYLLABLES = [chr(cp) for cp in range(0xAC00, 0xAC00 + 400)]  # 가..


def random_text(cfg: dict, rng: np.random.Generator) -> str:
    n = rng.integers(cfg["text"]["min_chars"], cfg["text"]["max_chars"] + 1)
    if cfg["language"] == "ko":
        pool = _KO_SYLLABLES
    elif cfg["language"] == "both":
        pool = list(cfg["text"]["charset"]) + _KO_SYLLABLES
    else:
        pool = list(cfg["text"]["charset"])
    return "".join(rng.choice(pool) for _ in range(int(n)))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_sharp(text: str, cfg: dict, font: ImageFont.FreeTypeFont,
                 rng: np.random.Generator) -> np.ndarray:
    """Render black text on white. Returns float32 grayscale in [0, 1], (H, W)."""
    H, W = cfg["image"]["height"], cfg["image"]["width"]
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # center with small jitter
    x = (W - tw) / 2 - bbox[0] + rng.integers(-6, 7)
    y = (H - th) / 2 - bbox[1] + rng.integers(-4, 5)
    draw.text((x, y), text, fill=0, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Camera shake -> gyro (IMU) + motion-blur PSF
# --------------------------------------------------------------------------- #
def sample_shake(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one exposure of camera shake.

    Returns:
        traj : (T, 2) pixel-space trajectory of the shake (integral of motion)
        gyro : (T, axes) angular-velocity time series with sensor noise (the IMU)
    """
    T = cfg["blur"]["exposure_steps"]
    axes = cfg["imu"]["axes"]
    alpha = float(cfg["blur"]["smoothness"])

    # 1) angular velocity = low-pass filtered white noise (smooth, correlated shake)
    w = rng.standard_normal((T, axes)).astype(np.float32)
    for t in range(1, T):
        w[t] = alpha * w[t - 1] + (1 - alpha) * w[t]

    # 2) trajectory = cumulative integral of angular velocity (small-angle approx)
    traj = np.cumsum(w, axis=0)
    traj -= traj.mean(axis=0, keepdims=True)          # center around origin
    span = np.abs(traj).max() + 1e-6
    traj = traj / span * (cfg["blur"]["max_shift_px"] / 2.0)

    # 3) the IMU we hand to the model = gyro (w) + sensor noise
    gyro = w + rng.standard_normal(w.shape).astype(np.float32) * cfg["imu"]["noise_std"]
    return traj.astype(np.float32), gyro.astype(np.float32)


def psf_from_traj(traj: np.ndarray, kernel_size: int) -> np.ndarray:
    """Rasterize a trajectory into a normalized motion-blur PSF (bilinear splat)."""
    k = kernel_size
    psf = np.zeros((k, k), dtype=np.float32)
    c = (k - 1) / 2.0
    for dx, dy in traj:
        fx, fy = c + dx, c + dy
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        wx, wy = fx - x0, fy - y0
        for xi, wxi in ((x0, 1 - wx), (x0 + 1, wx)):
            for yi, wyi in ((y0, 1 - wy), (y0 + 1, wy)):
                if 0 <= xi < k and 0 <= yi < k:
                    psf[yi, xi] += wxi * wyi
    s = psf.sum()
    if s < 1e-8:  # (near) no motion -> delta
        psf[:] = 0.0
        psf[int(c), int(c)] = 1.0
        return psf
    return psf / s


def apply_blur(sharp: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    blurred = convolve(sharp, psf, mode="reflect")
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# One sample + dataset generation
# --------------------------------------------------------------------------- #
def make_sample(cfg: dict, font: ImageFont.FreeTypeFont, rng: np.random.Generator):
    text = random_text(cfg, rng)
    sharp = render_sharp(text, cfg, font, rng)
    traj, gyro = sample_shake(cfg, rng)
    psf = psf_from_traj(traj, cfg["blur"]["kernel_size"])
    blur = apply_blur(sharp, psf)
    return text, sharp, blur, gyro, psf


def _save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def generate_split(cfg: dict, split: str, n: int, seed: int) -> Path:
    root = Path(cfg["data"]["root"])
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    font = get_font(cfg["text"]["font_size"], cfg["language"])
    rng = np.random.default_rng(seed)

    from tqdm import tqdm

    rows = []
    for i in tqdm(range(n), desc=f"gen {split}"):
        text, sharp, blur, gyro, _ = make_sample(cfg, font, rng)
        sid = f"{i:05d}"
        _save_gray(sharp, out_dir / f"{sid}_sharp.png")
        _save_gray(blur, out_dir / f"{sid}_blur.png")
        np.save(out_dir / f"{sid}_imu.npy", gyro)
        rows.append({"id": sid, "split": split, "text": text})

    manifest = root / f"{split}_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def read_manifest(cfg: dict, split: str) -> list[dict]:
    path = Path(cfg["data"]["root"]) / f"{split}_manifest.csv"
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_paths(cfg: dict, split: str, sid: str) -> dict:
    d = Path(cfg["data"]["root"]) / split
    return {
        "sharp": d / f"{sid}_sharp.png",
        "blur": d / f"{sid}_blur.png",
        "imu": d / f"{sid}_imu.npy",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train", type=int, default=None, help="override train count")
    ap.add_argument("--test", type=int, default=None, help="override test count")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_train = args.train if args.train is not None else cfg["data"]["train"]
    n_test = args.test if args.test is not None else cfg["data"]["test"]

    seed = cfg["seed"]
    m_train = generate_split(cfg, "train", n_train, seed)
    m_test = generate_split(cfg, "test", n_test, seed + 1)
    print(f"\nDone. Manifests:\n  {m_train}\n  {m_test}")
    print(json.dumps({"train": n_train, "test": n_test,
                      "language": cfg["language"],
                      "image": [cfg['image']['height'], cfg['image']['width']]}, indent=2))


if __name__ == "__main__":
    main()
