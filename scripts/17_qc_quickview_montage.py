"""Step 17: QC quickview montage for the in-flight final dataset.

Picks random EXTRACTED chips spread across batches (i.e. data generated at
different time periods) and renders a montage where each chip shows:
  left  -> RGB quicklook (B4/B3/B2, 2-98% stretch)
  right -> same RGB with the annotated solar-panel polygons overlaid
           (translucent red fill + bright outline)

Use it to eyeball whether the rasterised masks line up with the visible
panels across the whole generation run.

Outputs:
  docs/data_qc/quickview_montage_<UTC-stamp>.png       (grid of pairs)
  docs/data_qc/samples/<chip_id>_qc.png                (one PNG per chip)

Usage:
  python scripts/17_qc_quickview_montage.py
  python scripts/17_qc_quickview_montage.py --n 24 --seed 7 --cols 4
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction_pipeline import config as cfg

FINAL_DIR = cfg.PROJECT_ROOT / "outputs" / "final"
MASTER_CSV = FINAL_DIR / "master_manifest.csv"
CHIPS_DIR = FINAL_DIR / "chips"
QC_DIR = cfg.PROJECT_ROOT / "docs" / "data_qc"
SAMPLE_DIR = QC_DIR / "samples"

MASK_COLOR = np.array([1.0, 0.15, 0.15], dtype=np.float32)
# 0-indexed band positions for B4 (red), B3 (green), B2 (blue) in the 13-band stack
RGB_BANDS = (3, 2, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20, help="Number of chips to sample.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument("--cols", type=int, default=4, help="Number of chips per montage row.")
    p.add_argument("--dpi", type=int, default=120, help="Montage DPI.")
    p.add_argument("--no-samples", action="store_true", help="Skip per-chip PNGs; montage only.")
    return p.parse_args()


def percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((band.astype(np.float32) - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def read_rgb(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        bands = [src.read(b + 1).astype(np.float32) for b in RGB_BANDS]
    bands = [np.nan_to_num(b, nan=0.0) for b in bands]
    return np.stack([percentile_stretch(b) for b in bands], axis=-1)


def read_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def overlay_with_outline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    if mask.any():
        sel = mask == 1
        overlay[sel] = 0.55 * overlay[sel] + 0.45 * MASK_COLOR
    return overlay


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n rows spread as evenly as possible across batch_id."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["batch_id"] = df["batch_id"].fillna(-1).astype(int)
    batches = sorted(df["batch_id"].unique())
    if len(batches) <= n:
        picked_batches = batches
    else:
        idx = np.linspace(0, len(batches) - 1, n).round().astype(int)
        picked_batches = sorted({batches[i] for i in idx})
    rows = []
    for b in picked_batches:
        sub = df[df["batch_id"] == b]
        rows.append(sub.iloc[rng.integers(0, len(sub))])
    out = pd.DataFrame(rows)
    # top up if dedup of picked_batches left us short
    while len(out) < n:
        extra = df.sample(1, random_state=int(rng.integers(0, 1_000_000)))
        if extra.iloc[0]["chip_id_str"] not in set(out["chip_id_str"]):
            out = pd.concat([out, extra], ignore_index=True)
    return out.sort_values("batch_id").reset_index(drop=True).head(n)


def write_sample_png(rgb: np.ndarray, mask: np.ndarray, meta: pd.Series, out_path: Path) -> None:
    n_px = int(mask.sum())
    pct = 100.0 * n_px / mask.size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.3))
    ax1.imshow(rgb)
    ax1.set_title("RGB (B4/B3/B2)", fontsize=8)
    ax1.axis("off")
    ax2.imshow(overlay_with_outline(rgb, mask))
    ax2.contour(mask, levels=[0.5], colors=["yellow"], linewidths=0.7)
    ax2.set_title(f"polygons: {meta['n_polys']}  ·  {n_px}px ({pct:.2f}%)", fontsize=8)
    ax2.axis("off")
    fig.suptitle(
        f"{meta['chip_id_str']}  ·  batch {int(meta['batch_id'])}  ·  {meta['continent']}",
        fontsize=8, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not MASTER_CSV.exists():
        print(f"Master manifest not found: {MASTER_CSV}")
        return 1

    df = pd.read_csv(MASTER_CSV)
    df = df[df["status"] == "EXTRACTED"].copy()
    has_tifs = df["chip_id_str"].apply(
        lambda c: (CHIPS_DIR / f"{c}_image.tif").exists() and (CHIPS_DIR / f"{c}_mask.tif").exists()
    )
    df = df[has_tifs.values]
    if df.empty:
        print("No EXTRACTED chips with both image+mask tifs found.")
        return 1
    print(f"{len(df)} extracted chips available across {df['batch_id'].nunique()} batches.")

    picks = stratified_sample(df, args.n, args.seed)
    print(f"Sampled {len(picks)} chips from batches: {sorted(picks['batch_id'].astype(int).unique())}")

    QC_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_samples:
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    n = len(picks)
    cols = max(1, args.cols)
    rows = int(np.ceil(n / cols))
    # each chip = 2 sub-columns (RGB | overlay)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.0, rows * 2.25), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, (_, meta) in enumerate(picks.iterrows()):
        cid = meta["chip_id_str"]
        rgb = read_rgb(CHIPS_DIR / f"{cid}_image.tif")
        mask = read_mask(CHIPS_DIR / f"{cid}_mask.tif")
        n_px = int(mask.sum())
        pct = 100.0 * n_px / mask.size

        r, c = divmod(i, cols)
        a_rgb = axes[r][c * 2]
        a_ovl = axes[r][c * 2 + 1]
        a_rgb.imshow(rgb)
        a_rgb.set_title(f"{cid}\nb{int(meta['batch_id'])} · {meta['continent']}", fontsize=6.5)
        a_ovl.imshow(overlay_with_outline(rgb, mask))
        if mask.any():
            a_ovl.contour(mask, levels=[0.5], colors=["yellow"], linewidths=0.6)
        a_ovl.set_title(f"{int(meta['n_polys'])} poly · {n_px}px ({pct:.1f}%)", fontsize=6.5)

        if not args.no_samples:
            write_sample_png(rgb, mask, meta, SAMPLE_DIR / f"{cid}_qc.png")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fig.suptitle(
        f"Final-dataset QC · {n} random chips across {picks['batch_id'].nunique()} batches "
        f"(seed={args.seed}) · {stamp}",
        fontsize=10, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    montage_path = QC_DIR / f"quickview_montage_{stamp}.png"
    fig.savefig(montage_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMontage written -> {montage_path}")
    if not args.no_samples:
        print(f"Per-chip PNGs   -> {SAMPLE_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
