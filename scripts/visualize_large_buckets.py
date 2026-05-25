#!/usr/bin/env python
"""Render sample chip diagnostic figures for the three audit health buckets.

For each chip:
  Panel 1: 2024 training-chip RGB
  Panel 2: Hand-labelled ground-truth mask overlaid on RGB (cyan)
  Panel 3: R3 R101 detection at threshold 0.85 overlaid on RGB (red)

Sampling:
  - HEALTHY  (Dice >= 0.85): 10 random chips
  - MIDDLING (0.50-0.85)   : 15 random chips spread across the range
  - FAILING  (< 0.50)      : ALL 30

Output: docs/audit_30k_large_examples/<bucket>/<chip_id>_diag.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

AUDIT_CSV = REPO / "docs/audit_30k_h5.csv"
H5_PATH   = REPO / "outputs/final/final_dataset_repacked.h5"
OUT_DIR   = REPO / "docs/audit_30k_large_examples"


def make_rgb(img_chw: np.ndarray) -> np.ndarray:
    rgb = np.transpose(img_chw[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
    valid = rgb[rgb > 0]
    lo = np.percentile(valid, 2) if valid.size else 0
    hi = np.percentile(valid, 98) if valid.size else 1
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def render(chip_id: str, img_chw: np.ndarray, gt: np.ndarray, pred: np.ndarray,
           meta: dict, out_path: Path) -> None:
    rgb = make_rgb(img_chw)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))

    ax[0].imshow(rgb)
    ax[0].set_title("2024 RGB (B4/B3/B2)", fontsize=11)
    ax[0].axis("off")

    gt_overlay = rgb.copy()
    gt_overlay[gt > 0] = [0.15, 1.0, 1.0]   # cyan = hand-label
    ax[1].imshow(gt_overlay)
    ax[1].set_title(f"Manual label (cyan)  —  {int(gt.sum())} px", fontsize=11)
    ax[1].axis("off")

    pred_overlay = rgb.copy()
    pred_overlay[pred > 0] = [1.0, 0.15, 0.15]  # red = model
    ax[2].imshow(pred_overlay)
    ax[2].set_title(f"R3 R101 detection (red)  —  {int(pred.sum())} px", fontsize=11)
    ax[2].axis("off")

    fig.suptitle(
        f"{chip_id}  |  {meta['continent']}  lat={float(meta['lat']):+.2f}  lon={float(meta['lon']):+.2f}  "
        f"|  panel_frac={float(meta['panel_frac']):.2f}  "
        f"|  Dice={float(meta['dice']):.3f}  P={meta['precision']}  R={meta['recall']}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    # Load + filter
    rows = list(csv.DictReader(AUDIT_CSV.open()))
    large = [r for r in rows if r["label"] == "1" and float(r["panel_frac"]) >= 0.40]
    print(f"Large positives: {len(large):,}")
    rng = np.random.default_rng(42)

    healthy_pool  = [r for r in large if float(r["dice"]) >= 0.85]
    middling_pool = [r for r in large if 0.50 <= float(r["dice"]) < 0.85]
    failing_pool  = [r for r in large if float(r["dice"]) < 0.50]

    healthy_sample  = list(rng.choice(healthy_pool, size=min(10, len(healthy_pool)), replace=False))
    # spread middling across the range
    middling_sorted = sorted(middling_pool, key=lambda r: float(r["dice"]))
    step = max(1, len(middling_sorted) // 15)
    middling_sample = middling_sorted[::step][:15]
    failing_sample  = failing_pool  # all 30

    print(f"To render: healthy={len(healthy_sample)}  middling={len(middling_sample)}  failing={len(failing_sample)}")

    # Index H5
    with h5py.File(H5_PATH, "r") as h5:
        chip_ids = [c.decode() for c in h5["chip_ids"][:]]
    id_to_idx = {}
    for i, c in enumerate(chip_ids):
        if c not in id_to_idx:
            id_to_idx[c] = i

    # Model
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(registry.get("threshold", 0.85))

    for bucket, sample in [("healthy", healthy_sample),
                           ("middling", middling_sample),
                           ("failing", failing_sample)]:
        bucket_dir = OUT_DIR / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{bucket}] ({len(sample)} chips)")
        with h5py.File(H5_PATH, "r") as h5:
            for r in sample:
                cid = r["chip_id"]
                if cid not in id_to_idx:
                    print(f"  SKIP {cid}: not in H5"); continue
                idx = id_to_idx[cid]
                img_chw = h5["images"][idx].astype(np.float32)
                gt = (h5["masks"][idx].astype(np.float32) > 0.5).astype(np.uint8)
                img_hwc = np.transpose(img_chw, (1, 2, 0))[None]
                normed = normalize_zscore(img_hwc, band_stats)
                probs = model.predict(normed, verbose=0)[0, :, :, 0]
                pred = (probs > thr).astype(np.uint8)
                fname = f"dice_{float(r['dice']):.3f}_{cid}_diag.png".replace("..", ".")
                out = bucket_dir / fname
                render(cid, img_chw, gt, pred, r, out)
                print(f"  -> {out.relative_to(REPO)}  (dice={r['dice']})")


if __name__ == "__main__":
    main()
