#!/usr/bin/env python
"""Deep-dive diagnostic for one real-world test site.

Renders a 6-panel figure:
  1. 2024 training chip RGB (from H5)
  2. 2024 training mask (ground truth from H5, what the model was taught)
  3. 2026 mosaic RGB (the test mosaic we just generated)
  4. 2026 probability heatmap (continuous output, no threshold)
  5. 2026 detection at production threshold (0.85)
  6. 2026 detection at lower threshold (0.50)

Tells us whether "missed expansion" is a confidence-threshold issue
(panels visible at thr=0.50 but not 0.85) or a real model blind spot
(panels not detected even at thr=0.50).
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip-id", required=True, help="e.g. c+001720_r+000922")
    ap.add_argument("--nc-dir", default="docs/realworld_unseen_50_2026Q1",
                    help="Directory with <chip_id>_stack.nc files")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()

    cid = args.chip_id
    nc_path = REPO / args.nc_dir / f"{cid}_stack.nc"
    if not nc_path.exists():
        sys.exit(f"No stack: {nc_path}")

    # ---- Load 2024 training chip ----
    h5_path = REPO / "outputs/final/final_dataset_repacked.h5"
    train_rgb = None; train_mask = None; train_idx = None
    with h5py.File(h5_path, "r") as h5:
        chip_ids = [c.decode() for c in h5["chip_ids"][:]]
        for i, c in enumerate(chip_ids):
            if c == cid:
                train_idx = i; break
        if train_idx is not None:
            img_chw = h5["images"][train_idx].astype(np.float32)  # (13, 256, 256)
            train_mask = h5["masks"][train_idx].astype(np.float32)  # (256, 256)
            rgb = np.transpose(img_chw[[3, 2, 1]], (1, 2, 0))
            valid = rgb[rgb > 0]
            lo = np.percentile(valid, 2) if valid.size else 0
            hi = np.percentile(valid, 98) if valid.size else 1
            train_rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)

    # ---- Load 2026 mosaic + run inference with probabilities ----
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rw", REPO / "scripts/19_test_real_world_sites.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore

    spectral, scl, dates = mod.load_stack(nc_path)
    print(f"Stack: {spectral.shape[0]} scenes × 13 × {spectral.shape[2]}×{spectral.shape[3]}")
    print(f"Dates: {dates[0]} .. {dates[-1]}")
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    composite, info = create_temporal_mosaic(spectral, scl)
    print(f"Mosaic: {info['n_scenes_used']} scenes, "
          f"{info['fill_mode'].sum()*100/info['fill_mode'].size:.1f}% rescue fill")

    image_hwc = np.transpose(composite[:13], (1, 2, 0))
    test_rgb = mod.make_rgb(composite)

    # Tiled inference returning probabilities
    model, band_stats, registry = _get_model_and_stats()
    h, w, c = image_hwc.shape
    probs = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, 256):
        for x0 in range(0, w, 256):
            tile = image_hwc[y0:y0+256, x0:x0+256, :]
            th, tw = tile.shape[0], tile.shape[1]
            if th < 256 or tw < 256:
                pad = np.zeros((256, 256, c), dtype=np.float32)
                pad[:th, :tw, :] = tile; tile = pad
            n = normalize_zscore(tile[np.newaxis], band_stats)
            p = model.predict(n, verbose=0)
            probs[y0:y0+th, x0:x0+tw] = p[0, :th, :tw, 0]

    mask_85 = (probs > 0.85).astype(np.uint8)
    mask_50 = (probs > 0.50).astype(np.uint8)
    print(f"Detections: thr=0.85 -> {int(mask_85.sum())} px,  "
          f"thr=0.50 -> {int(mask_50.sum())} px  "
          f"(ratio 0.50/0.85 = {mask_50.sum()/max(mask_85.sum(),1):.2f}x)")

    # ---- Render ----
    fig, ax = plt.subplots(2, 3, figsize=(20, 13))
    if train_rgb is not None:
        ax[0, 0].imshow(train_rgb)
        ax[0, 0].set_title(f"2024 training chip RGB (256×256 / 2.56 km)\n"
                           f"chip_id={cid}  in v2_split={'val' if train_idx is not None else '?'}",
                           fontsize=11)
        ax[0, 1].imshow(train_mask > 0.5, cmap="gray")
        ax[0, 1].set_title(f"2024 training mask  ({int((train_mask>0.5).sum())} px)",
                           fontsize=11)
    else:
        ax[0, 0].text(0.5, 0.5, "No 2024 chip found in H5", ha="center", va="center")
        ax[0, 1].text(0.5, 0.5, "(no 2024 mask)", ha="center", va="center")
    ax[0, 0].axis("off"); ax[0, 1].axis("off")

    ax[0, 2].imshow(test_rgb)
    ax[0, 2].set_title(f"2026 mosaic RGB ({h}×{w} px ≈ {h*10/1000:.0f} km)",
                       fontsize=11); ax[0, 2].axis("off")

    pim = ax[1, 0].imshow(probs, cmap="magma", vmin=0, vmax=1)
    ax[1, 0].set_title("2026 probability heatmap (continuous)", fontsize=11)
    ax[1, 0].axis("off")
    plt.colorbar(pim, ax=ax[1, 0], fraction=0.04, pad=0.02)

    overlay85 = test_rgb.copy()
    overlay85[mask_85 > 0] = [1.0, 0.15, 0.15]
    ax[1, 1].imshow(overlay85)
    ax[1, 1].set_title(f"Detection @ thr=0.85 (production)\n"
                       f"{int(mask_85.sum())} px ({mask_85.sum()*100/mask_85.size:.2f}%)",
                       fontsize=11); ax[1, 1].axis("off")

    overlay50 = test_rgb.copy()
    overlay50[mask_50 > 0] = [0.15, 1.0, 0.15]
    ax[1, 2].imshow(overlay50)
    ax[1, 2].set_title(f"Detection @ thr=0.50 (more permissive)\n"
                       f"{int(mask_50.sum())} px ({mask_50.sum()*100/mask_50.size:.2f}%)",
                       fontsize=11); ax[1, 2].axis("off")

    fig.suptitle(f"Site diagnostic: {cid}  —  AOI={h*10/1000:.0f} km, "
                 f"mosaic dates {dates[0]} .. {dates[-1]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path(args.out) if args.out else REPO / f"docs/realworld_unseen_50_2026Q1/diagnostic_{cid}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
