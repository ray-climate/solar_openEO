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
    train_img_hwc = None
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
            # transpose to (H,W,C) for model input
            train_img_hwc = np.transpose(img_chw, (1, 2, 0))

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
    print(f"Detections (2026): thr=0.85 -> {int(mask_85.sum())} px,  "
          f"thr=0.50 -> {int(mask_50.sum())} px  "
          f"(ratio 0.50/0.85 = {mask_50.sum()/max(mask_85.sum(),1):.2f}x)")

    # ALSO: run inference on the 2024 training chip itself to prove the model
    # is healthy on its training-time distribution.
    train_probs = None
    train_pred_85 = None
    if train_img_hwc is not None:
        n_train = normalize_zscore(train_img_hwc[np.newaxis], band_stats)
        train_p = model.predict(n_train, verbose=0)
        train_probs = train_p[0, :, :, 0]
        train_pred_85 = (train_probs > 0.85).astype(np.uint8)
        gt_bin = (train_mask > 0.5).astype(np.uint8)
        tp = int(((gt_bin == 1) & (train_pred_85 == 1)).sum())
        fp = int(((gt_bin == 0) & (train_pred_85 == 1)).sum())
        fn = int(((gt_bin == 1) & (train_pred_85 == 0)).sum())
        dice_2024 = 2*tp / max(2*tp + fp + fn, 1)
        print(f"2024 inference dice (this chip): {dice_2024:.4f}  "
              f"(GT={gt_bin.sum()}, pred={train_pred_85.sum()}, TP={tp}, FP={fp}, FN={fn})")

    # ---- Render ----
    # 3 rows × 3 cols. Row 0 = 2024 (RGB, GT mask, MODEL prediction on 2024).
    # Row 1 = 2026 (RGB, prob heatmap, detection @ 0.85).  Row 2 = thresholds + spare.
    fig, ax = plt.subplots(3, 3, figsize=(20, 19))

    # ----- 2024 row -----
    if train_rgb is not None:
        ax[0, 0].imshow(train_rgb)
        ax[0, 0].set_title(f"2024 training chip RGB (256×256 / 2.56 km)\n"
                           f"chip_id={cid}  in v2_split={'val' if train_idx is not None else '?'}",
                           fontsize=11)
        ax[0, 1].imshow(train_mask > 0.5, cmap="gray")
        ax[0, 1].set_title(f"2024 ground-truth mask (hand-labelled, NOT a model output)\n"
                           f"{int((train_mask>0.5).sum())} px",
                           fontsize=11)
        if train_pred_85 is not None:
            overlay_2024 = train_rgb.copy()
            overlay_2024[train_pred_85 > 0] = [1.0, 0.15, 0.15]
            ax[0, 2].imshow(overlay_2024)
            gt_bin = (train_mask > 0.5).astype(np.uint8)
            tp = int(((gt_bin == 1) & (train_pred_85 == 1)).sum())
            fp = int(((gt_bin == 0) & (train_pred_85 == 1)).sum())
            fn = int(((gt_bin == 1) & (train_pred_85 == 0)).sum())
            dice_2024 = 2*tp / max(2*tp + fp + fn, 1)
            ax[0, 2].set_title(f"2024 MODEL PREDICTION on training imagery (thr=0.85)\n"
                               f"{int(train_pred_85.sum())} px, Dice vs GT = {dice_2024:.3f}",
                               fontsize=11)
    else:
        for j in range(3):
            ax[0, j].text(0.5, 0.5, "No 2024 chip found in H5", ha="center", va="center")
    for j in range(3):
        ax[0, j].axis("off")

    # ----- 2026 row 1: RGB + prob + detection -----
    ax[1, 0].imshow(test_rgb)
    ax[1, 0].set_title(f"2026 mosaic RGB ({h}×{w} px ≈ {h*10/1000:.0f} km)",
                       fontsize=11); ax[1, 0].axis("off")

    pim = ax[1, 1].imshow(probs, cmap="magma", vmin=0, vmax=1)
    ax[1, 1].set_title("2026 probability heatmap (raw sigmoid output)",
                       fontsize=11); ax[1, 1].axis("off")
    plt.colorbar(pim, ax=ax[1, 1], fraction=0.04, pad=0.02)

    overlay85 = test_rgb.copy()
    overlay85[mask_85 > 0] = [1.0, 0.15, 0.15]
    ax[1, 2].imshow(overlay85)
    ax[1, 2].set_title(f"2026 MODEL PREDICTION @ thr=0.85 (production)\n"
                       f"{int(mask_85.sum())} px ({mask_85.sum()*100/mask_85.size:.2f}%)",
                       fontsize=11); ax[1, 2].axis("off")

    # ----- 2026 row 2: thr=0.50 + thr=0.20 -----
    mask_20 = (probs > 0.20).astype(np.uint8)

    overlay50 = test_rgb.copy()
    overlay50[mask_50 > 0] = [0.15, 1.0, 0.15]
    ax[2, 0].imshow(overlay50)
    ax[2, 0].set_title(f"2026 MODEL PREDICTION @ thr=0.50\n"
                       f"{int(mask_50.sum())} px ({mask_50.sum()*100/mask_50.size:.2f}%)",
                       fontsize=11); ax[2, 0].axis("off")

    overlay20 = test_rgb.copy()
    overlay20[mask_20 > 0] = [0.15, 0.6, 1.0]
    ax[2, 1].imshow(overlay20)
    ax[2, 1].set_title(f"2026 MODEL PREDICTION @ thr=0.20 (very permissive)\n"
                       f"{int(mask_20.sum())} px ({mask_20.sum()*100/mask_20.size:.2f}%)",
                       fontsize=11); ax[2, 1].axis("off")

    ax[2, 2].axis("off")
    ax[2, 2].text(0.5, 0.5,
                  "Interpretation:\n\n"
                  "Compare 2024 model prediction (top-right) with 2024 GT (top-middle).\n"
                  "If they match → model is healthy on training-time distribution.\n\n"
                  "Then compare 2026 prediction at thr=0.85, 0.50, 0.20 (middle-right, bottom-left, bottom-middle).\n"
                  "If panels visible in 2026 RGB are still missing at thr=0.20 → real model blind spot\n"
                  "(model assigns near-zero probability to those pixels).\n\n"
                  "If they only appear at thr=0.20 → threshold/calibration issue (model sees them but\n"
                  "with low confidence).",
                  ha="center", va="center", fontsize=10, wrap=True)

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
