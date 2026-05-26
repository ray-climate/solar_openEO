#!/usr/bin/env python
"""Re-run inference on the 50 cached real-world AOI stacks using BOTH the
pre-fine-tune (R3 R101 dropout02) and post-fine-tune (R3 R101 reviewed) weights,
then render side-by-side comparison PNGs.

For each site:
  Panel 1: 2026 mosaic RGB
  Panel 2: Pre-fine-tune detection (red overlay)
  Panel 3: Post-fine-tune detection (cyan overlay)
  Panel 4: Δ map — green pixels gained, magenta pixels lost, gray=both/neither

Re-uses cached .nc stacks in docs/realworld_unseen_50_2026Q1/.
No OpenEO calls.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

NC_DIR  = REPO / "docs/realworld_unseen_50_2026Q1"
OUT_DIR = REPO / "docs/realworld_unseen_50_2026Q1_compare"

PRE_WEIGHTS  = REPO / "experiments/exp_round3_r101_dropout02/best.weights.h5"
POST_WEIGHTS = REPO / "experiments/exp_round3_r101_reviewed/best.weights.h5"


def load_models():
    """Load pre + post models, share band_stats."""
    from openeo_udp.udf.solar_pv_inference import (
        _build_model, _load_weights_compat, _load_band_stats, _load_registry,
    )
    reg = _load_registry()
    print(f"Building two model instances ({reg['backbone']}) ...", flush=True)
    m_pre  = _build_model(reg)
    m_post = _build_model(reg)
    print(f"Loading pre weights:  {PRE_WEIGHTS}", flush=True)
    _load_weights_compat(m_pre, PRE_WEIGHTS)
    print(f"Loading post weights: {POST_WEIGHTS}", flush=True)
    _load_weights_compat(m_post, POST_WEIGHTS)
    band_stats = _load_band_stats(REPO / reg["band_stats_local"])
    thr = float(reg.get("threshold", 0.85))
    print(f"Threshold: {thr}", flush=True)
    return m_pre, m_post, band_stats, thr


def tiled_predict(model, image_hwc, band_stats, thr):
    """256x256 non-overlapping tiled inference with edge padding."""
    from openeo_udp.udf.solar_pv_inference import normalize_zscore
    h, w, c = image_hwc.shape
    probs = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, 256):
        for x0 in range(0, w, 256):
            tile = image_hwc[y0:y0+256, x0:x0+256, :]
            th, tw = tile.shape[0], tile.shape[1]
            if th < 256 or tw < 256:
                pad = np.zeros((256, 256, c), dtype=np.float32)
                pad[:th, :tw, :] = tile; tile = pad
            normed = normalize_zscore(tile[np.newaxis], band_stats)
            p = model.predict(normed, verbose=0)[0, :, :, 0]
            probs[y0:y0+th, x0:x0+tw] = p[:th, :tw]
    return (probs > thr).astype(np.uint8), probs


def render_compare(site, rgb, mask_pre, mask_post, out_path):
    h, w = mask_pre.shape
    # Delta panel: green = added (post 1, pre 0), magenta = lost (pre 1, post 0),
    # white = unchanged detection (both 1), faint gray = unchanged background (both 0).
    delta = np.zeros((h, w, 3), dtype=np.float32)
    both = (mask_pre == 1) & (mask_post == 1)
    added = (mask_pre == 0) & (mask_post == 1)
    lost  = (mask_pre == 1) & (mask_post == 0)
    delta[both]  = [1.0, 1.0, 1.0]            # both
    delta[added] = [0.15, 0.85, 0.15]         # green: gained
    delta[lost]  = [0.85, 0.15, 0.85]         # magenta: lost

    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(rgb)
    ax[0].set_title(f"RGB ({h}×{w})", fontsize=11); ax[0].axis("off")

    pre_over = rgb.copy(); pre_over[mask_pre > 0] = [1.0, 0.15, 0.15]
    ax[1].imshow(pre_over)
    ax[1].set_title(f"PRE (red)  —  {int(mask_pre.sum())} px", fontsize=11); ax[1].axis("off")

    post_over = rgb.copy(); post_over[mask_post > 0] = [0.15, 1.0, 1.0]
    ax[2].imshow(post_over)
    ax[2].set_title(f"POST (cyan)  —  {int(mask_post.sum())} px", fontsize=11); ax[2].axis("off")

    ax[3].imshow(delta)
    n_added = int(added.sum()); n_lost = int(lost.sum()); n_both = int(both.sum())
    delta_net = n_added - n_lost
    ax[3].set_title(f"Δ  gained={n_added} lost={n_lost} both={n_both}  (net {delta_net:+d})",
                    fontsize=11); ax[3].axis("off")

    fig.suptitle(
        f"{site['chip_id']}  [{site['tier'].upper()}]  {site['continent']}  "
        f"panel_frac={site['panel_frac']:.2f}  AOI={2*site['half_size_km']:.0f} km",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse site list + .nc loader + mosaic builder from the original real-world script.
    spec = importlib.util.spec_from_file_location(
        "rw", REPO / "scripts/19_test_real_world_sites.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    sites = mod.select_sites(n_total=50, unseen_only=True, seed=42)
    print(f"Loaded {len(sites)} sites.")

    m_pre, m_post, band_stats, thr = load_models()

    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic

    rows = []
    for i, s in enumerate(sites, start=1):
        cid = s["chip_id"]
        nc = NC_DIR / f"{cid}_stack.nc"
        if not nc.exists():
            print(f"[{i:>2}/{len(sites)}] SKIP {cid}: no stack")
            continue
        out_path = OUT_DIR / f"compare_{s['tier']}_{cid}.png"
        if out_path.exists():
            print(f"[{i:>2}/{len(sites)}] [skip-existing] {out_path.name}", flush=True)
            continue
        spectral, scl, _ = mod.load_stack(nc)
        composite, info = create_temporal_mosaic(spectral, scl)
        image_hwc = np.transpose(composite[:13], (1, 2, 0))
        rgb = mod.make_rgb(composite)

        mask_pre,  _ = tiled_predict(m_pre,  image_hwc, band_stats, thr)
        mask_post, _ = tiled_predict(m_post, image_hwc, band_stats, thr)
        render_compare(s, rgb, mask_pre, mask_post, out_path)

        added = int(((mask_pre == 0) & (mask_post == 1)).sum())
        lost  = int(((mask_pre == 1) & (mask_post == 0)).sum())
        rows.append({
            "chip_id":     cid,
            "tier":        s["tier"],
            "continent":   s["continent"],
            "lat":         f"{s['lat']:.4f}",
            "lon":         f"{s['lon']:.4f}",
            "panel_frac":  f"{s['panel_frac']:.4f}",
            "pre_px":      int(mask_pre.sum()),
            "post_px":     int(mask_post.sum()),
            "delta_net":   added - lost,
            "added":       added,
            "lost":        lost,
        })
        print(f"[{i:>2}/{len(sites)}] {cid} ({s['tier']}, {s['continent']:<10s}): "
              f"pre={mask_pre.sum():>6}  post={mask_post.sum():>6}  Δ={added-lost:>+7}", flush=True)

    # Summary CSV (only write if we generated any new rows this run)
    summary = OUT_DIR / "summary.csv"
    if rows:
        with summary.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nWrote {summary.relative_to(REPO)}")
    else:
        print("\nAll PNGs already existed (no new rows).")
    print(f"PNGs in {OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
