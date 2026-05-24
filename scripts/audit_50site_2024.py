#!/usr/bin/env python
"""Audit Round-3 R101's behaviour on the 2024 training-time imagery for the
50 unseen test sites.

For each chip:
  - load image + GT mask from final_dataset_repacked.h5
  - run inference at the production threshold (0.85)
  - compute Dice / IoU / precision / recall against the GT
  - join continent / lat-lon / panel_frac / v2-split metadata

Outputs:
  - docs/realworld_unseen_50_2026Q1/audit_2024_h5.csv
  - one-page summary printed to stdout (counts in each health bucket).

NOT covered: any OpenEO-pipeline difference or 2024 -> 2026 drift.  This is
purely a check of model health on training-time pixel data.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

OUT_CSV = REPO / "docs/realworld_unseen_50_2026Q1/audit_2024_h5.csv"
H5 = REPO / "outputs/final/final_dataset_repacked.h5"
MASTER = REPO / "outputs/final/master_manifest.csv"
V2_MANIFEST = REPO / "outputs/training_prep/final_v2/split_manifest.csv"


def load_master() -> dict[str, dict]:
    rows = {}
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            rows[r["chip_id_str"]] = r
    return rows


def load_v2_splits() -> dict[str, str]:
    splits = {}
    with open(V2_MANIFEST) as f:
        for r in csv.DictReader(f):
            splits[r["chip_id"]] = r["split"]
    return splits


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "rw", REPO / "scripts/19_test_real_world_sites.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sites = mod.select_sites(n_total=50, unseen_only=True, seed=42)
    chip_ids = [s["chip_id"] for s in sites]
    site_by_id = {s["chip_id"]: s for s in sites}

    master = load_master()
    splits = load_v2_splits()

    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(registry.get("threshold", 0.85))
    print(f"Loaded model: backbone={registry['backbone']}  threshold={thr}")

    # Map chip_id -> H5 index
    with h5py.File(H5, "r") as h5:
        h5_chip_ids = [c.decode() for c in h5["chip_ids"][:]]
    id_to_idx: dict[str, int] = {}
    for i, c in enumerate(h5_chip_ids):
        if c in chip_ids and c not in id_to_idx:
            id_to_idx[c] = i
    missing = [c for c in chip_ids if c not in id_to_idx]
    if missing:
        print(f"WARN: {len(missing)} chips not in H5: {missing[:5]}{'...' if len(missing)>5 else ''}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chip_id", "tier", "split", "continent", "lat", "lon",
        "panel_frac_meta", "gt_px", "pred_px",
        "tp", "fp", "fn", "dice", "iou", "precision", "recall",
        "mean_prob", "max_prob",
    ]
    rows_out: list[dict] = []

    with h5py.File(H5, "r") as h5:
        images = h5["images"]; masks = h5["masks"]
        for cid in chip_ids:
            if cid not in id_to_idx:
                continue
            idx = id_to_idx[cid]
            img_chw = images[idx].astype(np.float32)
            mask_hw = masks[idx].astype(np.float32)
            img_hwc = np.transpose(img_chw, (1, 2, 0))[None]
            normed = normalize_zscore(img_hwc, band_stats)
            probs = model.predict(normed, verbose=0)[0, :, :, 0]
            pred = (probs > thr).astype(np.uint8)
            gt = (mask_hw > 0.5).astype(np.uint8)
            tp = int(((gt == 1) & (pred == 1)).sum())
            fp = int(((gt == 0) & (pred == 1)).sum())
            fn = int(((gt == 1) & (pred == 0)).sum())
            dice = 2*tp / max(2*tp + fp + fn, 1)
            iou  = tp / max(tp + fp + fn, 1)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            s = site_by_id[cid]
            m = master.get(cid, {})
            rows_out.append({
                "chip_id":          cid,
                "tier":             s["tier"],
                "split":            splits.get(cid, "?"),
                "continent":        s["continent"],
                "lat":              s["lat"],
                "lon":              s["lon"],
                "panel_frac_meta":  s["panel_frac"],
                "gt_px":            int(gt.sum()),
                "pred_px":          int(pred.sum()),
                "tp":               tp, "fp": fp, "fn": fn,
                "dice":             round(dice, 4),
                "iou":              round(iou, 4),
                "precision":        round(prec, 4),
                "recall":           round(rec, 4),
                "mean_prob":        round(float(probs.mean()), 4),
                "max_prob":         round(float(probs.max()), 4),
            })
            print(f"  {cid:<22s} {s['tier']:<6s} {splits.get(cid,'?'):<4s} {s['continent']:<10s} "
                  f"pf={s['panel_frac']:.2f}  dice={dice:.3f}  P={prec:.2f}  R={rec:.2f}  "
                  f"gt={gt.sum():>5} pred={pred.sum():>5}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nWrote {len(rows_out)} rows -> {OUT_CSV.relative_to(REPO)}")

    # Summary by bucket
    healthy   = [r for r in rows_out if r["dice"] >= 0.85]
    middling  = [r for r in rows_out if 0.50 <= r["dice"] < 0.85]
    failing   = [r for r in rows_out if r["dice"] < 0.50]
    print(f"\n=== HEALTH BUCKETS (Dice on 2024 training-time imagery) ===")
    print(f"  healthy  (>=0.85): {len(healthy):>2d} / {len(rows_out)}")
    print(f"  middling (0.50-0.85): {len(middling):>2d} / {len(rows_out)}")
    print(f"  failing  (<0.50): {len(failing):>2d} / {len(rows_out)}")
    overall_dice = (sum(r["tp"] for r in rows_out) * 2
                    / max(sum(2*r["tp"] + r["fp"] + r["fn"] for r in rows_out), 1))
    median_dice = float(np.median([r["dice"] for r in rows_out]))
    print(f"\n  Per-chip median dice: {median_dice:.4f}")
    print(f"  Pixel-pooled dice   : {overall_dice:.4f}")

    if failing:
        print(f"\n=== FAILING SITES (Dice < 0.50) ===")
        for r in sorted(failing, key=lambda x: x["dice"]):
            print(f"  {r['chip_id']:<22s} {r['tier']:<6s} {r['continent']:<10s} "
                  f"lat={r['lat']:>+6.1f}  pf={r['panel_frac_meta']:.2f}  "
                  f"dice={r['dice']:.3f}  R={r['recall']:.2f}")

    # Continent breakdown
    print(f"\n=== BY CONTINENT (median dice / N) ===")
    by_cont: dict[str, list[float]] = {}
    for r in rows_out:
        by_cont.setdefault(r["continent"], []).append(r["dice"])
    for c, ds in sorted(by_cont.items(), key=lambda x: float(np.median(x[1]))):
        print(f"  {c:<12s}  n={len(ds):>2d}  median={float(np.median(ds)):.3f}  "
              f"min={min(ds):.3f}  max={max(ds):.3f}")

    # Tier breakdown
    print(f"\n=== BY TIER (median dice / N) ===")
    for tier in ("small", "medium", "large"):
        ds = [r["dice"] for r in rows_out if r["tier"] == tier]
        if ds:
            print(f"  {tier:<6s}  n={len(ds):>2d}  median={float(np.median(ds)):.3f}  "
                  f"min={min(ds):.3f}  max={max(ds):.3f}")


if __name__ == "__main__":
    main()
