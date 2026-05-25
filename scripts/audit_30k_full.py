#!/usr/bin/env python
"""Full self-audit: run R3 R101 (the deployed model) on every chip in
final_dataset_repacked.h5 and compute Dice/IoU/precision/recall against
the hand-labelled ground-truth masks.

Output:
    docs/realworld_unseen_50_2026Q1/../audit_30k_h5.csv   (per-chip metrics + metadata)
    plus a printed summary: bucket counts, by-continent, by-split, by-tier.

This is the diagnostic that tells us which chips in our 30K-chip training
set the model gets wrong. Failures cluster -> target Phase 3 expansion.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

H5 = REPO / "outputs/final/final_dataset_repacked.h5"
MASTER = REPO / "outputs/final/master_manifest.csv"
V2_MANIFEST = REPO / "outputs/training_prep/final_v2/split_manifest.csv"
OUT_CSV = REPO / "docs/audit_30k_h5.csv"

BATCH_SIZE = 32  # chips per model.predict call


def load_master() -> dict[str, dict]:
    rows = {}
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            rows[r["chip_id_str"]] = r
    return rows


def load_v2_splits() -> dict[str, str]:
    s = {}
    with open(V2_MANIFEST) as f:
        for r in csv.DictReader(f):
            s[r["chip_id"]] = r["split"]
    return s


def tier_for(panel_frac: float) -> str:
    if panel_frac < 0.10: return "small"
    if panel_frac < 0.40: return "medium"
    return "large"


def main() -> None:
    master = load_master()
    splits = load_v2_splits()

    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(registry.get("threshold", 0.85))
    print(f"Loaded model: backbone={registry['backbone']}  threshold={thr}")
    print(f"H5 path: {H5.relative_to(REPO)}")
    print(f"Master manifest: {len(master):,} rows  |  v2 splits: {len(splits):,}")

    rows_out = []
    t_start = time.time()
    # Incremental CSV writer — flush every chunk so a killed job still leaves data.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_handle = OUT_CSV.open("w", newline="")
    csv_writer = None
    CSV_FIELDS = [
        "chip_id", "label", "tier", "split", "continent", "neg_category",
        "lat", "lon", "panel_frac",
        "gt_px", "pred_px", "tp", "fp", "fn",
        "dice", "iou", "precision", "recall", "mean_prob", "max_prob",
    ]
    csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    csv_handle.flush()

    with h5py.File(H5, "r") as h5:
        all_chip_ids = [c.decode() for c in h5["chip_ids"][:]]
        labels = h5["labels"][:]

        n_total = len(all_chip_ids)
        print(f"H5 chip count: {n_total:,}")

        # Dedupe to first-occurrence indices, then process sequentially.
        seen: set[str] = set()
        unique_idx_list: list[int] = []
        for i, cid in enumerate(all_chip_ids):
            if cid in seen:
                continue
            seen.add(cid)
            unique_idx_list.append(i)
        print(f"Unique chip_ids: {len(unique_idx_list):,}", flush=True)

        # Single-batch-at-a-time pattern (proven in audit_50site_2024).
        # H5 random-access reads for batches of 32 work fine and don't degrade.
        n_proc = 0
        for start in range(0, len(unique_idx_list), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(unique_idx_list))
            batch_indices = unique_idx_list[start:end]
            imgs = h5["images"][batch_indices].astype(np.float32)
            gts_arr = (h5["masks"][batch_indices].astype(np.float32) > 0.5).astype(np.uint8)
            gts = gts_arr
            imgs_hwc = np.transpose(imgs, (0, 2, 3, 1))
            normed = normalize_zscore(imgs_hwc, band_stats)
            probs = model.predict(normed, verbose=0)
            preds = (probs[..., 0] > thr).astype(np.uint8)

            for k, h5_idx in enumerate(batch_indices):
                cid = all_chip_ids[h5_idx]
                gt = gts[k]; pred = preds[k]
                tp = int(((gt == 1) & (pred == 1)).sum())
                fp = int(((gt == 0) & (pred == 1)).sum())
                fn = int(((gt == 1) & (pred == 0)).sum())
                gt_px = int(gt.sum()); pred_px = int(pred.sum())
                if labels[h5_idx] == 0:
                    # negative chip — define "dice" as 1 - false-positive rate so we
                    # have a comparable metric; the gt is empty.
                    dice = 1.0 if pred_px == 0 else 0.0
                    iou = 1.0 if pred_px == 0 else 0.0
                    prec = float("nan"); rec = float("nan")
                else:
                    dice = 2 * tp / max(2*tp + fp + fn, 1)
                    iou  = tp / max(tp + fp + fn, 1)
                    prec = tp / max(tp + fp, 1)
                    rec  = tp / max(tp + fn, 1)

                m = master.get(cid, {})
                pf = float(m.get("panel_frac")) if (m.get("panel_frac") and m.get("panel_frac") != "") else 0.0
                tier = tier_for(pf) if labels[h5_idx] == 1 else "negative"
                rows_out.append({
                    "chip_id": cid,
                    "label": int(labels[h5_idx]),
                    "tier": tier,
                    "split": splits.get(cid, "?"),
                    "continent": m.get("continent", "?"),
                    "neg_category": m.get("neg_category", ""),
                    "lat": m.get("chip_center_lat", ""),
                    "lon": m.get("chip_center_lon", ""),
                    "panel_frac": round(pf, 4),
                    "gt_px": gt_px,
                    "pred_px": pred_px,
                    "tp": tp, "fp": fp, "fn": fn,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "precision": round(prec, 4) if not np.isnan(prec) else "",
                    "recall": round(rec, 4) if not np.isnan(rec) else "",
                    "mean_prob": round(float(probs[k, :, :, 0].mean()), 4),
                    "max_prob": round(float(probs[k, :, :, 0].max()), 4),
                })
            # Flush this batch's rows to disk immediately
            for r in rows_out[-len(batch_indices):]:
                csv_writer.writerow(r)
            csv_handle.flush()
            n_proc += len(batch_indices)
            if n_proc % 320 == 0 or n_proc == len(unique_idx_list):
                rate = n_proc / max(time.time() - t_start, 1e-3)
                eta = (len(unique_idx_list) - n_proc) / max(rate, 1e-3)
                print(f"  [{n_proc:>6,}/{len(unique_idx_list):,}]  rate={rate:.1f} chip/s  ETA={eta/60:.1f} min", flush=True)

    csv_handle.close()
    print(f"\nWrote {len(rows_out):,} rows -> {OUT_CSV.relative_to(REPO)}")

    # ------------------- Summary -------------------
    pos = [r for r in rows_out if r["label"] == 1]
    neg = [r for r in rows_out if r["label"] == 0]
    print(f"\n=== HEALTH BUCKETS (positives, n={len(pos):,}) ===")
    healthy  = [r for r in pos if float(r["dice"]) >= 0.85]
    middling = [r for r in pos if 0.50 <= float(r["dice"]) < 0.85]
    failing  = [r for r in pos if float(r["dice"]) < 0.50]
    print(f"  healthy  (Dice >= 0.85): {len(healthy):>6,} ({100*len(healthy)/len(pos):.1f}%)")
    print(f"  middling (0.50-0.85)   : {len(middling):>6,} ({100*len(middling)/len(pos):.1f}%)")
    print(f"  failing  (< 0.50)      : {len(failing):>6,} ({100*len(failing)/len(pos):.1f}%)")
    if pos:
        ds = [float(r["dice"]) for r in pos]
        print(f"  median dice            : {np.median(ds):.4f}")
        print(f"  p25 / p75              : {np.quantile(ds, .25):.4f} / {np.quantile(ds, .75):.4f}")
        rec = [float(r["recall"]) for r in pos if r["recall"] != ""]
        pre = [float(r["precision"]) for r in pos if r["precision"] != ""]
        print(f"  median recall          : {np.median(rec):.4f}")
        print(f"  median precision       : {np.median(pre):.4f}")

    if neg:
        fp_rate = [r["pred_px"] / 65536 for r in neg]
        n_fp = sum(1 for r in neg if r["pred_px"] > 0)
        print(f"\n=== NEGATIVES (n={len(neg):,}) ===")
        print(f"  chips with any false positive: {n_fp:,} ({100*n_fp/len(neg):.1f}%)")
        print(f"  median FP-pixel rate         : {np.median(fp_rate)*100:.4f}%")

    # By split
    print(f"\n=== POSITIVES BY V2 SPLIT (median dice) ===")
    for sp in ("train", "val", "test", "?"):
        ds = [float(r["dice"]) for r in pos if r["split"] == sp]
        if ds:
            print(f"  {sp:<6s}  n={len(ds):>6,}  median={np.median(ds):.4f}  "
                  f"q25={np.quantile(ds, .25):.4f}  q75={np.quantile(ds, .75):.4f}")

    # By continent
    print(f"\n=== POSITIVES BY CONTINENT (median dice) ===")
    cont_groups: dict[str, list[float]] = {}
    for r in pos:
        cont_groups.setdefault(r["continent"], []).append(float(r["dice"]))
    for c, ds in sorted(cont_groups.items(), key=lambda kv: np.median(kv[1])):
        print(f"  {c:<12s} n={len(ds):>6,}  median={np.median(ds):.4f}  "
              f"q25={np.quantile(ds, .25):.4f}  q75={np.quantile(ds, .75):.4f}")

    # By tier
    print(f"\n=== POSITIVES BY TIER (median dice) ===")
    for t in ("small", "medium", "large"):
        ds = [float(r["dice"]) for r in pos if r["tier"] == t]
        if ds:
            print(f"  {t:<6s}  n={len(ds):>6,}  median={np.median(ds):.4f}  "
                  f"q25={np.quantile(ds, .25):.4f}  q75={np.quantile(ds, .75):.4f}")

    # By panel_frac decile
    print(f"\n=== POSITIVES BY panel_frac DECILE (median dice) ===")
    if pos:
        pfs = np.array([r["panel_frac"] for r in pos])
        dices = np.array([float(r["dice"]) for r in pos])
        deciles = np.quantile(pfs, np.linspace(0, 1, 11))
        for i in range(10):
            lo, hi = deciles[i], deciles[i+1]
            mask = (pfs >= lo) & (pfs <= hi if i == 9 else pfs < hi)
            if mask.sum() > 0:
                print(f"  decile {i+1}  pf [{lo:.3f},{hi:.3f}]  n={int(mask.sum()):>5}  median dice={np.median(dices[mask]):.4f}")


if __name__ == "__main__":
    main()
