#!/usr/bin/env python
"""Render the full review gallery for all 1,323 large positives.

For each large chip (panel_frac >= 0.40), produce a 4-panel diagnostic PNG
in worst-Dice-first order:
    Panel 1: 2024 RGB
    Panel 2: GT mask (cyan) over RGB
    Panel 3: polygons-by-year (red = 2024, gray = pre-2024) over RGB
    Panel 4: model detection (red) over RGB

Also emit docs/review_large/manifest.json — one entry per chip with
metadata consumed by the static HTML viewer.

Outputs go to docs/review_large/<rank>_<chip_id>.png so the on-disk and
in-browser sort order are the same (worst Dice = lowest rank).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

from extraction_pipeline.tiling import chip_id_to_bounds  # noqa: E402
from scripts.visualize_triage import render, parse_chip_grid  # noqa: E402

AUDIT_CSV = REPO / "docs/audit_30k_h5.csv"
H5_PATH   = REPO / "outputs/final/final_dataset_repacked.h5"
GPKG_PATH = REPO / "data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
OUT_DIR   = REPO / "docs/review_large"


def main() -> None:
    import geopandas as gpd
    from shapely.geometry import box as shp_box

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(AUDIT_CSV.open()))
    large = [r for r in rows if r["label"] == "1" and float(r["panel_frac"]) >= 0.40]
    large.sort(key=lambda r: float(r["dice"]))  # worst first
    print(f"Large positives: {len(large)}  (worst Dice = {large[0]['dice']}, "
          f"best = {large[-1]['dice']})", flush=True)

    # GPKG once, with invalid-geometry repair
    print("Loading polygon GPKG ...", flush=True)
    polys = gpd.read_file(GPKG_PATH)
    if polys.crs.to_epsg() != 3857:
        polys = polys.to_crs("EPSG:3857")
    invalid = ~polys.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"  fixing {n_invalid:,} invalid geometries ...", flush=True)
        polys.loc[invalid, "geometry"] = polys.loc[invalid, "geometry"].buffer(0)
    print(f"  {len(polys):,} polygons in EPSG:3857", flush=True)

    # H5 chip_id -> index
    with h5py.File(H5_PATH, "r") as h5:
        all_chip_ids = [c.decode() for c in h5["chip_ids"][:]]
    idx_map: dict[str, int] = {}
    for i, c in enumerate(all_chip_ids):
        if c not in idx_map:
            idx_map[c] = i

    # Model
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(registry.get("threshold", 0.85))
    print(f"Model: backbone={registry['backbone']}  threshold={thr}", flush=True)

    pad = max(4, len(str(len(large))))
    manifest_entries = []

    with h5py.File(H5_PATH, "r") as h5:
        for rank, r in enumerate(large, start=1):
            cid = r["chip_id"]
            if cid not in idx_map:
                print(f"  [{rank}] SKIP {cid}: not in H5"); continue
            idx = idx_map[cid]
            img = h5["images"][idx].astype(np.float32)
            gt  = (h5["masks"][idx].astype(np.float32) > 0.5).astype(np.uint8)
            normed = normalize_zscore(np.transpose(img, (1, 2, 0))[None], band_stats)
            pred = (model.predict(normed, verbose=0)[0, :, :, 0] > thr).astype(np.uint8)

            chip_col, chip_row = parse_chip_grid(cid)
            xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
            chip_geom = shp_box(xmin, ymin, xmax, ymax)
            cand = list(polys.sindex.intersection((xmin, ymin, xmax, ymax)))
            if cand:
                sub = polys.iloc[cand].copy()
                bad = ~sub.geometry.is_valid
                if bad.any():
                    sub.loc[bad, "geometry"] = sub.loc[bad, "geometry"].buffer(0)
                try:
                    sub["geometry"] = sub.geometry.intersection(chip_geom)
                except Exception:
                    from shapely.errors import GEOSException
                    clean = []
                    for g in sub.geometry:
                        try:
                            clean.append(g.intersection(chip_geom))
                        except (GEOSException, Exception):
                            clean.append(g.buffer(0).intersection(chip_geom))
                    sub["geometry"] = clean
                sub = sub[~sub.geometry.is_empty]
            else:
                sub = polys.iloc[[]]

            png_name = f"{rank:0{pad}d}_{cid}.png"
            out_path = OUT_DIR / png_name
            # The render() helper expects a meta dict with these keys; add a stub for 'type'+'rationale'
            meta = {
                "type": "?",
                "rationale": "",
                "continent": r["continent"],
                "lat": r["lat"],
                "lon": r["lon"],
                "panel_frac": r["panel_frac"],
                "dice": r["dice"],
                "precision": r["precision"],
                "recall": r["recall"],
                "frac_year_2024": "",
                "in_b11": "",
                "in_ndvi": "",
            }
            render(cid, img, gt, pred, sub, meta, out_path)

            manifest_entries.append({
                "chip_id":   cid,
                "rank":      rank,
                "png":       png_name,
                "dice":      float(r["dice"]),
                "precision": float(r["precision"]) if r["precision"] not in ("", "nan") else None,
                "recall":    float(r["recall"])    if r["recall"]    not in ("", "nan") else None,
                "gt_px":     int(r["gt_px"]),
                "pred_px":   int(r["pred_px"]),
                "panel_frac": float(r["panel_frac"]),
                "continent": r["continent"],
                "lat":       float(r["lat"]) if r["lat"] not in ("", None) else None,
                "lon":       float(r["lon"]) if r["lon"] not in ("", None) else None,
                "split":     r["split"],
            })
            if rank % 50 == 0 or rank == len(large):
                print(f"  [{rank}/{len(large)}]  -> {png_name}", flush=True)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({
        "n_chips": len(manifest_entries),
        "model": {"backbone": registry["backbone"], "threshold": thr,
                  "version": str(registry.get("version", ""))},
        "chips": manifest_entries,
    }, indent=1))
    print(f"\nWrote manifest -> {manifest_path.relative_to(REPO)}", flush=True)
    print(f"Wrote {len(manifest_entries)} PNGs to {OUT_DIR.relative_to(REPO)}/", flush=True)


if __name__ == "__main__":
    main()
