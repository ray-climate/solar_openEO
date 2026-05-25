#!/usr/bin/env python
"""Render diagnostic PNGs for the triaged problem chips, grouped by type.

For every chip in docs/audit_30k_large_triage.csv produce a 4-panel figure:
  1. 2024 chip RGB
  2. GT mask overlay (cyan) on RGB
  3. Polygons color-coded by year (red = 2024, gray = pre-2024) on RGB
  4. Model detection overlay (red) on RGB

Output: docs/audit_30k_large_triage/{type_A,type_B,type_C}/dice_<dice>_<chip_id>_triage.png
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
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
from extraction_pipeline.tiling import chip_id_to_bounds  # noqa: E402

TRIAGE_CSV = REPO / "docs/audit_30k_large_triage.csv"
H5_PATH    = REPO / "outputs/final/final_dataset_repacked.h5"
GPKG_PATH  = REPO / "data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
OUT_DIR    = REPO / "docs/audit_30k_large_triage"


def make_rgb(img_chw: np.ndarray) -> np.ndarray:
    rgb = np.transpose(img_chw[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
    valid = rgb[rgb > 0]
    lo = np.percentile(valid, 2) if valid.size else 0
    hi = np.percentile(valid, 98) if valid.size else 1
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def parse_chip_grid(chip_id: str) -> tuple[int, int]:
    col_part, row_part = chip_id.split("_")
    return int(col_part[1:]), int(row_part[1:])


def chip_world_to_pixel(geom_xy: np.ndarray, chip_col: int, chip_row: int) -> np.ndarray:
    """Map an (N,2) array of EPSG:3857 (x,y) into 256×256 pixel space."""
    xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
    px = (geom_xy[:, 0] - xmin) / (xmax - xmin) * 256
    py = (ymax - geom_xy[:, 1]) / (ymax - ymin) * 256
    return np.stack([px, py], axis=1)


def render(chip_id: str, img_chw: np.ndarray, gt: np.ndarray, pred: np.ndarray,
           polys_in_chip, meta: dict, out_path: Path) -> None:
    rgb = make_rgb(img_chw)
    chip_col, chip_row = parse_chip_grid(chip_id)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))

    ax[0].imshow(rgb); ax[0].set_title("2024 RGB"); ax[0].axis("off")

    gt_overlay = rgb.copy()
    gt_overlay[gt > 0] = [0.15, 1.0, 1.0]
    ax[1].imshow(gt_overlay)
    ax[1].set_title(f"GT mask (cyan)  {int(gt.sum())} px"); ax[1].axis("off")

    ax[2].imshow(rgb)
    n_2024 = 0; n_other = 0
    if polys_in_chip is not None and not polys_in_chip.empty:
        for _, row in polys_in_chip.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            year = int(row["year"])
            color = "red" if year == 2024 else "lightgray"
            geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for g in geoms:
                xy = np.array(g.exterior.coords)
                if xy.size == 0:
                    continue
                px = chip_world_to_pixel(xy, chip_col, chip_row)
                ax[2].add_patch(MplPolygon(px, closed=True, fill=False,
                                           edgecolor=color, linewidth=1.8,
                                           alpha=0.95))
            if year == 2024: n_2024 += 1
            else: n_other += 1
    ax[2].set_xlim(0, 256); ax[2].set_ylim(256, 0)
    ax[2].set_title(f"Polygons by year  (red=2024 [{n_2024}], gray=pre-2024 [{n_other}])")
    ax[2].axis("off")

    det_overlay = rgb.copy()
    det_overlay[pred > 0] = [1.0, 0.15, 0.15]
    ax[3].imshow(det_overlay)
    ax[3].set_title(f"Model detection (red)  {int(pred.sum())} px")
    ax[3].axis("off")

    pf = float(meta["panel_frac"])
    fy = meta.get("frac_year_2024", "")
    fig.suptitle(
        f"{chip_id}  [Type {meta['type']}]  |  {meta['continent']}  "
        f"lat={float(meta['lat']):+.2f}  lon={float(meta['lon']):+.2f}  "
        f"pf={pf:.2f}  Dice={meta['dice']}  P={meta['precision']}  R={meta['recall']}  "
        f"|  frac_year_2024={fy}  in_b11={meta.get('in_b11','')}  ndvi={meta.get('in_ndvi','')}  "
        f"|  {meta['rationale']}",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    import geopandas as gpd
    from shapely.geometry import box as shp_box

    rows = list(csv.DictReader(TRIAGE_CSV.open()))
    print(f"Triage CSV: {len(rows)} chips", flush=True)

    # Load GPKG once + reproject + repair invalid geometries
    print("Loading GPKG ...", flush=True)
    polys = gpd.read_file(GPKG_PATH)
    if polys.crs.to_epsg() != 3857:
        polys = polys.to_crs("EPSG:3857")
    invalid = ~polys.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"  fixing {n_invalid:,} invalid geometries with buffer(0) ...", flush=True)
        polys.loc[invalid, "geometry"] = polys.loc[invalid, "geometry"].buffer(0)
    print(f"  {len(polys):,} polygons in EPSG:3857", flush=True)

    # H5 chip_id -> index
    with h5py.File(H5_PATH, "r") as h5:
        all_chip_ids = [c.decode() for c in h5["chip_ids"][:]]
    idx_map: dict[str, int] = {}
    for i, c in enumerate(all_chip_ids):
        if c not in idx_map:
            idx_map[c] = i

    # Run inference for the detection panel
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(registry.get("threshold", 0.85))
    print(f"Model: backbone={registry['backbone']}  threshold={thr}", flush=True)

    n_done = 0
    with h5py.File(H5_PATH, "r") as h5:
        for r in rows:
            cid = r["chip_id"]
            t   = r["type"]
            if cid not in idx_map:
                continue
            idx = idx_map[cid]
            img = h5["images"][idx].astype(np.float32)
            gt  = (h5["masks"][idx].astype(np.float32) > 0.5).astype(np.uint8)
            normed = normalize_zscore(np.transpose(img, (1,2,0))[None], band_stats)
            pred = (model.predict(normed, verbose=0)[0,:,:,0] > thr).astype(np.uint8)

            chip_col, chip_row = parse_chip_grid(cid)
            xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
            chip_geom = shp_box(xmin, ymin, xmax, ymax)
            cand = list(polys.sindex.intersection((xmin, ymin, xmax, ymax)))
            if cand:
                sub = polys.iloc[cand].copy()
                bad = ~sub.geometry.is_valid
                if bad.any():
                    sub.loc[bad, "geometry"] = sub.loc[bad, "geometry"].buffer(0)
                # clip to chip — fall back to per-geometry intersection on errors
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

            bucket_dir = OUT_DIR / f"type_{t}"
            bucket_dir.mkdir(parents=True, exist_ok=True)
            out_path = bucket_dir / f"dice_{float(r['dice']):.3f}_{cid}_triage.png"
            render(cid, img, gt, pred, sub, r, out_path)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  [{n_done}/{len(rows)}]", flush=True)
    print(f"\nWrote {n_done} PNGs -> {OUT_DIR.relative_to(REPO)}/", flush=True)


if __name__ == "__main__":
    main()
