#!/usr/bin/env python
"""Triage the 167 problem large chips into Type A / B / C.

Inputs (read-only):
  - docs/audit_30k_h5.csv                                           per-chip Dice from the 30K self-audit
  - outputs/final/final_dataset_repacked.h5                          training imagery + GT masks
  - data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg
                                                                     polygon DB with per-polygon predicted install year
  - outputs/final/master_manifest.csv                                chip lat/lon + continent metadata

Output:
  - docs/audit_30k_large_triage.csv                                  per-chip type + signals

Type definitions:
  A — keep & hard-mine. GT correct, model misses.
  B — drop from training. GT inconsistent (polygon present, panels not visible in 2024 imagery).
  C — uncertain; defer.

Classification rule (conservative):
  Let
      F = frac_year_2024 = (area of year==2024 polygons in chip) / (total polygon area in chip)
      P = in-mask "looks like bare ground" score, derived from B11 / NDVI vs healthy baseline
      R = model recall on this chip (from audit)
  Then:
      Type B if F >= 0.50 AND P high AND R < 0.10
      Type A if R low AND in-mask spectrum looks panel-like (P low)
      Type C otherwise
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

from extraction_pipeline.tiling import chip_id_to_bounds  # noqa: E402

AUDIT_CSV   = REPO / "docs/audit_30k_h5.csv"
H5_PATH     = REPO / "outputs/final/final_dataset_repacked.h5"
GPKG_PATH   = REPO / "data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
MASTER_CSV  = REPO / "outputs/final/master_manifest.csv"
OUT_CSV     = REPO / "docs/audit_30k_large_triage.csv"

# Bands used for spectral signature: B4 (idx 3 = red), B8 (idx 7 = NIR), B11 (idx 11 = SWIR1)
BAND_RED   = 3
BAND_NIR   = 7
BAND_SWIR1 = 11

# How many healthy chips to use as the baseline distribution
N_HEALTHY_BASELINE = 200
HEALTHY_SEED = 42


def load_audit() -> list[dict]:
    return list(csv.DictReader(AUDIT_CSV.open()))


def load_master() -> dict[str, dict]:
    rows = {}
    with MASTER_CSV.open() as f:
        for r in csv.DictReader(f):
            rows[r["chip_id_str"]] = r
    return rows


def load_polygons():
    """Load GPKG, reproject to EPSG:3857, repair invalid geometries, build sindex."""
    import geopandas as gpd
    print(f"Loading polygon GPKG ({GPKG_PATH.name}) ...", flush=True)
    gdf = gpd.read_file(GPKG_PATH)
    print(f"  {len(gdf):,} polygons in EPSG:{gdf.crs.to_epsg()}", flush=True)
    if gdf.crs.to_epsg() != 3857:
        print("  reprojecting to EPSG:3857 ...", flush=True)
        gdf = gdf.to_crs("EPSG:3857")
    # Repair invalid geometries (self-intersections, etc.)
    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"  fixing {n_invalid:,} invalid geometries with buffer(0) ...", flush=True)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    # rebuild sindex implicitly when accessed
    return gdf


def compute_inmask_stats(img_chw: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mean reflectance for B4/B8/B11 plus NDVI inside the labeled mask."""
    m = mask.astype(bool)
    n = int(m.sum())
    if n == 0:
        return {"in_b4": float("nan"), "in_b8": float("nan"), "in_b11": float("nan"),
                "in_ndvi": float("nan"), "in_n": 0}
    b4   = img_chw[BAND_RED][m].astype(np.float64).mean()
    b8   = img_chw[BAND_NIR][m].astype(np.float64).mean()
    b11  = img_chw[BAND_SWIR1][m].astype(np.float64).mean()
    eps  = 1e-6
    ndvi = float((b8 - b4) / (b8 + b4 + eps))
    return {"in_b4": float(b4), "in_b8": float(b8), "in_b11": float(b11),
            "in_ndvi": ndvi, "in_n": n}


def compute_chip_year_overlap(chip_col: int, chip_row: int, polys) -> dict[str, float]:
    """For one chip, compute total polygon area + share that is year==2024."""
    from shapely.geometry import box as shp_box
    xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
    chip_geom = shp_box(xmin, ymin, xmax, ymax)
    candidate_idx = list(polys.sindex.intersection((xmin, ymin, xmax, ymax)))
    if not candidate_idx:
        return {"poly_total_area_m2": 0.0, "poly_2024_area_m2": 0.0,
                "frac_year_2024": float("nan"), "n_polys": 0,
                "min_year": float("nan"), "max_year": float("nan")}
    sub = polys.iloc[candidate_idx].copy()
    # Defensive: any still-invalid geometries get buffered to 0 individually
    bad = ~sub.geometry.is_valid
    if bad.any():
        sub.loc[bad, "geometry"] = sub.loc[bad, "geometry"].buffer(0)
    try:
        inter = sub.geometry.intersection(chip_geom)
    except Exception:
        # Per-geometry intersection as a last-resort fallback
        from shapely.errors import GEOSException
        clean = []
        for g in sub.geometry:
            try:
                clean.append(g.intersection(chip_geom))
            except (GEOSException, Exception):
                clean.append(g.buffer(0).intersection(chip_geom))
        import geopandas as gpd
        inter = gpd.GeoSeries(clean, index=sub.index)
    total_area = float(inter.area.sum())
    if total_area <= 0:
        return {"poly_total_area_m2": 0.0, "poly_2024_area_m2": 0.0,
                "frac_year_2024": float("nan"), "n_polys": 0,
                "min_year": float("nan"), "max_year": float("nan")}
    is_2024 = sub["year"].astype(int) == 2024
    area_2024 = float(inter[is_2024.values].area.sum())
    years = sub["year"].astype(int).tolist()
    return {
        "poly_total_area_m2": total_area,
        "poly_2024_area_m2": area_2024,
        "frac_year_2024": area_2024 / total_area,
        "n_polys": int((inter.area > 0).sum()),
        "min_year": float(min(years)) if years else float("nan"),
        "max_year": float(max(years)) if years else float("nan"),
    }


def parse_chip_grid(chip_id: str) -> tuple[int, int]:
    """Parse 'c+004217_r+001757' -> (4217, 1757). Signs preserved."""
    col_part, row_part = chip_id.split("_")
    return int(col_part[1:]), int(row_part[1:])


def classify(signals: dict, baseline: dict) -> tuple[str, str]:
    """Return (type, rationale).

    Rules (in priority order):
      Type B (drop): polygon present, panels NOT visible.
        - in-mask B11 > BARE_B11 (clearly bare-ground signature) AND recall < 0.10  OR
        - frac_year_2024 >= 0.50 AND recall < 0.10
      Type A (keep + hard-mine): panels visible in the imagery, model misses them.
        - in-mask B11 < PANEL_B11 (panel-like signature) AND recall < 0.70
      Type C: ambiguous — moderate B11, mixed signals, defer.
    """
    F = signals.get("frac_year_2024")
    R = signals.get("recall")
    b11 = signals.get("in_b11")
    if F is None or math.isnan(F): F = 0.0
    if R is None or math.isnan(R): R = 0.0

    # Baseline B11 distribution from 200 healthy large chips:
    #   median ~3230, q90 ~3935.
    # Threshold halfway between median and q90 to separate "panel-bright"
    # (clearly above panel median) from "panel-like".
    BARE_B11  = (baseline["b11_q50"] + baseline["b11_q90"]) / 2.0   # ~3580
    PANEL_B11 = baseline["b11_q50"] + 100.0                         # ~3330

    bare_like  = b11 is not None and not math.isnan(b11) and b11 > BARE_B11
    panel_like = b11 is not None and not math.isnan(b11) and b11 < PANEL_B11

    # B has two paths to flag: (a) bare-spectrum + zero recall (the dominant
    # case based on visual inspection); (b) year-flagged polygons + zero recall.
    if bare_like and R < 0.10:
        return "B", f"in_b11={b11:.0f}>{BARE_B11:.0f} (bare-like) AND R={R:.2f}<0.10"
    if F >= 0.50 and R < 0.10:
        return "B", f"frac_year_2024={F:.2f}>=0.50 AND R={R:.2f}<0.10"
    if panel_like and R < 0.70:
        return "A", f"in_b11={b11:.0f}<{PANEL_B11:.0f} (panel-like) AND R={R:.2f}<0.70"
    return "C", f"F={F:.2f}, R={R:.2f}, in_b11={b11 if b11 is None else round(b11)} (between {PANEL_B11:.0f}-{BARE_B11:.0f})"


def build_healthy_baseline(rows: list[dict], chip_idx_map: dict[str, int]) -> dict:
    rng = np.random.default_rng(HEALTHY_SEED)
    pool = [r for r in rows if r["label"] == "1"
            and float(r["panel_frac"]) >= 0.40
            and float(r["dice"]) >= 0.85
            and r["chip_id"] in chip_idx_map]
    n = min(N_HEALTHY_BASELINE, len(pool))
    sample = list(rng.choice(pool, size=n, replace=False))
    print(f"Computing healthy-large baseline from {len(sample)} chips ...", flush=True)
    b4s, b8s, b11s, ndvis = [], [], [], []
    with h5py.File(H5_PATH, "r") as h5:
        for r in sample:
            idx = chip_idx_map[r["chip_id"]]
            img = h5["images"][idx].astype(np.float32)
            mask = (h5["masks"][idx].astype(np.float32) > 0.5).astype(np.uint8)
            stats = compute_inmask_stats(img, mask)
            if stats["in_n"] > 0:
                b4s.append(stats["in_b4"]); b8s.append(stats["in_b8"])
                b11s.append(stats["in_b11"]); ndvis.append(stats["in_ndvi"])
    out = {
        "b4_q50": float(np.median(b4s)),  "b4_q90":  float(np.quantile(b4s,  .90)),
        "b8_q50": float(np.median(b8s)),  "b8_q90":  float(np.quantile(b8s,  .90)),
        "b11_q50": float(np.median(b11s)), "b11_q10": float(np.quantile(b11s, .10)),
        "b11_q90": float(np.quantile(b11s, .90)),
        "ndvi_q50": float(np.median(ndvis)),
        "n_baseline": len(b11s),
    }
    print(f"  healthy in-mask:  median B4={out['b4_q50']:.0f}  B8={out['b8_q50']:.0f}  "
          f"B11={out['b11_q50']:.0f} (q10={out['b11_q10']:.0f}, q90={out['b11_q90']:.0f})  "
          f"NDVI={out['ndvi_q50']:.3f}", flush=True)
    return out


def main() -> None:
    audit = load_audit()
    master = load_master()

    # Index H5 chip_ids -> array index (first occurrence)
    with h5py.File(H5_PATH, "r") as h5:
        all_chip_ids = [c.decode() for c in h5["chip_ids"][:]]
    chip_idx_map: dict[str, int] = {}
    for i, c in enumerate(all_chip_ids):
        if c not in chip_idx_map:
            chip_idx_map[c] = i

    # Build healthy baseline first
    baseline = build_healthy_baseline(audit, chip_idx_map)

    # Filter problem large chips: label=1, panel_frac>=0.40, dice<0.85
    problem = [r for r in audit
               if r["label"] == "1"
               and float(r["panel_frac"]) >= 0.40
               and float(r["dice"]) < 0.85]
    print(f"\n{len(problem):,} problem large chips to triage.\n", flush=True)

    # Load polygon GPKG once
    polys = load_polygons()

    # Process each problem chip
    out_rows: list[dict] = []
    counts = {"A": 0, "B": 0, "C": 0}
    with h5py.File(H5_PATH, "r") as h5:
        for j, r in enumerate(problem):
            cid = r["chip_id"]
            chip_col, chip_row = parse_chip_grid(cid)
            year_stats = compute_chip_year_overlap(chip_col, chip_row, polys)
            if cid in chip_idx_map:
                idx = chip_idx_map[cid]
                img = h5["images"][idx].astype(np.float32)
                mask = (h5["masks"][idx].astype(np.float32) > 0.5).astype(np.uint8)
                spec = compute_inmask_stats(img, mask)
            else:
                spec = {"in_b4": float("nan"), "in_b8": float("nan"),
                        "in_b11": float("nan"), "in_ndvi": float("nan"), "in_n": 0}

            recall = float(r["recall"]) if r["recall"] not in ("", "nan") else 0.0
            precision = float(r["precision"]) if r["precision"] not in ("", "nan") else 0.0
            signals = {
                "frac_year_2024": year_stats["frac_year_2024"],
                "recall": recall, "precision": precision,
                "in_b11": spec["in_b11"], "in_ndvi": spec["in_ndvi"],
            }
            t, rationale = classify(signals, baseline)
            counts[t] += 1

            out_rows.append({
                "chip_id":           cid,
                "type":              t,
                "rationale":         rationale,
                "continent":         r["continent"],
                "lat":               r["lat"], "lon": r["lon"],
                "panel_frac":        r["panel_frac"],
                "dice":              r["dice"],
                "recall":            r["recall"], "precision": r["precision"],
                "gt_px":             r["gt_px"],   "pred_px":   r["pred_px"],
                "frac_year_2024":    f"{year_stats['frac_year_2024']:.4f}" if not math.isnan(year_stats['frac_year_2024']) else "",
                "poly_total_area_m2": f"{year_stats['poly_total_area_m2']:.0f}",
                "poly_2024_area_m2":  f"{year_stats['poly_2024_area_m2']:.0f}",
                "n_polys":            year_stats["n_polys"],
                "min_year":           year_stats["min_year"],
                "max_year":           year_stats["max_year"],
                "in_b4":              f"{spec['in_b4']:.1f}"   if not math.isnan(spec['in_b4'])   else "",
                "in_b8":              f"{spec['in_b8']:.1f}"   if not math.isnan(spec['in_b8'])   else "",
                "in_b11":             f"{spec['in_b11']:.1f}"  if not math.isnan(spec['in_b11'])  else "",
                "in_ndvi":            f"{spec['in_ndvi']:.4f}" if not math.isnan(spec['in_ndvi']) else "",
                "in_n":               spec["in_n"],
            })
            if (j+1) % 20 == 0 or j+1 == len(problem):
                print(f"  [{j+1:>4}/{len(problem)}]  A={counts['A']} B={counts['B']} C={counts['C']}",
                      flush=True)

    # Write
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows: w.writerow(r)
    print(f"\nWrote {len(out_rows)} rows -> {OUT_CSV.relative_to(REPO)}", flush=True)

    print(f"\n=== TRIAGE SUMMARY ===")
    print(f"  Type A (keep & hard-mine): {counts['A']:>3d}")
    print(f"  Type B (drop)            : {counts['B']:>3d}")
    print(f"  Type C (uncertain)       : {counts['C']:>3d}")
    print(f"  Total                    : {sum(counts.values()):>3d}")


if __name__ == "__main__":
    main()
