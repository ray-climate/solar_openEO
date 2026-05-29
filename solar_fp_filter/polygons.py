"""Phase 1 — build the polygon training set for the FP classifier.

Three sources of labelled training polygons:

  * Positives — stratified sample of solar PV polygons from the project
    polygon DB, each expanded into three labelled time windows
    (pre_build / transition / fully_built). The pre_build window doubles
    as a same-location hard negative.

  * Land negatives — random continental land polygons placed at least
    5 km from any known PV polygon, with size matched to the positives'
    visible-area distribution.

  * Water / industrial negatives — optional, sourced from external
    GeoPackages dropped into ``data/``. Skipped if the files don't
    exist; the module still produces a usable training set without
    them, but with weaker discrimination on those FP classes.

Outputs a single GeoPackage at ``outputs/fp_classifier/polygons.gpkg``
in EPSG:4326, with the schema described in the project plan.
"""
from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

REPO = Path(__file__).resolve().parent.parent
PV_GPKG = REPO / "data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
OUT_GPKG = REPO / "outputs/fp_classifier/polygons.gpkg"

# Optional negative-source GeoPackages — used if present. See README.
EXTRA_WATER_GPKG      = REPO / "data/fp_negatives_water.gpkg"
EXTRA_INDUSTRIAL_GPKG = REPO / "data/fp_negatives_industrial.gpkg"

LAT_BANDS = [(-60, -30), (-30, 0), (0, 30), (30, 50), (50, 72)]
MIN_YEAR_BUILT = 2019         # so (year-2) still falls in stable S2 L2A era
MIN_AREA_M2    = 1_000        # below this aggregate_spatial is too noisy
PV_BUFFER_KM   = 5.0          # land negatives must be at least this far from any PV


# ---------------------------------------------------------------------------
# Positives
# ---------------------------------------------------------------------------

def _area_tier(area_m2: float) -> str:
    if area_m2 < 10_000:    return "small"
    if area_m2 < 100_000:   return "medium"
    if area_m2 < 1_000_000: return "large"
    return "huge"


def load_pv_polygons() -> gpd.GeoDataFrame:
    """Read polygon DB, compute true area in EPSG:3857, filter to
    ``year >= MIN_YEAR_BUILT`` and ``area_m2 >= MIN_AREA_M2``.
    """
    gdf = gpd.read_file(PV_GPKG)
    gdf = gdf.to_crs("EPSG:3857")
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[(gdf["year"] >= MIN_YEAR_BUILT) & (gdf["area_m2"] >= MIN_AREA_M2)].copy()
    gdf["lat_band"] = pd.cut(
        gdf["latitude"], bins=[b[0] for b in LAT_BANDS] + [LAT_BANDS[-1][1]],
        labels=[f"lat_{a}_{b}" for a, b in LAT_BANDS], include_lowest=True,
    )
    gdf["area_tier"] = gdf["area_m2"].apply(_area_tier)
    return gdf


def sample_positives(n_target: int = 10_000, seed: int = 42) -> gpd.GeoDataFrame:
    """Stratified sample of PV polygons by (lat_band x year x area_tier).
    Returns a GeoDataFrame in EPSG:3857 with one row per spatial polygon.
    """
    rng = np.random.default_rng(seed)
    gdf = load_pv_polygons()
    keys = ["lat_band", "year", "area_tier"]
    groups = gdf.groupby(keys, observed=True)
    n_groups = len(groups)
    per_group = max(1, n_target // max(n_groups, 1))
    picked = []
    for _, sub in groups:
        take = min(per_group, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        picked.append(sub.iloc[idx])
    picked_df = pd.concat(picked, ignore_index=True)
    # Trim or top up to the target
    if len(picked_df) > n_target:
        idx = rng.choice(len(picked_df), size=n_target, replace=False)
        picked_df = picked_df.iloc[idx].reset_index(drop=True)
    print(f"  positives: sampled {len(picked_df):,} from {len(gdf):,} eligible "
          f"(stratified across {n_groups} cells)", flush=True)
    return gpd.GeoDataFrame(picked_df, geometry="geometry", crs="EPSG:3857")


def expand_positive_windows(positives_3857: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per (polygon, window). Three windows per positive:
        pre_build, transition, fully_built.
    """
    rows = []
    for _, r in positives_3857.iterrows():
        y = int(r["year"])
        base = {
            "polygon_id":    f"pv_{r.name}",
            "lat":           float(r["latitude"]),
            "lon":           float(r["longitude"]),
            "area_m2":       float(r["area_m2"]),
            "year_built":    y,
            "negative_source": None,
            "geometry":      r["geometry"],
        }
        for window_name, ystart, yend, label in [
            ("pre_build",   y - 2, y - 1, "not_pv"),
            ("transition",  y - 1, y + 1, "becoming_pv"),
            ("fully_built", y + 1, y + 2, "pv"),
        ]:
            rows.append({
                **base,
                "sample_id":    f"pv_{r.name}_{window_name}",
                "class":        label,
                "window_name":  window_name,
                "window_start": f"{ystart}-01-01",
                "window_end":   f"{yend}-01-01",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Land negatives
# ---------------------------------------------------------------------------

def _natural_earth_land_valid() -> gpd.GeoDataFrame:
    """Natural Earth country polygons in EPSG:4326 with invalid geometries
    fixed (some have topology issues near antimeridian/poles)."""
    path = gpd.datasets.get_path("naturalearth_lowres")
    land = gpd.read_file(path)
    land = land[land["continent"] != "Antarctica"].copy()
    bad = ~land.is_valid
    if bad.any():
        land.loc[bad, "geometry"] = land.loc[bad, "geometry"].buffer(0)
    return land  # CRS = EPSG:4326


def _random_points_in_polygon(poly: Polygon, n: int, rng) -> list[Point]:
    """Rejection sampling — uniform random points inside the polygon."""
    minx, miny, maxx, maxy = poly.bounds
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            out.append(p)
    return out


def sample_land_negatives(positives_3857: gpd.GeoDataFrame,
                          n_target: int = 5_000,
                          seed: int = 43) -> pd.DataFrame:
    """Random points on continents at least PV_BUFFER_KM from any PV polygon,
    expanded into polygon buffers matching the positives' size distribution.

    Strategy: dissolve in EPSG:4326 (cheap), allocate per-continent by
    area-weight in 4326 (good enough for budgeting), sample points in
    4326, then reproject points + buffer in 3857 (PV-buffer + area
    matching are in metres so they need 3857).
    """
    rng = np.random.default_rng(seed)
    land_4326 = _natural_earth_land_valid()
    # Build the PV exclusion buffer in 3857 (so radius is in metres)
    pv_buf_3857 = positives_3857.copy()
    pv_buf_3857["geometry"] = pv_buf_3857.geometry.buffer(PV_BUFFER_KM * 1000.0)
    # Use a 4326-projected version of the union for cheap point-in-poly
    pv_buf_4326 = pv_buf_3857.to_crs("EPSG:4326")
    pv_union_4326 = pv_buf_4326.unary_union

    by_continent = land_4326.dissolve(by="continent").reset_index()
    by_continent["area"] = by_continent.geometry.area  # 4326 degrees^2 — fine for relative allocation
    total = by_continent["area"].sum()
    by_continent["alloc"] = (by_continent["area"] / total * n_target).round().astype(int)

    pos_areas = positives_3857["area_m2"].to_numpy()

    samples: list[dict] = []
    next_id = 0
    for _, c in by_continent.iterrows():
        n_here = int(c["alloc"])
        if n_here <= 0: continue
        # Oversample to allow rejections near PV
        pts = _random_points_in_polygon(c.geometry, n=int(n_here * 2.0), rng=rng)
        kept = 0
        for p in pts:
            if kept >= n_here: break
            if pv_union_4326.contains(p): continue
            # Reproject the point to 3857 and buffer by a radius matched
            # to a random positive's area (metres).
            pt_3857 = gpd.GeoSeries([p], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            tgt_area = float(rng.choice(pos_areas))
            radius = math.sqrt(tgt_area / math.pi)
            polyN = pt_3857.buffer(radius)
            samples.append({
                "sample_id":     f"land_{next_id}",
                "polygon_id":    f"land_{next_id}",
                "class":         "not_pv",
                "negative_source": "land",
                "window_name":   "single",
                "window_start":  "2023-01-01",
                "window_end":    "2024-01-01",
                "lat":           None,
                "lon":           None,
                "area_m2":       tgt_area,
                "year_built":    None,
                "geometry":      polyN,
            })
            next_id += 1
            kept += 1
        print(f"  land negatives: {kept:>5}/{n_here} on {c['continent']}", flush=True)
    print(f"  land negatives total: {len(samples):,}", flush=True)
    return pd.DataFrame(samples)


# ---------------------------------------------------------------------------
# Optional negatives — used if external GeoPackages are present
# ---------------------------------------------------------------------------

def _load_optional_negatives(path: Path, source_name: str,
                             positives_3857: gpd.GeoDataFrame,
                             n_target: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        print(f"  {source_name}: SKIPPED (no file at {path})", flush=True)
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    ext = gpd.read_file(path)
    if ext.crs is None or ext.crs.to_epsg() != 3857:
        ext = ext.to_crs("EPSG:3857")
    ext = ext[ext.is_valid & (ext.geometry.area >= MIN_AREA_M2)].reset_index(drop=True)
    pos_areas = positives_3857["area_m2"].to_numpy()
    take = min(n_target, len(ext))
    idx = rng.choice(len(ext), size=take, replace=False)
    rows = []
    for j, i in enumerate(idx):
        geom = ext.iloc[i].geometry
        # If the source polygon is much bigger than the positive distribution,
        # buffer a centroid to a positive-size area to match the spatial scale.
        if geom.area > 5 * np.percentile(pos_areas, 95):
            tgt_area = float(rng.choice(pos_areas))
            radius = math.sqrt(tgt_area / math.pi)
            geom = geom.centroid.buffer(radius)
        rows.append({
            "sample_id":     f"{source_name}_{j}",
            "polygon_id":    f"{source_name}_{j}",
            "class":         "not_pv",
            "negative_source": source_name,
            "window_name":   "single",
            "window_start":  "2023-01-01",
            "window_end":    "2024-01-01",
            "lat":           None,
            "lon":           None,
            "area_m2":       float(geom.area),
            "year_built":    None,
            "geometry":      geom,
        })
    print(f"  {source_name}: {len(rows):,} loaded from {path.name}", flush=True)
    return pd.DataFrame(rows)


def sample_water_negatives(positives_3857, n_target=2_000, seed=44) -> pd.DataFrame:
    return _load_optional_negatives(EXTRA_WATER_GPKG, "water",
                                    positives_3857, n_target, seed)


def sample_industrial_negatives(positives_3857, n_target=3_000, seed=45) -> pd.DataFrame:
    return _load_optional_negatives(EXTRA_INDUSTRIAL_GPKG, "industrial",
                                    positives_3857, n_target, seed)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_polygon_set(out_path: Path = OUT_GPKG,
                      n_positives: int = 10_000,
                      n_land: int = 5_000,
                      n_water: int = 2_000,
                      n_industrial: int = 3_000,
                      seed: int = 42) -> gpd.GeoDataFrame:
    """End-to-end Phase 1: sample, label, write the training-set GeoPackage."""
    print(f"\n=== Phase 1 polygon sampling (seed={seed}) ===", flush=True)
    print(f"Targets: {n_positives:,} positives, {n_land:,} land, "
          f"{n_water:,} water, {n_industrial:,} industrial\n", flush=True)

    positives_3857 = sample_positives(n_positives, seed=seed)
    pos_windows    = expand_positive_windows(positives_3857)
    land           = sample_land_negatives(positives_3857, n_land, seed=seed + 1)
    water          = sample_water_negatives(positives_3857, n_water, seed=seed + 2)
    industrial     = sample_industrial_negatives(positives_3857, n_industrial, seed=seed + 3)

    all_rows = pd.concat([pos_windows, land, water, industrial], ignore_index=True)
    gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs="EPSG:3857")
    # Compute lat/lon for negatives that don't have them set
    miss = gdf["lat"].isna()
    if miss.any():
        c4326 = gdf.loc[miss, "geometry"].to_crs("EPSG:4326").centroid
        gdf.loc[miss, "lon"] = c4326.x.values
        gdf.loc[miss, "lat"] = c4326.y.values
    gdf = gdf.to_crs("EPSG:4326")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    gdf.to_file(out_path, driver="GPKG")
    print(f"\nWrote {len(gdf):,} samples to {out_path}", flush=True)
    print("By class:")
    print(gdf["class"].value_counts().to_string())
    print("By negative_source:")
    print(gdf["negative_source"].value_counts(dropna=False).to_string())
    return gdf


def plot_distribution(gpkg: Path = OUT_GPKG,
                      out_png: Path = REPO / "outputs/fp_classifier/polygons_distribution.png"):
    """Verification step 1 — world map of all samples coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = gpd.read_file(gpkg)
    fig, ax = plt.subplots(figsize=(14, 7))
    land = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    land.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.4)
    colour = {"pv": "tab:green", "becoming_pv": "tab:olive", "not_pv": "tab:red"}
    for cls, c in colour.items():
        sub = gdf[gdf["class"] == cls]
        if len(sub) == 0: continue
        ax.scatter(sub["lon"].values, sub["lat"].values,
                   s=4, c=c, alpha=0.5, edgecolor="none",
                   label=f"{cls} (n={len(sub):,})")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_title("FP-classifier training polygons by class")
    ax.legend(loc="lower left", fontsize=10, markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote distribution plot to {out_png}", flush=True)
