"""Sample polygons from the full solar panel GeoPackage.

Stage-1 strategy:
  1. Filter to a single continent (default: Europe).
  2. Compute polygon area from geometry (the stored area columns are constants).
  3. Assign size categories based on area quantiles (small / medium / large).
  4. Sample equal-quota from each size category so the training set covers
     the full range of PV installation sizes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box as shapely_box

from . import config as cfg

LOGGER = logging.getLogger(__name__)

# Simple bounding-box continent lookup (lon_min, lat_min, lon_max, lat_max).
# Applied in priority order — first match wins.
_CONTINENT_BOXES: list[tuple[str, float, float, float, float]] = [
    ("Oceania",     112.0,  -47.0,  180.0,  -10.0),
    ("S_America",   -82.0,  -56.0,  -34.0,   13.0),
    ("N_America",  -168.0,    7.0,  -52.0,   84.0),
    ("Africa",      -17.0,  -35.0,   51.0,   37.0),
    ("Europe",       -9.0,   36.0,   60.0,   71.0),
    ("Asia",         26.0,  -10.0,  180.0,   77.0),
]

_SIZE_LABELS = {1: "small", 2: "medium", 3: "large"}   # for 3-bin case


def _assign_continent_series(lon: pd.Series, lat: pd.Series) -> pd.Series:
    result = pd.Series("Other", index=lon.index, dtype=str)
    for name, lon_min, lat_min, lon_max, lat_max in _CONTINENT_BOXES:
        mask = (
            (lon >= lon_min) & (lon <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max) &
            (result == "Other")
        )
        result[mask] = name
    return result


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_polygons(gpkg_path: str | Path) -> gpd.GeoDataFrame:
    """Load the full polygon dataset, reproject to EPSG:3857.

    Adds:
      ``lon_wgs84`` / ``lat_wgs84`` — WGS84 centroid coordinates
      ``area_m2``                   — computed geometry area in m²
                                      (stored area columns are constants)
    """
    LOGGER.info("Loading polygons from %s", gpkg_path)
    gdf = gpd.read_file(gpkg_path)
    LOGGER.info("Loaded %d polygons  CRS=%s", len(gdf), gdf.crs)

    gdf = gdf.to_crs("EPSG:3857")

    # WGS84 centroids for continent assignment (computed in metric CRS)
    centroids_m = gdf.geometry.centroid
    tr = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lons, lats = tr.transform(centroids_m.x.values, centroids_m.y.values)
    gdf["lon_wgs84"] = lons
    gdf["lat_wgs84"] = lats

    # Compute actual polygon area from geometry
    gdf["area_m2"] = gdf.geometry.area

    return gdf


def assign_continent(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a ``continent`` column using bounding-box rules."""
    gdf = gdf.copy()
    gdf["continent"] = _assign_continent_series(gdf["lon_wgs84"], gdf["lat_wgs84"])
    LOGGER.info("Continent distribution:\n%s",
                gdf["continent"].value_counts().to_string())
    return gdf


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_continent(
    gdf: gpd.GeoDataFrame,
    continent: str | None,
) -> gpd.GeoDataFrame:
    """Keep only polygons from the specified continent.

    If ``continent`` is None, all polygons are kept.
    """
    if continent is None:
        return gdf
    filtered = gdf[gdf["continent"] == continent].copy()
    LOGGER.info(
        "Filtered to %s: %d / %d polygons", continent, len(filtered), len(gdf)
    )
    return filtered


# ---------------------------------------------------------------------------
# Size stratification
# ---------------------------------------------------------------------------

def assign_size_category(
    gdf: gpd.GeoDataFrame,
    n_bins: int = 3,
    area_col: str = "area_m2",
) -> gpd.GeoDataFrame:
    """Add a ``size_category`` column based on polygon area quantiles.

    Bins are computed within the provided GeoDataFrame (not globally), so
    they reflect the size distribution of the filtered subset.

    With n_bins=3:  small | medium | large  (equal polygon count per bin)
    """
    gdf = gdf.copy()
    labels = [_SIZE_LABELS.get(i + 1, str(i + 1)) for i in range(n_bins)]
    quantiles = np.linspace(0, 1, n_bins + 1)
    boundaries = np.quantile(gdf[area_col], quantiles)

    # pd.cut with computed boundaries (more robust than pd.qcut for duplicates)
    boundaries[0] -= 1e-6        # make left edge inclusive
    gdf["size_category"] = pd.cut(
        gdf[area_col],
        bins=boundaries,
        labels=labels,
    ).astype(str)

    bin_stats = gdf.groupby("size_category")[area_col].agg(["min", "max", "count"])
    LOGGER.info(
        "Size categories (area_m2, n_bins=%d):\n%s", n_bins, bin_stats.to_string()
    )
    return gdf


# ---------------------------------------------------------------------------
# Chip-level PV density weighting
# ---------------------------------------------------------------------------

def compute_chip_pv_weights(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Compute per-polygon sampling weights based on chip-level PV pixel ratio.

    For each polygon the weight equals the maximum chip PV ratio across all chip
    cells the polygon touches.  Chip PV ratio = (sum of all polygon intersection
    areas in that chip) / chip_area.

    Using ALL polygons in gdf (not just a sample) to compute the chip sums means
    the weights reflect true PV density in the landscape.

    Returns a pd.Series of floats indexed by gdf.index.
    """
    # Import here to avoid circular import (tiling imports config, not sampling)
    from .tiling import chip_id_to_bounds, make_chip_id_str, polygon_to_chip_ids

    chip_area = float(cfg.CHIP_SIZE_M ** 2)  # 2560 m × 2560 m = 6,553,600 m²

    LOGGER.info(
        "Computing chip-level PV density for %d polygons (this may take ~1 min)...",
        len(gdf),
    )

    # Pass 1: accumulate intersection area per chip across all polygons.
    chip_poly_area: dict[str, float] = {}
    poly_chips: dict[int, list[str]] = {}   # gdf index → chip IDs it touches

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not geom.is_valid:
            geom = geom.buffer(0)   # repair self-intersecting rings
        chips_for_poly: list[str] = []
        for col, r in polygon_to_chip_ids(geom):
            xmin, ymin, xmax, ymax = chip_id_to_bounds(col, r)
            try:
                inter_area = geom.intersection(shapely_box(xmin, ymin, xmax, ymax)).area
            except Exception:
                inter_area = 0.0
            if inter_area > 0:
                cid = make_chip_id_str(col, r)
                chip_poly_area[cid] = chip_poly_area.get(cid, 0.0) + inter_area
                chips_for_poly.append(cid)
        poly_chips[idx] = chips_for_poly

    # Pass 2: chip PV ratio (capped at 1.0 in case of buffered/overlapping polys).
    chip_pv_ratio = {
        cid: min(area / chip_area, 1.0)
        for cid, area in chip_poly_area.items()
    }

    LOGGER.info(
        "Chip PV ratio stats across %d chips: "
        "mean=%.4f  median=%.4f  p90=%.4f  max=%.4f",
        len(chip_pv_ratio),
        np.mean(list(chip_pv_ratio.values())),
        np.median(list(chip_pv_ratio.values())),
        np.percentile(list(chip_pv_ratio.values()), 90),
        max(chip_pv_ratio.values()),
    )

    # Pass 3: for each polygon, assign the max chip PV ratio across its chips.
    weights = {
        idx: max((chip_pv_ratio.get(c, 0.0) for c in poly_chips.get(idx, [])),
                 default=0.0)
        for idx in gdf.index
    }
    return pd.Series(weights, index=gdf.index)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    gdf: gpd.GeoDataFrame,
    n: int = 5000,
    by: str = "size_category",
    seed: int = 42,
    weight_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Sample ``n`` polygons with equal quota per stratum.

    Each stratum gets floor(n / n_strata) samples.  Any remainder is added
    to the largest stratum.  Strata smaller than their quota are fully included.

    If ``weight_col`` is given, sampling within each stratum is weighted by that
    column (probability proportional to value), so higher-weight polygons are
    more likely to be selected.
    """
    strata = gdf[by].value_counts()
    n_strata = len(strata)
    base_quota = n // n_strata
    remainder = n - base_quota * n_strata

    quotas = {s: base_quota for s in strata.index}
    # Add remainder to the largest stratum
    quotas[strata.idxmax()] += remainder
    # Clamp to actual stratum size
    quotas = {s: min(q, strata[s]) for s, q in quotas.items()}

    frames: list[gpd.GeoDataFrame] = []
    for stratum, count in quotas.items():
        subset = gdf[gdf[by] == stratum]
        weights = subset[weight_col] if weight_col is not None else None
        frames.append(subset.sample(n=int(count), random_state=seed, weights=weights))

    sample = gpd.GeoDataFrame(
        pd.concat(frames), crs=gdf.crs
    ).reset_index(drop=True)

    LOGGER.info(
        "Sampled %d polygons across %d strata (by='%s'):\n%s",
        len(sample), n_strata, by,
        sample[by].value_counts().to_string(),
    )
    if weight_col is not None:
        LOGGER.info(
            "Sampling weight ('%s') stats in final sample: "
            "mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
            weight_col,
            sample[weight_col].mean(),
            sample[weight_col].median(),
            sample[weight_col].min(),
            sample[weight_col].max(),
        )
    return sample


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_sample(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    """Save sampled GeoDataFrame to a GeoPackage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop any columns that cause GPKG write issues (e.g. Categorical dtype)
    out = gdf.copy()
    for col in out.select_dtypes(include="category").columns:
        out[col] = out[col].astype(str)
    out.to_file(path, driver="GPKG")
    LOGGER.info("Saved %d sampled polygons to %s", len(gdf), path)
