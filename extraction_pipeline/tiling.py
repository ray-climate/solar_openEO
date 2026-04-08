"""Assign polygons to globally consistent 256×256 px chip cells.

Grid definition (EPSG:3857, origin at (0, 0)):
  chip_col = floor(x / CHIP_SIZE_M)      # negative for western hemisphere
  chip_row = floor(y / CHIP_SIZE_M)      # negative for southern hemisphere

Each polygon is assigned to ALL chip cells whose geometry it intersects
(not just the centroid cell).  This guarantees that every panel pixel
appears in at least one chip's mask, regardless of polygon size or position
relative to cell boundaries.

Each chip is later exported directly at the 2.56 km chip AOI — no
intermediate 10 km tile needed.  The same global grid is used for
production inference, so training and inference patches are identical.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pyproj
from shapely.geometry import box

from . import config as cfg

LOGGER = logging.getLogger(__name__)

_TRANSFORMER_TO_4326 = pyproj.Transformer.from_crs(
    "EPSG:3857", "EPSG:4326", always_xy=True
)


# ---------------------------------------------------------------------------
# Low-level grid helpers
# ---------------------------------------------------------------------------

def chip_id_to_bounds(chip_col: int, chip_row: int) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in EPSG:3857 for a chip cell."""
    xmin = chip_col * cfg.CHIP_SIZE_M
    ymin = chip_row * cfg.CHIP_SIZE_M
    return xmin, ymin, xmin + cfg.CHIP_SIZE_M, ymin + cfg.CHIP_SIZE_M


def chip_id_to_center_latlon(chip_col: int, chip_row: int) -> tuple[float, float]:
    """Return (center_lat, center_lon) of a chip in WGS84."""
    x_center = chip_col * cfg.CHIP_SIZE_M + cfg.CHIP_SIZE_M / 2
    y_center = chip_row * cfg.CHIP_SIZE_M + cfg.CHIP_SIZE_M / 2
    lon, lat = _TRANSFORMER_TO_4326.transform(x_center, y_center)
    return float(lat), float(lon)


def make_chip_id_str(chip_col: int, chip_row: int) -> str:
    """Human-readable chip ID.  Format: ``c{col:+07d}_r{row:+07d}``

    Sign prefix ensures correct lexicographic sort across hemispheres.
    Example: ``c+002504_r-000213``
    """
    return f"c{chip_col:+07d}_r{chip_row:+07d}"


def polygon_to_chip_ids(geom) -> list[tuple[int, int]]:
    """Return all chip cells that the polygon geometry intersects.

    Steps:
      1. Enumerate candidate cells from the polygon bounding box.
      2. Filter to cells where the polygon actually intersects the cell box
         (handles concave polygons and edge cases).
    """
    minx, miny, maxx, maxy = geom.bounds

    col_min = math.floor(minx / cfg.CHIP_SIZE_M)
    col_max = math.floor(maxx / cfg.CHIP_SIZE_M)   # may be 1 too many; geometry check filters
    row_min = math.floor(miny / cfg.CHIP_SIZE_M)
    row_max = math.floor(maxy / cfg.CHIP_SIZE_M)

    chips = []
    for col in range(col_min, col_max + 1):
        for row in range(row_min, row_max + 1):
            cell_box = box(*chip_id_to_bounds(col, row))
            if geom.intersects(cell_box):
                chips.append((col, row))

    return chips


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def build_chip_manifest(sample_gdf) -> pd.DataFrame:
    """Assign each polygon to all chip cells it intersects.

    Returns a DataFrame with one row per (polygon, chip) pair.
    Large polygons spanning multiple cells produce multiple rows.

    Columns:
        fid, chip_col, chip_row, chip_id_str,
        chip_xmin, chip_ymin, chip_xmax, chip_ymax,
        chip_center_lat, chip_center_lon,
        continent  (if present in sample_gdf)
    """
    assert sample_gdf.crs and sample_gdf.crs.to_epsg() == 3857, (
        "sample_gdf must be in EPSG:3857"
    )

    rows = []
    multi_chip_count = 0

    for idx, poly_row in sample_gdf.iterrows():
        geom = poly_row.geometry
        fid = poly_row["fid"] if "fid" in sample_gdf.columns else idx
        continent = poly_row.get("continent", "") if "continent" in sample_gdf.columns else ""

        chip_ids = polygon_to_chip_ids(geom)
        if len(chip_ids) > 1:
            multi_chip_count += 1

        for col, row in chip_ids:
            xmin, ymin, xmax, ymax = chip_id_to_bounds(col, row)
            lat, lon = chip_id_to_center_latlon(col, row)
            rows.append({
                "fid": fid,
                "chip_col": col,
                "chip_row": row,
                "chip_id_str": make_chip_id_str(col, row),
                "chip_xmin": xmin,
                "chip_ymin": ymin,
                "chip_xmax": xmax,
                "chip_ymax": ymax,
                "chip_center_lat": lat,
                "chip_center_lon": lon,
                "continent": continent,
            })

    df = pd.DataFrame(rows)
    LOGGER.info(
        "Chip manifest: %d polygons → %d (fid, chip) rows  "
        "(%d polygons span >1 chip)",
        len(sample_gdf), len(df), multi_chip_count,
    )
    LOGGER.info(
        "Unique chips: %d  (mean %.2f chips/polygon)",
        df["chip_id_str"].nunique(),
        len(df) / len(sample_gdf),
    )
    return df


def build_unique_chip_manifest(chip_manifest: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the chip manifest to one row per unique chip cell.

    This is the export job list: one GEE task per unique chip.

    Columns:
        chip_id_str, chip_col, chip_row,
        chip_center_lat, chip_center_lon,
        n_polys
    """
    grouped = (
        chip_manifest
        .groupby(["chip_id_str", "chip_col", "chip_row",
                  "chip_center_lat", "chip_center_lon"])
        .agg(n_polys=("fid", "nunique"))
        .reset_index()
        .sort_values("chip_id_str")
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Unique chip manifest: %d chips  (mean %.2f polys/chip, max %d)",
        len(grouped),
        grouped["n_polys"].mean(),
        grouped["n_polys"].max(),
    )
    return grouped
