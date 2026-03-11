"""Extract training chips and binary masks from downloaded chip mosaics.

Since each GEE export covers exactly one 256×256 chip cell, the downloaded
GeoTIFF IS the chip — no window carving is needed.  This module simply reads
the full file, rasterises the polygon mask, and writes the outputs.

Runs entirely locally — no GEE required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from . import config as cfg
from .dataset_writer import write_chip_geotiff
from .tiling import chip_id_to_bounds

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mosaic file lookup
# ---------------------------------------------------------------------------

def find_chip_mosaic_path(chip_id_str: str, mosaics_dir: str | Path) -> Path | None:
    """Locate the downloaded GeoTIFF for a chip.

    GEE exports the file as ``stage1_<safe_chip_id>.tif`` where ``+``→``p``
    and ``-``→``m``.  We search flexibly for any ``.tif`` whose stem starts
    with the sanitised export name.
    """
    mosaics_dir = Path(mosaics_dir)
    safe = chip_id_str.replace("+", "p").replace("-", "m")
    export_stem = f"stage1_{safe}"

    # Exact match
    candidate = mosaics_dir / f"{export_stem}.tif"
    if candidate.exists():
        return candidate

    # GEE sometimes appends sequence numbers
    matches = sorted(mosaics_dir.glob(f"{export_stem}*.tif"))
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Read chip array (full file — no window carving needed)
# ---------------------------------------------------------------------------

def read_chip_array(mosaic_path: Path) -> np.ndarray:
    """Read the full chip GeoTIFF into a (13, 256, 256) float32 array.

    The spectral bands are the first N_BANDS bands in the file.
    QA bands (source_scene_id, etc.) are ignored here.
    """
    with rasterio.open(mosaic_path) as src:
        n_bands = min(cfg.N_BANDS, src.count)
        data = src.read(
            indexes=list(range(1, n_bands + 1)),
            out_shape=(n_bands, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX),
            resampling=rasterio.enums.Resampling.bilinear,
        )
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Mask rasterisation
# ---------------------------------------------------------------------------

def rasterize_polygon_mask(polygon_geoms, chip_col: int, chip_row: int) -> np.ndarray:
    """Burn polygon geometries into a binary chip mask.

    Parameters
    ----------
    polygon_geoms:
        Iterable of Shapely geometries in EPSG:3857.
    chip_col, chip_row:
        Grid indices identifying the chip cell.

    Returns
    -------
    np.ndarray of shape (256, 256), dtype uint8  [1=panel, 0=background].
    """
    xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
    transform = from_bounds(xmin, ymin, xmax, ymax, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX)

    geom_list = [
        (geom, 1) for geom in polygon_geoms
        if geom is not None and not geom.is_empty
    ]
    if not geom_list:
        return np.zeros((cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX), dtype=np.uint8)

    mask = rasterize(
        shapes=geom_list,
        out_shape=(cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_all_chips(
    chip_manifest: pd.DataFrame,
    mask_gdf: gpd.GeoDataFrame,
    mosaics_dir: str | Path,
    chips_dir: str | Path,
    overwrite: bool = False,
    chip_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Extract chips for all rows in the unique chip list.

    For each unique chip:
      1. Locate the downloaded mosaic GeoTIFF (skip if not yet available).
      2. Read the full (13, 256, 256) spectral array.
      3. Rasterise the binary mask from ALL polygons in mask_gdf that
         spatially overlap the chip — not just the sampled subset.
      4. Write <chip_id>_image.tif and <chip_id>_mask.tif.
      5. Record stats (n_panel_px, panel_frac, coverage_ok).

    Resumable: skips already-extracted chips unless ``overwrite=True``.

    Parameters
    ----------
    chip_manifest:
        One row per (fid, chip) pair — drives which chips to process.
    mask_gdf:
        Full polygon dataset (EPSG:3857) used for mask rasterisation.
        Should contain ALL known solar polygons, not just the sample.

    Returns
    -------
    DataFrame of per-chip metadata with extraction stats.
    """
    from shapely.geometry import box as shapely_box

    mosaics_dir = Path(mosaics_dir)
    chips_dir = Path(chips_dir)
    chips_dir.mkdir(parents=True, exist_ok=True)

    assert mask_gdf.crs and mask_gdf.crs.to_epsg() == 3857, \
        "mask_gdf must be in EPSG:3857"

    # Filter to specific chips if requested
    unique_chips = chip_manifest.drop_duplicates("chip_id_str").copy()
    if chip_ids is not None:
        unique_chips = unique_chips[unique_chips["chip_id_str"].isin(chip_ids)]

    records: list[dict] = []

    for _, chip_row in unique_chips.iterrows():
        chip_id_str = chip_row["chip_id_str"]
        chip_col = int(chip_row["chip_col"])
        chip_row_idx = int(chip_row["chip_row"])
        continent = chip_row.get("continent", "")

        image_path = chips_dir / f"{chip_id_str}_image.tif"
        mask_path = chips_dir / f"{chip_id_str}_mask.tif"

        # Resume: skip already-extracted chips
        if image_path.exists() and mask_path.exists() and not overwrite:
            LOGGER.debug("Skipping already-extracted chip %s", chip_id_str)
            records.append(_existing_record(chip_id_str, chip_col, chip_row_idx,
                                            continent, mask_path))
            continue

        # Locate mosaic
        mosaic_path = find_chip_mosaic_path(chip_id_str, mosaics_dir)
        if mosaic_path is None:
            LOGGER.debug("Mosaic not yet downloaded for chip %s — skipping", chip_id_str)
            records.append(_missing_record(chip_id_str, chip_col, chip_row_idx, continent))
            continue

        # Read spectral array
        image_arr = read_chip_array(mosaic_path)

        # Rasterise mask from ALL polygons overlapping this chip
        bounds = chip_id_to_bounds(chip_col, chip_row_idx)
        chip_box = shapely_box(*bounds)
        candidate_idx = list(mask_gdf.sindex.intersection(bounds))
        geoms = mask_gdf.iloc[candidate_idx].geometry[
            mask_gdf.iloc[candidate_idx].geometry.intersects(chip_box)
        ].tolist()
        mask_arr = rasterize_polygon_mask(geoms, chip_col, chip_row_idx)

        # Write chip + mask
        write_chip_geotiff(image_arr, mask_arr, chip_id_str, bounds, chips_dir)

        n_panel_px = int(mask_arr.sum())
        panel_frac = n_panel_px / (cfg.CHIP_SIZE_PX ** 2)
        LOGGER.info(
            "Extracted %s  n_polys=%d  panel_px=%d (%.2f%%)",
            chip_id_str, len(geoms), n_panel_px, panel_frac * 100,
        )
        records.append({
            "chip_id_str": chip_id_str,
            "chip_col": chip_col,
            "chip_row": chip_row_idx,
            "n_polys_in_chip": len(geoms),
            "n_panel_px": n_panel_px,
            "panel_frac": round(panel_frac, 6),
            "coverage_ok": True,
            "continent": continent,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper record builders
# ---------------------------------------------------------------------------

def _existing_record(chip_id_str, chip_col, chip_row_idx, continent, mask_path):
    n_panel_px, panel_frac = 0, 0.0
    if Path(mask_path).exists():
        with rasterio.open(mask_path) as src:
            arr = src.read(1)
            n_panel_px = int(arr.sum())
            panel_frac = round(n_panel_px / (cfg.CHIP_SIZE_PX ** 2), 6)
    return {
        "chip_id_str": chip_id_str,
        "chip_col": chip_col, "chip_row": chip_row_idx,
        "n_panel_px": n_panel_px, "panel_frac": panel_frac,
        "coverage_ok": True, "continent": continent,
    }


def _missing_record(chip_id_str, chip_col, chip_row_idx, continent):
    return {
        "chip_id_str": chip_id_str,
        "chip_col": chip_col, "chip_row": chip_row_idx,
        "n_panel_px": 0, "panel_frac": 0.0,
        "coverage_ok": False, "continent": continent,
    }
