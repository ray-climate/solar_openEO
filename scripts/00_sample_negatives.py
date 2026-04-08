"""Step 0-neg: Sample panel-free negative chips from global land areas.

Negative chips are chip cells that have:
  - zero overlap with any known solar panel polygon in the full GPKG
  - centres within one of the six continental bounding boxes

Chips are distributed across continents with fixed proportional quotas.
The full GPKG is used for polygon exclusion — no continent filter.
Chips already in the positives unique_chip_manifest are excluded.

Outputs:
  outputs/stage1/negatives/unique_chip_manifest.csv
  — same schema as unique_chip_manifest.csv (positives)
  — n_polys column is always 0

Usage:
  conda run -n geospatial python scripts/00_sample_negatives.py
  conda run -n geospatial python scripts/00_sample_negatives.py --n-total 2000 --seed 123
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.sampling import load_polygons
from extraction_pipeline.tiling import chip_id_to_bounds, chip_id_to_center_latlon, make_chip_id_str

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Continent bounding boxes (lon_min, lat_min, lon_max, lat_max) — WGS84
# Mirrors extraction_pipeline.sampling._CONTINENT_BOXES
# ---------------------------------------------------------------------------
CONTINENT_BOXES: dict[str, tuple[float, float, float, float]] = {
    "Europe":    (-9.0,   36.0,   60.0,  71.0),
    "Asia":      (26.0,  -10.0,  180.0,  77.0),
    "N_America": (-168.0,  7.0,  -52.0,  84.0),
    "Africa":    (-17.0, -35.0,   51.0,  37.0),
    "S_America": (-82.0, -56.0,  -34.0,  13.0),
    "Oceania":   (112.0, -47.0,  180.0, -10.0),
}

# Default quota per continent — must sum to 2000.
# Scale proportionally when --n-total differs.
_DEFAULT_QUOTAS: dict[str, int] = {
    "Europe":    400,
    "Asia":      500,
    "N_America": 350,
    "Africa":    300,
    "S_America": 250,
    "Oceania":   200,
}

_TR_4326_TO_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lonlat_to_chip_colrow(lon: float, lat: float) -> tuple[int, int]:
    x, y = _TR_4326_TO_3857.transform(lon, lat)
    return math.floor(x / cfg.CHIP_SIZE_M), math.floor(y / cfg.CHIP_SIZE_M)


def _bbox_to_chip_range(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> tuple[int, int, int, int]:
    """Return (col_min, col_max, row_min, row_max) chip indices for a WGS84 bbox."""
    col_min, row_min = _lonlat_to_chip_colrow(lon_min, lat_min)
    col_max, row_max = _lonlat_to_chip_colrow(lon_max, lat_max)
    return col_min, col_max, row_min, row_max


# ---------------------------------------------------------------------------
# Quota helpers
# ---------------------------------------------------------------------------

def _scale_quotas(n_total: int) -> dict[str, int]:
    """Scale default quotas proportionally to sum to n_total."""
    default_total = sum(_DEFAULT_QUOTAS.values())
    scaled = {k: max(1, round(v * n_total / default_total)) for k, v in _DEFAULT_QUOTAS.items()}
    # Correct rounding residual on the largest-quota continent
    diff = n_total - sum(scaled.values())
    if diff != 0:
        largest = max(scaled, key=scaled.__getitem__)
        scaled[largest] += diff
    return scaled


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_negatives_for_continent(
    continent: str,
    quota: int,
    full_gdf,
    excluded_ids: set[str],
    rng: np.random.Generator,
    oversample: int = 30,
) -> list[tuple[int, int, str]]:
    """Return up to `quota` (col, row, chip_id_str) tuples with no polygon overlap.

    Candidates are drawn uniformly at random within the continent's bounding box.
    A chip is accepted if its bounding box does not intersect any polygon in full_gdf
    (spatial index bbox check — slightly conservative but fast).

    Parameters
    ----------
    excluded_ids:
        Chip IDs already claimed by the positives manifest or earlier continents.
        Updated in-place as new chips are accepted.
    oversample:
        Candidate multiplier per batch.  With typical global solar density,
        oversample=30 is more than sufficient to find quotas quickly.
    """
    lon_min, lat_min, lon_max, lat_max = CONTINENT_BOXES[continent]
    col_min, col_max, row_min, row_max = _bbox_to_chip_range(
        lon_min, lat_min, lon_max, lat_max
    )

    selected: list[tuple[int, int, str]] = []
    batch_size = max(quota * oversample, 2000)
    max_batches = 10

    for batch_idx in range(max_batches):
        if len(selected) >= quota:
            break

        cols = rng.integers(col_min, col_max + 1, size=batch_size)
        rows = rng.integers(row_min, row_max + 1, size=batch_size)

        for col, row in zip(cols.tolist(), rows.tolist()):
            chip_id = make_chip_id_str(col, row)
            if chip_id in excluded_ids:
                continue
            bounds = chip_id_to_bounds(col, row)
            if not list(full_gdf.sindex.intersection(bounds)):
                selected.append((col, row, chip_id))
                excluded_ids.add(chip_id)
                if len(selected) >= quota:
                    break

        LOGGER.debug(
            "  %s batch %d: %d / %d found so far",
            continent, batch_idx + 1, len(selected), quota,
        )

    if len(selected) < quota:
        LOGGER.warning(
            "%s: found only %d / %d negatives after %d batches of %d candidates",
            continent, len(selected), quota, max_batches, batch_size,
        )

    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-total", type=int, default=2000,
        help="Total negative chips to sample across all continents (default 2000).",
    )
    p.add_argument(
        "--out-csv", default=None,
        help="Output manifest CSV (default: outputs/stage1/negatives/unique_chip_manifest.csv).",
    )
    p.add_argument(
        "--positives-manifest", default=str(cfg.UNIQUE_CHIP_MANIFEST_CSV),
        help="Positives manifest to exclude (default: unique_chip_manifest.csv).",
    )
    p.add_argument(
        "--seed", type=int, default=cfg.RANDOM_SEED,
        help="Random seed (default: from config).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else cfg.OUTPUT_DIR / "negatives" / "unique_chip_manifest.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Exclude chips already in the positives manifest ---
    excluded_ids: set[str] = set()
    pos_manifest = Path(args.positives_manifest)
    if pos_manifest.exists():
        df_pos = pd.read_csv(pos_manifest)
        excluded_ids.update(df_pos["chip_id_str"].tolist())
        LOGGER.info("Loaded %d positive chip IDs to exclude", len(excluded_ids))
    else:
        LOGGER.warning(
            "Positives manifest not found at %s — no positive exclusion applied",
            pos_manifest,
        )

    # --- Load full GPKG (no continent filter) for polygon exclusion ---
    LOGGER.info("Loading full polygon GPKG (no continent filter): %s", cfg.GPKG_PATH)
    full_gdf = load_polygons(str(cfg.GPKG_PATH))
    LOGGER.info("Loaded %d polygons; building spatial index ...", len(full_gdf))
    _ = full_gdf.sindex  # trigger index build
    LOGGER.info("Spatial index ready.")

    # --- Scale quotas ---
    quotas = _scale_quotas(args.n_total)
    LOGGER.info(
        "Target quotas: %s  (total=%d)",
        {k: v for k, v in quotas.items()},
        sum(quotas.values()),
    )

    rng = np.random.default_rng(args.seed)
    # excluded_ids is shared and updated in-place across continents
    all_selected: list[dict] = []

    for continent, quota in quotas.items():
        LOGGER.info("Sampling %d negatives for %s ...", quota, continent)
        chips = _sample_negatives_for_continent(
            continent, quota, full_gdf, excluded_ids, rng
        )
        LOGGER.info("  → accepted %d chips for %s", len(chips), continent)

        for col, row, chip_id in chips:
            lat, lon = chip_id_to_center_latlon(col, row)
            all_selected.append({
                "chip_id_str": chip_id,
                "chip_col": col,
                "chip_row": row,
                "chip_center_lat": lat,
                "chip_center_lon": lon,
                "continent": continent,
                "n_polys": 0,
            })

    # --- Write manifest ---
    out_df = pd.DataFrame(all_selected)
    out_df.to_csv(out_csv, index=False)
    LOGGER.info("Negative chip manifest → %s  (%d chips)", out_csv, len(out_df))

    # --- Summary ---
    print("\nStep 0-neg complete")
    print(f"  Total negatives sampled : {len(out_df)}")
    for cont, cnt in out_df["continent"].value_counts().sort_index().items():
        print(f"    {cont:15s}: {cnt}")
    print(f"\nNext: run scripts/02_launch_gee_exports.py \\")
    print(f"        --manifest {out_csv} \\")
    print(f"        --drive-folder solar_openEO_negatives_v1 \\")
    print(f"        --jobs-csv outputs/stage1/negatives/export_jobs.csv")


if __name__ == "__main__":
    main()
