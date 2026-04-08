"""Central configuration for Stage-1 extraction pipeline.

All tunable parameters live here. Scripts and modules import from this file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPKG_PATH = PROJECT_ROOT / "data" / "solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage1"

SAMPLE_GPKG = OUTPUT_DIR / "sample_5k.gpkg"
CHIP_MANIFEST_CSV = OUTPUT_DIR / "chip_manifest.csv"          # one row per (fid, chip)
UNIQUE_CHIP_MANIFEST_CSV = OUTPUT_DIR / "unique_chip_manifest.csv"  # one row per unique chip
EXPORT_JOBS_CSV = OUTPUT_DIR / "export_jobs.csv"
MOSAICS_DIR = OUTPUT_DIR / "mosaics"
CHIPS_DIR = OUTPUT_DIR / "chips"
CHIP_METADATA_CSV = OUTPUT_DIR / "chip_metadata.csv"
DATASET_H5 = OUTPUT_DIR / "stage1_positives.h5"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
N_SAMPLE = 5000
FILTER_CONTINENT = "Europe"      # only sample from this continent (None = all)
SAMPLING_BY = "size_category"    # stratification key: "size_category" or "continent"
SIZE_N_BINS = 3                  # number of equal-count size strata (small/medium/large)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Temporal window  (fixed for all tiles — all polygons confirmed Q1 2024)
# ---------------------------------------------------------------------------
START_DATE = "2024-05-01"
END_DATE = "2024-07-31"

# ---------------------------------------------------------------------------
# Spatial grid  — defined in EPSG:3857
#
# Each chip is 256×256 px at 10 m/px = 2560 m per side.
# The grid origin is (0, 0) in EPSG:3857; col/row can be negative.
#
#   chip_col = floor(x / CHIP_SIZE_M)
#   chip_row = floor(y / CHIP_SIZE_M)
#
# Each polygon is assigned to ALL chip cells its geometry overlaps
# (not just the centroid cell).  This guarantees complete polygon coverage.
# Chips are exported directly at chip AOI size — no intermediate 10 km tile.
# The same grid is used for production inference → no domain gap.
# ---------------------------------------------------------------------------
SCALE_M = 10                               # metres per pixel
CHIP_SIZE_PX = 256                         # pixels per chip side
CHIP_SIZE_M = CHIP_SIZE_PX * SCALE_M      # 2560 m per chip side
GRID_CRS = "EPSG:3857"

# ---------------------------------------------------------------------------
# Mosaic module settings  (passed to create_temporal_mosaic)
# Tuned for a 2.56 km chip AOI rather than a 10 km tile.
# ---------------------------------------------------------------------------
MOSAIC_PARAMS: dict = dict(
    out_scale=SCALE_M,
    snic_size_px=20,       # smaller superpixels for 256 px chip (~12×12 clusters)
    patch_size_px=32,      # hierarchical fallback start size
    min_patch_px=8,        # hierarchical fallback minimum
    top_n_scenes=8,
    top_n_scenes_rescue=10,
    clear_thresh=0.8,
    use_shadow_mask=False,
    feather_px=0,          # no feathering — blurs entire chip at snic_size_px=20
)

# ---------------------------------------------------------------------------
# GEE export
# ---------------------------------------------------------------------------
DRIVE_FOLDER = "solar_openEO_stage1_v3"
RCLONE_REMOTE = "google"

# ---------------------------------------------------------------------------
# Spectral bands to extract  (all 13 L1C bands)
# ---------------------------------------------------------------------------
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
N_BANDS = len(BANDS)
