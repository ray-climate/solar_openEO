"""Step 4: Extract 256×256 training chips from downloaded chip mosaics.

Since each GEE export IS the chip, this step simply reads the GeoTIFF,
rasterises the polygon mask, and writes the chip/mask pair.

Mosaics must be in outputs/stage1/mosaics/ before running.
(Download them from Google Drive folder 'solar_openEO_stage1_mosaics'.)

Outputs:
  outputs/stage1/chips/<chip_id>_image.tif   (13, 256, 256) float32
  outputs/stage1/chips/<chip_id>_mask.tif    (256, 256) uint8
  outputs/stage1/chip_metadata.csv

Usage:
  conda run -n geospatial python scripts/04_extract_chips.py

  # Process specific chips only
  conda run -n geospatial python scripts/04_extract_chips.py \\
      --chip-ids c+0002504_r+0001234

  # Re-extract even if output already exists
  conda run -n geospatial python scripts/04_extract_chips.py --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.chip_extractor import extract_all_chips
from extraction_pipeline.sampling import assign_continent, filter_continent, load_polygons

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip-ids", nargs="+", default=None,
                   help="Process only these chip_id_str values.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract chips even if output files already exist.")
    p.add_argument("--mosaics-dir", default=str(cfg.MOSAICS_DIR),
                   help="Directory containing downloaded chip GeoTIFFs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for required in [cfg.CHIP_MANIFEST_CSV, cfg.SAMPLE_GPKG]:
        if not required.exists():
            LOGGER.error("Required file not found: %s", required)
            LOGGER.error("Run scripts/01_sample_and_tile.py first.")
            sys.exit(1)

    chip_manifest = pd.read_csv(cfg.CHIP_MANIFEST_CSV)

    # Load FULL polygon dataset for masking (not just the 5K sample).
    # This ensures all known solar polygons are burned into the mask,
    # even those not selected during sampling.
    LOGGER.info("Loading full polygon dataset for mask generation: %s", cfg.GPKG_PATH)
    full_gdf = load_polygons(str(cfg.GPKG_PATH))
    full_gdf = assign_continent(full_gdf)
    if cfg.FILTER_CONTINENT:
        full_gdf = filter_continent(full_gdf, cfg.FILTER_CONTINENT)
    mask_gdf = full_gdf.to_crs("EPSG:3857")
    LOGGER.info(
        "Loaded chip manifest (%d rows) and %d mask polygons (%s only)",
        len(chip_manifest), len(mask_gdf), cfg.FILTER_CONTINENT or "all",
    )

    n_mosaics = len(list(Path(args.mosaics_dir).glob("*.tif")))
    LOGGER.info("Mosaics available in %s: %d", args.mosaics_dir, n_mosaics)
    if n_mosaics == 0:
        LOGGER.warning(
            "No .tif files in %s — download mosaics from Drive first.", args.mosaics_dir
        )

    chip_metadata = extract_all_chips(
        chip_manifest=chip_manifest,
        mask_gdf=mask_gdf,
        mosaics_dir=args.mosaics_dir,
        chips_dir=cfg.CHIPS_DIR,
        overwrite=args.overwrite,
        chip_ids=args.chip_ids,
    )

    chip_metadata.to_csv(cfg.CHIP_METADATA_CSV, index=False)
    LOGGER.info("Chip metadata → %s", cfg.CHIP_METADATA_CSV)

    ok = chip_metadata["coverage_ok"].sum()
    total = len(chip_metadata)
    print(f"\nStep 4 complete")
    print(f"  Unique chips processed    : {total}")
    print(f"  Successfully extracted    : {ok}")
    print(f"  Skipped (no mosaic yet)   : {total - ok}")
    if ok > 0:
        print(f"\nNext: run scripts/06_visualize_chips.py  (QA figures)")
        print(f"      then scripts/05_package_dataset.py  (HDF5)")


if __name__ == "__main__":
    main()
