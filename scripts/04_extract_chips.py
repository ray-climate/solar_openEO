"""Step 4: Extract 256×256 training chips from downloaded chip mosaics.

Since each GEE export IS the chip, this step simply reads the GeoTIFF,
rasterises the polygon mask, and writes the chip/mask pair.

Mosaics must be in outputs/stage1/mosaics/ before running.
(Download them from Google Drive folder 'solar_openEO_stage1_v3'.)

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

from __future__ import annotations

import argparse
import logging
import re
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
_SAFE_CHIP_RE = re.compile(r"^stage1_(c[pm]\d+_r[pm]\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip-ids", nargs="+", default=None,
                   help="Process only these chip_id_str values.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract chips even if output files already exist.")
    p.add_argument("--mosaics-dir", default=str(cfg.MOSAICS_DIR),
                   help="Directory containing downloaded chip GeoTIFFs.")
    p.add_argument("--chips-dir", default=str(cfg.CHIPS_DIR),
                   help="Output directory for extracted chip files "
                        "(default: outputs/stage1/chips).")
    p.add_argument("--chip-metadata-csv", default=str(cfg.CHIP_METADATA_CSV),
                   help="Output path for per-chip metadata CSV "
                        "(default: outputs/stage1/chip_metadata.csv).")
    p.add_argument("--no-continent-filter", action="store_true",
                   help="Load the full polygon GPKG without continent filtering. "
                        "Required for global negative chips where panels outside "
                        "Europe must still be masked correctly.")
    return p.parse_args()


def safe_export_stem_to_chip_id(stem: str) -> str | None:
    match = _SAFE_CHIP_RE.match(stem)
    if not match:
        return None
    safe_chip = match.group(1)
    return safe_chip.replace("p", "+").replace("m", "-")


def chip_id_to_cols_rows(chip_id_str: str) -> tuple[int, int]:
    col_str, row_str = chip_id_str.split("_")
    return int(col_str[1:]), int(row_str[1:])


def build_chip_manifest_from_mosaics(mosaics_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for mosaic_path in sorted(Path(mosaics_dir).glob("*.tif")):
        chip_id_str = safe_export_stem_to_chip_id(mosaic_path.stem)
        if chip_id_str is None:
            continue
        chip_col, chip_row = chip_id_to_cols_rows(chip_id_str)
        rows.append({
            "fid": -1,
            "chip_col": chip_col,
            "chip_row": chip_row,
            "chip_id_str": chip_id_str,
            "continent": "",
        })
    return pd.DataFrame(rows).drop_duplicates("chip_id_str").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if not cfg.GPKG_PATH.exists():
        LOGGER.error("Required file not found: %s", cfg.GPKG_PATH)
        LOGGER.error("The full polygon label source is required to rasterize masks.")
        sys.exit(1)

    if cfg.CHIP_MANIFEST_CSV.exists():
        chip_manifest = pd.read_csv(cfg.CHIP_MANIFEST_CSV)
        LOGGER.info("Loaded chip manifest: %s", cfg.CHIP_MANIFEST_CSV)
    else:
        LOGGER.warning("Chip manifest not found: %s", cfg.CHIP_MANIFEST_CSV)
        LOGGER.warning("Recovering unique chip list from %s instead.", args.mosaics_dir)
        chip_manifest = build_chip_manifest_from_mosaics(args.mosaics_dir)
        if chip_manifest.empty:
            LOGGER.error("No mosaic GeoTIFFs found in %s", args.mosaics_dir)
            sys.exit(1)
        LOGGER.info("Recovered %d unique chips from mosaic filenames", len(chip_manifest))

    # Load FULL polygon dataset for masking (not just the 5K sample).
    # This ensures all known solar polygons are burned into the mask,
    # even those not selected during sampling.
    LOGGER.info("Loading full polygon dataset for mask generation: %s", cfg.GPKG_PATH)
    full_gdf = load_polygons(str(cfg.GPKG_PATH))
    full_gdf = assign_continent(full_gdf)
    if cfg.FILTER_CONTINENT and not args.no_continent_filter:
        full_gdf = filter_continent(full_gdf, cfg.FILTER_CONTINENT)
        continent_label = cfg.FILTER_CONTINENT
    else:
        continent_label = "all"
    mask_gdf = full_gdf.to_crs("EPSG:3857")
    LOGGER.info(
        "Loaded chip manifest (%d rows) and %d mask polygons (%s)",
        len(chip_manifest), len(mask_gdf), continent_label,
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
        chips_dir=Path(args.chips_dir),
        overwrite=args.overwrite,
        chip_ids=args.chip_ids,
    )

    chip_metadata_csv_out = Path(args.chip_metadata_csv)
    chip_metadata_csv_out.parent.mkdir(parents=True, exist_ok=True)
    chip_metadata.to_csv(chip_metadata_csv_out, index=False)
    LOGGER.info("Chip metadata → %s", chip_metadata_csv_out)

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
