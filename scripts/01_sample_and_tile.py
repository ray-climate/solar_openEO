"""Step 1: Sample polygons and assign them to the global chip grid.

Each polygon is assigned to ALL chip cells it intersects (bbox-overlap +
geometry check), guaranteeing complete polygon coverage.

Inputs:
  data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg

Outputs:
  outputs/stage1/sample_5k.gpkg              — sampled polygons (EPSG:3857)
  outputs/stage1/chip_manifest.csv           — one row per (fid, chip) pair
  outputs/stage1/unique_chip_manifest.csv    — one row per unique chip to export

Usage:
  conda run -n geospatial python scripts/01_sample_and_tile.py
  conda run -n geospatial python scripts/01_sample_and_tile.py --n-sample 100
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.sampling import (
    assign_continent,
    assign_size_category,
    compute_chip_pv_weights,
    filter_continent,
    load_polygons,
    save_sample,
    stratified_sample,
)
from extraction_pipeline.tiling import build_chip_manifest, build_unique_chip_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpkg-path", default=str(cfg.GPKG_PATH))
    p.add_argument("--n-sample", type=int, default=cfg.N_SAMPLE)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load, filter, stratify, sample ---
    gdf = load_polygons(args.gpkg_path)
    gdf = assign_continent(gdf)
    gdf = filter_continent(gdf, cfg.FILTER_CONTINENT)
    gdf = assign_size_category(gdf, n_bins=cfg.SIZE_N_BINS)
    gdf["chip_pv_weight"] = compute_chip_pv_weights(gdf)
    sample = stratified_sample(gdf, n=args.n_sample, by=cfg.SAMPLING_BY, seed=args.seed,
                               weight_col="chip_pv_weight")
    save_sample(sample, cfg.SAMPLE_GPKG)

    # --- Assign to chip grid (bbox-overlap) ---
    chip_manifest = build_chip_manifest(sample)
    unique_chip_manifest = build_unique_chip_manifest(chip_manifest)

    # --- Save ---
    chip_manifest.to_csv(cfg.CHIP_MANIFEST_CSV, index=False)
    unique_chip_manifest.to_csv(cfg.UNIQUE_CHIP_MANIFEST_CSV, index=False)
    LOGGER.info("Chip manifest        → %s", cfg.CHIP_MANIFEST_CSV)
    LOGGER.info("Unique chip manifest → %s", cfg.UNIQUE_CHIP_MANIFEST_CSV)

    # --- Summary ---
    multi = chip_manifest.groupby("fid")["chip_id_str"].count()
    print(f"\nStep 1 complete")
    if cfg.FILTER_CONTINENT:
        print(f"  Continent filter      : {cfg.FILTER_CONTINENT}")
    print(f"  Polygons sampled      : {len(sample)}")
    if "size_category" in sample.columns:
        for cat, cnt in sample["size_category"].value_counts().sort_index().items():
            print(f"    {cat:10s}: {cnt}")
    print(f"  Total (fid, chip) rows: {len(chip_manifest)}")
    print(f"  Unique chips to export: {len(unique_chip_manifest)}")
    print(f"  Polygons spanning >1 chip: {(multi > 1).sum()}")
    print(f"\nNext: run scripts/02_launch_gee_exports.py")


if __name__ == "__main__":
    main()
