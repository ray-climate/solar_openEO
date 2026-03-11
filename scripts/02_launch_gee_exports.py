"""Step 2: Submit GEE export jobs for chips in the unique chip manifest.

Each chip gets one create_temporal_mosaic() call.  The exported 256×256
GeoTIFF is the chip directly — no carving step needed.

Export status is saved to outputs/stage1/export_jobs.csv.
Run is resumable: chips already in export_jobs.csv are skipped.

Usage:
  # Dry run — print what would be submitted
  conda run -n geospatial python scripts/02_launch_gee_exports.py --dry-run

  # Submit 1 chip for testing
  conda run -n geospatial python scripts/02_launch_gee_exports.py --batch-size 1

  # Submit a diverse test set of 30 chips (geographically spread)
  conda run -n geospatial python scripts/02_launch_gee_exports.py --test-n 30

  # Submit next 50 chips
  conda run -n geospatial python scripts/02_launch_gee_exports.py --batch-size 50

  # Submit specific chip IDs
  conda run -n geospatial python scripts/02_launch_gee_exports.py \\
      --chip-ids c+0002504_r+0001234 c+0002505_r+0001234
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.export_manager import print_status_summary, submit_all_exports

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be submitted without calling GEE.")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Max chips to submit per run (default 50).")
    p.add_argument("--test-n", type=int, default=None,
                   help="Submit N geographically diverse chips for testing.")
    p.add_argument("--chip-ids", nargs="+", default=None,
                   help="Submit only these chip_id_str values.")
    p.add_argument("--retry-errors", action="store_true",
                   help="Re-submit chips that previously failed with SUBMIT_ERROR.")
    return p.parse_args()


def diverse_sample(manifest: pd.DataFrame, n: int) -> list[str]:
    """Pick N chips spread across continents for a diverse test set."""
    # Merge continent info from chip_manifest
    chip_manifest = pd.read_csv(cfg.CHIP_MANIFEST_CSV)
    continent_map = (
        chip_manifest.groupby("chip_id_str")["continent"]
        .first()
        .reset_index()
    )
    merged = manifest.merge(continent_map, on="chip_id_str", how="left")
    merged["continent"] = merged["continent"].fillna("Other")

    continents = merged["continent"].value_counts()
    quota = max(1, n // len(continents))

    selected = []
    for cont in continents.index:
        subset = merged[merged["continent"] == cont]
        take = min(quota, len(subset))
        selected.append(subset.sample(take, random_state=42))

    result = pd.concat(selected).head(n)
    LOGGER.info("Diverse test sample (%d chips) by continent:\n%s",
                len(result), result["continent"].value_counts().to_string())
    return result["chip_id_str"].tolist()


def main() -> None:
    args = parse_args()

    if not cfg.UNIQUE_CHIP_MANIFEST_CSV.exists():
        LOGGER.error("Unique chip manifest not found: %s", cfg.UNIQUE_CHIP_MANIFEST_CSV)
        LOGGER.error("Run scripts/01_sample_and_tile.py first.")
        sys.exit(1)

    unique_chip_manifest = pd.read_csv(cfg.UNIQUE_CHIP_MANIFEST_CSV)
    LOGGER.info("Loaded unique chip manifest: %d chips", len(unique_chip_manifest))

    # Resolve chip_ids
    chip_ids = args.chip_ids
    if args.test_n is not None:
        chip_ids = diverse_sample(unique_chip_manifest, args.test_n)
        LOGGER.info("Test mode: submitting %d diverse chips", len(chip_ids))

    jobs = submit_all_exports(
        unique_chip_manifest=unique_chip_manifest,
        jobs_csv_path=cfg.EXPORT_JOBS_CSV,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        chip_ids=chip_ids,
        retry_errors=args.retry_errors,
    )

    if not args.dry_run and cfg.EXPORT_JOBS_CSV.exists():
        print_status_summary(cfg.EXPORT_JOBS_CSV)
        print("Next: monitor with  scripts/03_check_export_status.py")
        print(f"      then download mosaics from Google Drive '{cfg.DRIVE_FOLDER}'")
        print(f"      → place .tif files in {cfg.MOSAICS_DIR}")


if __name__ == "__main__":
    main()
