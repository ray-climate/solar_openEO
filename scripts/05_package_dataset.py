"""Step 5: Package all extracted chips into a single HDF5 dataset.

Reads all _image.tif and _mask.tif files from outputs/stage1/chips/
and stacks them into outputs/stage1/stage1_positives.h5.

HDF5 layout:
  /images    (N, 13, 256, 256)  float32
  /masks     (N, 256, 256)      uint8
  /chip_ids  (N,)               bytes
  /metadata  JSON string (chip_metadata DataFrame)

Usage:
  conda run -n geospatial python scripts/05_package_dataset.py
"""

import logging
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.dataset_writer import (
    build_chip_metadata_from_chip_dir,
    build_hdf5_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips-dir", default=str(cfg.CHIPS_DIR))
    parser.add_argument("--chip-metadata-csv", default=str(cfg.CHIP_METADATA_CSV))
    parser.add_argument("--out-h5", default=str(cfg.DATASET_H5))
    parser.add_argument(
        "--write-scanned-metadata",
        action="store_true",
        help="If metadata CSV is missing, scan chips dir and save recovered metadata CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chips_dir = Path(args.chips_dir)
    metadata_csv = Path(args.chip_metadata_csv)
    out_h5 = Path(args.out_h5)

    if metadata_csv.exists():
        chip_metadata = pd.read_csv(metadata_csv)
        LOGGER.info("Loaded chip metadata: %s", metadata_csv)
    else:
        LOGGER.warning("Chip metadata not found: %s", metadata_csv)
        LOGGER.warning("Scanning %s for extracted chip pairs instead.", chips_dir)
        chip_metadata = build_chip_metadata_from_chip_dir(chips_dir)
        if chip_metadata.empty:
            LOGGER.error("No extracted chip pairs found in: %s", chips_dir)
            sys.exit(1)
        if args.write_scanned_metadata:
            metadata_csv.parent.mkdir(parents=True, exist_ok=True)
            chip_metadata.to_csv(metadata_csv, index=False)
            LOGGER.info("Recovered chip metadata -> %s", metadata_csv)

    n_valid = (chip_metadata["coverage_ok"] == True).sum()  # noqa: E712
    LOGGER.info("Chip metadata: %d total, %d valid", len(chip_metadata), n_valid)

    if n_valid == 0:
        LOGGER.error("No valid chips to package. Check that extracted chip pairs exist.")
        sys.exit(1)

    build_hdf5_dataset(
        chips_dir=chips_dir,
        chip_metadata=chip_metadata,
        output_h5_path=out_h5,
    )

    print(f"Dataset written → {out_h5}")


if __name__ == "__main__":
    main()
