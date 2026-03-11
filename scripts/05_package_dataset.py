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
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.dataset_writer import build_hdf5_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    if not cfg.CHIP_METADATA_CSV.exists():
        LOGGER.error("Chip metadata not found: %s", cfg.CHIP_METADATA_CSV)
        LOGGER.error("Run scripts/04_extract_chips.py first.")
        sys.exit(1)

    chip_metadata = pd.read_csv(cfg.CHIP_METADATA_CSV)
    n_valid = (chip_metadata["coverage_ok"] == True).sum()  # noqa: E712
    LOGGER.info("Chip metadata: %d total, %d valid", len(chip_metadata), n_valid)

    if n_valid == 0:
        LOGGER.error("No valid chips to package. Check that mosaics were downloaded.")
        sys.exit(1)

    build_hdf5_dataset(
        chips_dir=cfg.CHIPS_DIR,
        chip_metadata=chip_metadata,
        output_h5_path=cfg.DATASET_H5,
    )

    print(f"Dataset written → {cfg.DATASET_H5}")


if __name__ == "__main__":
    main()
