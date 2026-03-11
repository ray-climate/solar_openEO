"""One-off: re-export two test chips with NN resampling fix to a new Drive folder.

Usage:
  conda run -n geospatial python scripts/test_nn_export.py
"""

import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.tiling import chip_id_to_bounds
from mosaic_module import create_temporal_mosaic
from mosaic_module.io_helpers import initialize_ee

TEST_DRIVE_FOLDER = "solar_openEO_stage1_v3_nntest"
TEST_CHIPS = ["c+000048_r+002705", "c+000054_r+002135"]


def chip_to_export_name(chip_id_str: str) -> str:
    safe = chip_id_str.replace("+", "p").replace("-", "m")
    return f"stage1_{safe}"


def main():
    initialize_ee()

    for chip_id in TEST_CHIPS:
        parts = chip_id.split("_")
        col = int(parts[0][1:])
        row = int(parts[1][1:])
        bounds = chip_id_to_bounds(col, row)

        # Rough center for the API (not used when aoi_bounds_3857 is set)
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        # Convert EPSG:3857 center to lat/lon for the API signature
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(cx, cy)

        export_name = chip_to_export_name(chip_id)
        print(f"Submitting {chip_id} → {export_name} to {TEST_DRIVE_FOLDER} ...")

        result = create_temporal_mosaic(
            center_lat=lat,
            center_lon=lon,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            export_name=export_name,
            drive_folder=TEST_DRIVE_FOLDER,
            export_rgb=False,
            aoi_bounds_3857=bounds,
            **cfg.MOSAIC_PARAMS,
        )

        print(f"  task_id={result['task_id']}  state={result['task_state']}")

    print(f"\nBoth exports submitted to Drive folder: {TEST_DRIVE_FOLDER}")
    print("Monitor with: earthengine task list")


if __name__ == "__main__":
    main()
