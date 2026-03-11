"""Step 7: Generate quickview PNGs for all completed chips.

Downloads mosaics from Google Drive (if not already local), generates a
side-by-side figure (RGB left, mask overlay right) for each chip, and
optionally removes the TIF afterwards to save disk space.

Outputs:
  outputs/stage1/quickviews/<chip_id>_quickview.png

Usage:
  # Download from Drive + generate quickviews (keep TIFs)
  conda run -n geospatial python scripts/07_generate_quickviews.py

  # Same, but delete TIFs after generating each quickview
  conda run -n geospatial python scripts/07_generate_quickviews.py --cleanup

  # Generate only for mosaics already on disk (no rclone download)
  conda run -n geospatial python scripts/07_generate_quickviews.py --skip-download

  # Process specific chips only
  conda run -n geospatial python scripts/07_generate_quickviews.py --chip-ids c-000046_r+001798
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction_pipeline import config as cfg
from extraction_pipeline.chip_extractor import find_chip_mosaic_path
from extraction_pipeline.sampling import assign_continent, filter_continent, load_polygons
from extraction_pipeline.tiling import chip_id_to_bounds

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

QUICKVIEW_DIR = cfg.OUTPUT_DIR / "quickviews"


# ---------------------------------------------------------------------------
# RGB helpers (same logic as dataset_writer)
# ---------------------------------------------------------------------------

def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((band.astype(np.float32) - p_lo) / (p_hi - p_lo), 0, 1)


def make_rgb(mosaic_path: Path) -> np.ndarray:
    """Read B4/B3/B2 from a mosaic TIF and return (256, 256, 3) uint8 RGB."""
    sz = cfg.CHIP_SIZE_PX
    with rasterio.open(mosaic_path) as src:
        # Read at exact chip size to handle off-by-one GEE exports
        b4 = src.read(4, out_shape=(sz, sz),
                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        b3 = src.read(3, out_shape=(sz, sz),
                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        b2 = src.read(2, out_shape=(sz, sz),
                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    # Replace NaN with 0 before stretch
    b4 = np.nan_to_num(b4, nan=0.0)
    b3 = np.nan_to_num(b3, nan=0.0)
    b2 = np.nan_to_num(b2, nan=0.0)
    r = (_percentile_stretch(b4) * 255).astype(np.uint8)
    g = (_percentile_stretch(b3) * 255).astype(np.uint8)
    b = (_percentile_stretch(b2) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Mask rasterisation
# ---------------------------------------------------------------------------

def rasterize_mask(polygon_geoms, chip_col: int, chip_row: int) -> np.ndarray:
    """Burn polygon geometries into a binary chip mask (256x256 uint8)."""
    xmin, ymin, xmax, ymax = chip_id_to_bounds(chip_col, chip_row)
    transform = from_bounds(xmin, ymin, xmax, ymax, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX)
    geom_list = [(g, 1) for g in polygon_geoms if g is not None and not g.is_empty]
    if not geom_list:
        return np.zeros((cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX), dtype=np.uint8)
    return rasterize(
        shapes=geom_list,
        out_shape=(cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


# ---------------------------------------------------------------------------
# Quickview figure
# ---------------------------------------------------------------------------

def write_quickview(rgb: np.ndarray, mask: np.ndarray, chip_id_str: str, out_dir: Path) -> None:
    """Write a side-by-side PNG: RGB (left) + mask overlay (right)."""
    overlay = rgb.astype(float) / 255.0
    if mask.any():
        panel_px = mask == 1
        mask_color = np.array([1.0, 0.15, 0.15])
        overlay[panel_px] = 0.55 * overlay[panel_px] + 0.45 * mask_color

    n_px = int(mask.sum())
    panel_pct = 100.0 * n_px / (cfg.CHIP_SIZE_PX ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    ax1.imshow(rgb)
    ax1.set_title("RGB (B4/B3/B2)", fontsize=8)
    ax1.axis("off")
    ax2.imshow(overlay)
    ax2.set_title(f"Mask overlay  {n_px} px  ({panel_pct:.2f}%)", fontsize=8)
    ax2.axis("off")
    fig.suptitle(chip_id_str, fontsize=7, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"{chip_id_str}_quickview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# rclone download
# ---------------------------------------------------------------------------

def download_mosaics_from_drive(
    export_jobs: pd.DataFrame,
    mosaics_dir: Path,
    drive_folder: str,
) -> int:
    """Download completed mosaics from Drive via rclone. Returns count of new files."""
    import subprocess

    mosaics_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in mosaics_dir.glob("*.tif")}

    completed = export_jobs[export_jobs["status"] == "COMPLETED"]
    to_download = completed[~completed["export_name"].isin(existing)]

    if to_download.empty:
        LOGGER.info("All completed mosaics already on disk.")
        return 0

    LOGGER.info("Downloading %d mosaics from Drive folder '%s' ...", len(to_download), drive_folder)

    # Use rclone copy with include filters for efficiency
    # Build a filter file with all needed filenames
    filter_path = mosaics_dir / "_rclone_filter.txt"
    with open(filter_path, "w") as f:
        for name in to_download["export_name"]:
            f.write(f"+ {name}.tif\n")
        f.write("- *\n")

    cmd = [
        "rclone", "copy",
        f"rui.song90:{drive_folder}",
        str(mosaics_dir),
        "--filter-from", str(filter_path),
        "--transfers", "8",
        "--progress",
    ]
    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    filter_path.unlink(missing_ok=True)

    if result.returncode != 0:
        LOGGER.warning("rclone returned exit code %d — some files may not have downloaded.", result.returncode)

    new_count = len(list(mosaics_dir.glob("*.tif"))) - len(existing)
    LOGGER.info("Downloaded %d new mosaic TIFs.", new_count)
    return new_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chip-ids", nargs="+", default=None,
                   help="Process only these chip_id_str values.")
    p.add_argument("--cleanup", action="store_true",
                   help="Delete each mosaic TIF after generating its quickview.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip rclone download; only process mosaics already on disk.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate quickviews even if they already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load polygon data for mask rasterisation ---
    for required in [cfg.CHIP_MANIFEST_CSV, cfg.EXPORT_JOBS_CSV]:
        if not required.exists():
            LOGGER.error("Required file not found: %s", required)
            sys.exit(1)

    chip_manifest = pd.read_csv(cfg.CHIP_MANIFEST_CSV)
    export_jobs = pd.read_csv(cfg.EXPORT_JOBS_CSV, dtype=str)

    LOGGER.info("Loading polygon dataset for mask generation...")
    full_gdf = load_polygons(str(cfg.GPKG_PATH))
    full_gdf = assign_continent(full_gdf)
    if cfg.FILTER_CONTINENT:
        full_gdf = filter_continent(full_gdf, cfg.FILTER_CONTINENT)
    mask_gdf = full_gdf.to_crs("EPSG:3857")
    LOGGER.info("Loaded %d mask polygons.", len(mask_gdf))

    # --- Download from Drive if needed ---
    if not args.skip_download:
        download_mosaics_from_drive(export_jobs, cfg.MOSAICS_DIR, cfg.DRIVE_FOLDER)

    # --- Build unique chip list ---
    unique_chips = chip_manifest.drop_duplicates("chip_id_str").copy()
    if args.chip_ids:
        unique_chips = unique_chips[unique_chips["chip_id_str"].isin(args.chip_ids)]

    QUICKVIEW_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(unique_chips)
    n_ok, n_skip, n_miss = 0, 0, 0

    for i, (_, row) in enumerate(unique_chips.iterrows()):
        chip_id_str = row["chip_id_str"]
        chip_col = int(row["chip_col"])
        chip_row = int(row["chip_row"])

        out_path = QUICKVIEW_DIR / f"{chip_id_str}_quickview.png"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        # Locate mosaic TIF
        mosaic_path = find_chip_mosaic_path(chip_id_str, cfg.MOSAICS_DIR)
        if mosaic_path is None:
            n_miss += 1
            continue

        # Generate RGB from mosaic
        rgb = make_rgb(mosaic_path)

        # Rasterise mask from polygons
        bounds = chip_id_to_bounds(chip_col, chip_row)
        chip_box = shapely_box(*bounds)
        candidate_idx = list(mask_gdf.sindex.intersection(bounds))
        geoms = mask_gdf.iloc[candidate_idx].geometry[
            mask_gdf.iloc[candidate_idx].geometry.intersects(chip_box)
        ].tolist()
        mask = rasterize_mask(geoms, chip_col, chip_row)

        write_quickview(rgb, mask, chip_id_str, QUICKVIEW_DIR)
        n_ok += 1

        if args.cleanup:
            mosaic_path.unlink()

        if (n_ok + n_skip) % 200 == 0:
            LOGGER.info("  Progress: %d/%d  (generated=%d, skipped=%d, missing=%d)",
                        i + 1, n_total, n_ok, n_skip, n_miss)

    print(f"\nQuickview generation complete")
    print(f"  Total chips         : {n_total}")
    print(f"  Generated           : {n_ok}")
    print(f"  Skipped (existing)  : {n_skip}")
    print(f"  Missing (no mosaic) : {n_miss}")
    print(f"  Output directory    : {QUICKVIEW_DIR}")


if __name__ == "__main__":
    main()
