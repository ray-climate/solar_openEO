"""Write chip GeoTIFFs and package them into an HDF5 dataset.

Per chip, two files are written:
  <chip_id>_image.tif       — 13-band spectral data (float32, EPSG:3857)
  <chip_id>_mask.tif        — 1-band uint8 binary panel mask (1=panel, 0=background)
  <chip_id>_quickview.png   — 2-panel RGB quickview + mask overlay (local QA only)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from . import config as cfg

LOGGER = logging.getLogger(__name__)

_CHIP_ID_RE = re.compile(r"^(c[+-]\d+_r[+-]\d+)_image$")

# Hardcoded WKT for EPSG:3857 — avoids PROJ database lookup at write time.
_CRS_3857 = rasterio.CRS.from_wkt(
    'PROJCS["WGS 84 / Pseudo-Mercator",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","3857"]]'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Linear percentile stretch to [0, 1], ignoring zero/nodata pixels."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((band.astype(np.float32) - p_lo) / (p_hi - p_lo), 0, 1)


def _make_rgb(image_arr: np.ndarray) -> np.ndarray:
    """Build (H, W, 3) uint8 RGB from (13, H, W) using B4/B3/B2."""
    r = (_percentile_stretch(image_arr[3]) * 255).astype(np.uint8)
    g = (_percentile_stretch(image_arr[2]) * 255).astype(np.uint8)
    b = (_percentile_stretch(image_arr[1]) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Per-chip GeoTIFF writer
# ---------------------------------------------------------------------------

def write_chip_geotiff(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    chip_id_str: str,
    chip_bounds_3857: tuple[float, float, float, float],
    chips_dir: str | Path,
) -> None:
    """Write two georeferenced GeoTIFFs + one quickview PNG for one chip.

    Files written (all EPSG:3857, 10 m/px):
      ``<chip_id>_image.tif``      — (13, 256, 256) float32  — 13-band spectral
      ``<chip_id>_mask.tif``       — (1,  256, 256) uint8    — binary panel mask
      ``<chip_id>_quickview.png``  — 2-panel RGB + mask overlay (local QA)
    """
    chips_dir = Path(chips_dir)
    chips_dir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = chip_bounds_3857
    transform = from_bounds(xmin, ymin, xmax, ymax, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX)
    crs = _CRS_3857

    # --- 13-band spectral image ---
    n_bands, h, w = image_arr.shape
    with rasterio.open(
        chips_dir / f"{chip_id_str}_image.tif", "w",
        driver="GTiff", height=h, width=w, count=n_bands,
        dtype=np.float32, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(image_arr)
        dst.update_tags(bands=",".join(cfg.BANDS))

    # --- Binary mask ---
    mask_2d = mask_arr if mask_arr.ndim == 2 else mask_arr[0]
    mask_write = mask_2d[np.newaxis]   # (1, 256, 256)
    with rasterio.open(
        chips_dir / f"{chip_id_str}_mask.tif", "w",
        driver="GTiff", height=cfg.CHIP_SIZE_PX, width=cfg.CHIP_SIZE_PX, count=1,
        dtype=np.uint8, crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(mask_write)
        dst.update_tags(description="binary solar panel mask (1=panel 0=background)")

    # --- 2-panel quickview PNG ---
    _write_quickview_png(image_arr, mask_2d, chip_id_str, chips_dir)


def _write_quickview_png(
    image_arr: np.ndarray,
    mask_2d: np.ndarray,
    chip_id_str: str,
    chips_dir: Path,
) -> None:
    """Write a 2-panel PNG: RGB quickview (left) + mask overlay (right)."""
    rgb = _make_rgb(image_arr)

    # Mask overlay: blend red onto panel pixels
    overlay = rgb.astype(float) / 255.0
    if mask_2d.any():
        panel_px = mask_2d == 1
        mask_color = np.array([1.0, 0.15, 0.15])
        overlay[panel_px] = 0.55 * overlay[panel_px] + 0.45 * mask_color

    n_px = int(mask_2d.sum())
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
    fig.savefig(chips_dir / f"{chip_id_str}_quickview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HDF5 dataset packager
# ---------------------------------------------------------------------------

def build_hdf5_dataset(
    chips_dir: str | Path,
    chip_metadata: pd.DataFrame,
    output_h5_path: str | Path,
) -> None:
    """Stack all valid chip GeoTIFFs into a single HDF5 file.

    HDF5 layout
    -----------
    /images    (N, 13, 256, 256)  float32   — spectral chips
    /masks     (N, 256, 256)      uint8     — binary panel masks
    /chip_ids  (N,)               bytes     — chip_id_str per sample
    /metadata  scalar str         JSON      — chip_metadata DataFrame

    Only rows where ``coverage_ok == True`` are included.
    """
    output_h5_path = Path(output_h5_path)
    chips_dir = Path(chips_dir)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    valid = chip_metadata[chip_metadata["coverage_ok"] == True].copy()  # noqa: E712
    LOGGER.info("Packaging %d valid chips into %s", len(valid), output_h5_path)

    n = len(valid)
    images = np.zeros((n, cfg.N_BANDS, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX), dtype=np.float32)
    masks = np.zeros((n, cfg.CHIP_SIZE_PX, cfg.CHIP_SIZE_PX), dtype=np.uint8)
    chip_ids = []

    for i, (_, row) in enumerate(valid.iterrows()):
        cid = row["chip_id_str"]
        img_path = chips_dir / f"{cid}_image.tif"
        msk_path = chips_dir / f"{cid}_mask.tif"

        if not img_path.exists() or not msk_path.exists():
            LOGGER.warning("Missing files for chip %s — filling with zeros", cid)
            chip_ids.append(cid.encode())
            continue

        with rasterio.open(img_path) as src:
            images[i] = src.read().astype(np.float32)

        with rasterio.open(msk_path) as src:
            masks[i] = src.read(1).astype(np.uint8)

        chip_ids.append(cid.encode())

        if (i + 1) % 100 == 0:
            LOGGER.info("  Packed %d / %d chips", i + 1, n)

    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
        f.create_dataset("masks", data=masks, compression="gzip", compression_opts=4)
        f.create_dataset("chip_ids", data=np.array(chip_ids))
        f.attrs["metadata"] = valid.to_json(orient="records")
        f.attrs["bands"] = json.dumps(cfg.BANDS)
        f.attrs["chip_size_px"] = cfg.CHIP_SIZE_PX
        f.attrs["scale_m"] = cfg.SCALE_M
        f.attrs["crs"] = cfg.GRID_CRS
        f.attrs["start_date"] = cfg.START_DATE
        f.attrs["end_date"] = cfg.END_DATE

    LOGGER.info("HDF5 dataset written: %s  shape=images%s", output_h5_path, images.shape)
    _print_dataset_summary(valid, masks)


def _print_dataset_summary(chip_metadata: pd.DataFrame, masks: np.ndarray) -> None:
    n = len(chip_metadata)
    total_px = n * cfg.CHIP_SIZE_PX ** 2
    panel_px = int(masks.sum())
    print(f"\nStage-1 dataset summary")
    print(f"  Chips           : {n}")
    print(f"  Panel pixels    : {panel_px:,} / {total_px:,}  ({100*panel_px/total_px:.2f}%)")
    if "continent" in chip_metadata.columns:
        print(f"  By continent    :")
        for cont, cnt in chip_metadata["continent"].value_counts().items():
            print(f"    {cont:15s}: {cnt}")
    print()


def parse_chip_id_str(chip_id_str: str) -> tuple[int, int]:
    """Parse ``c+000123_r-000456`` into ``(chip_col, chip_row)``."""
    col_str, row_str = chip_id_str.split("_")
    return int(col_str[1:]), int(row_str[1:])


def build_chip_metadata_from_chip_dir(chips_dir: str | Path) -> pd.DataFrame:
    """Build minimal chip metadata by scanning ``*_image.tif``/``*_mask.tif`` pairs.

    This is a recovery path for cases where extracted chips exist but
    ``chip_metadata.csv`` is missing.
    """
    chips_dir = Path(chips_dir)
    records: list[dict] = []

    image_files = sorted(chips_dir.glob("*_image.tif"))
    for image_path in image_files:
        match = _CHIP_ID_RE.match(image_path.stem)
        if not match:
            continue

        chip_id_str = match.group(1)
        chip_col, chip_row = parse_chip_id_str(chip_id_str)
        mask_path = chips_dir / f"{chip_id_str}_mask.tif"
        coverage_ok = mask_path.exists()
        n_panel_px = 0
        panel_frac = 0.0

        if coverage_ok:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.uint8)
            n_panel_px = int(mask.sum())
            panel_frac = round(n_panel_px / (cfg.CHIP_SIZE_PX ** 2), 6)

        records.append({
            "chip_id_str": chip_id_str,
            "chip_col": chip_col,
            "chip_row": chip_row,
            "n_panel_px": n_panel_px,
            "panel_frac": panel_frac,
            "coverage_ok": coverage_ok,
            "continent": "",
        })

    if not records:
        return pd.DataFrame(columns=[
            "chip_id_str", "chip_col", "chip_row", "n_panel_px",
            "panel_frac", "coverage_ok", "continent",
        ])

    df = pd.DataFrame(records).sort_values("chip_id_str").reset_index(drop=True)
    LOGGER.info("Scanned %d extracted chips from %s", len(df), chips_dir)
    return df
