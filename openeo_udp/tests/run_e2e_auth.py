#!/usr/bin/env python
"""Authenticate + run e2e test in a single process.

Tests the full pipeline:
  1. Connect to CDSE OpenEO backend (device code auth)
  2. Download multi-temporal L1C + SCL stack from OpenEO
  3. Apply SLIC-based temporal mosaic UDF locally (faithful to GEE)
  4. Run local UDF inference on the mosaic
  5. Compare against ground truth from HDF5 dataset
  6. Generate a 3-panel quickview (RGB | ground truth | detection)

Usage:
    python openeo_udp/tests/run_e2e_auth.py [--skip-download]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
TEST_OUTPUT_DIR = REPO / "openeo_udp" / "tests" / "test_outputs"

# --- Test AOI ---
# Test chip: c-000179_r+001853 (Spain, ~40K positive pixels, TEST split)
# This chip was NOT used in training or validation.
TEST_CHIP_ID = "c-000179_r+001853"
TEST_CHIP_COL = -179
TEST_CHIP_ROW = 1853

# Compute EPSG:3857 bounds from chip grid
CELL_SIZE = 2560  # metres
CHIP_XMIN = TEST_CHIP_COL * CELL_SIZE
CHIP_YMIN = TEST_CHIP_ROW * CELL_SIZE
CHIP_XMAX = CHIP_XMIN + CELL_SIZE
CHIP_YMAX = CHIP_YMIN + CELL_SIZE

# EPSG:4326 bbox (pre-computed from EPSG:3857 bounds above)
TEST_AOI = {
    "west": -4.116440,
    "south": 39.153479,
    "east": -4.093443,
    "north": 39.171310,
    "crs": "EPSG:4326",
}

# Summer temporal window matching training data
TEST_TEMPORAL = ["2024-05-01", "2024-07-31"]

# L1C spectral bands (all 13, matching training data)
S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


def load_ground_truth() -> np.ndarray | None:
    """Load the ground truth mask for the test chip from HDF5."""
    manifest_path = REPO / "outputs/training_prep/stage1_v3/split_manifest.csv"
    h5_path = REPO / "outputs/stage1/stage1_positives.h5"

    if not manifest_path.exists() or not h5_path.exists():
        print("   [WARN] Ground truth data not found")
        return None

    # Find the chip index
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["chip_id"] == TEST_CHIP_ID:
                idx = int(row["index"])
                break
        else:
            print(f"   [WARN] Chip {TEST_CHIP_ID} not found in manifest")
            return None

    with h5py.File(h5_path, "r") as h5:
        mask = h5["masks"][idx].astype(np.float32)  # (256, 256)

    print(f"   Ground truth loaded: {TEST_CHIP_ID}, {int(np.sum(mask > 0))} positive pixels")
    return mask


def generate_quickview(
    image_hwc: np.ndarray,
    ground_truth: np.ndarray | None,
    detection: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
):
    """Generate a 3-panel quickview: RGB | ground truth | detection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    h, w = detection.shape

    # Build RGB from bands B04 (idx 3), B03 (idx 2), B02 (idx 1)
    if image_hwc.shape[2] >= 4:
        rgb = image_hwc[:, :, [3, 2, 1]].astype(np.float32)
        # Scale to [0, 1] using typical L1C range
        p2 = np.percentile(rgb[rgb > 0], 2) if np.any(rgb > 0) else 0
        p98 = np.percentile(rgb[rgb > 0], 98) if np.any(rgb > 0) else 1
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1), 0, 1)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float32)

    n_panels = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Overlay colormap: transparent for 0, red for 1
    overlay_cmap = ListedColormap(["none", "red"])

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (B4/B3/B2)", fontsize=12)
    axes[0].axis("off")

    if ground_truth is not None:
        # Panel 2: RGB + ground truth overlay
        axes[1].imshow(rgb)
        mask_display = (ground_truth > 0.5).astype(np.float32)
        # Crop/resize ground truth to match mosaic dimensions if needed
        gh, gw = mask_display.shape
        if gh != h or gw != w:
            # Resize ground truth to match mosaic
            from scipy.ndimage import zoom
            zoom_h = h / gh
            zoom_w = w / gw
            mask_display = zoom(mask_display, (zoom_h, zoom_w), order=0)
        axes[1].imshow(mask_display, cmap=overlay_cmap, alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth ({int(np.sum(ground_truth > 0.5))} px)", fontsize=12)
        axes[1].axis("off")

        # Panel 3: RGB + detection overlay
        axes[2].imshow(rgb)
        axes[2].imshow(detection.astype(np.float32), cmap=overlay_cmap, alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f"Detection (t={0.80}, {int(np.sum(detection))} px)", fontsize=12)
        axes[2].axis("off")
    else:
        # Panel 2: RGB + detection overlay
        axes[1].imshow(rgb)
        axes[1].imshow(detection.astype(np.float32), cmap=overlay_cmap, alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title(f"Detection (t={0.80}, {int(np.sum(detection))} px)", fontsize=12)
        axes[1].axis("off")

    fig.suptitle(
        f"Solar PV Detection — E2E Test\n"
        f"Chip: {TEST_CHIP_ID} (test split, unseen during training)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Quickview saved: {output_path}")


def main():
    import openeo

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OpenEO job, use previously downloaded multi-temporal stack")
    args = parser.parse_args()

    stack_path = TEST_OUTPUT_DIR / "test_stack_multitemporal.nc"
    mosaic_path = TEST_OUTPUT_DIR / "test_mosaic.tif"

    print("=" * 60)
    print("Solar PV UDP — End-to-End OpenEO Test")
    print("=" * 60)
    print(f"Test chip: {TEST_CHIP_ID} (test split, Spain)")
    print(f"AOI: {TEST_AOI}")

    if args.skip_download and stack_path.exists():
        print(f"\nSkipping download, using: {stack_path}")
    else:
        # --- Step 1: Authenticate ---
        print("\n1. Connecting and authenticating...")
        conn = openeo.connect(BACKEND_URL)
        conn.authenticate_oidc_device()
        user = conn.describe_account().get("user_id", "unknown")
        print(f"   Authenticated as: {user}")

        # --- Step 2: Download multi-temporal L1C + SCL stack ---
        #
        # Strategy: download full temporal stack (all scenes) so we can
        # apply the SLIC-based temporal mosaic UDF locally.
        # This is the most faithful approach to the GEE training pipeline.
        #
        # OpenEO loads:
        #   - L1C: 13 spectral bands (TOA reflectance, matches training)
        #   - L2A: SCL band (cloud classification, for masking)
        # Merged into 14 bands per timestep, preserving temporal dimension.
        print("\n2. Building multi-temporal stack (L1C + SCL)...")
        print(f"   L1C (13 bands) + L2A SCL → 14-band temporal stack")
        print(f"   Temporal: {TEST_TEMPORAL}")

        # Load L1C spectral data (all 13 bands, TOA reflectance)
        # Resample to EPSG:3857 at 10m to match training grid exactly.
        # Without this, CDSE returns native EPSG:4326 at ~10m → 202x202
        # instead of the required 256x256 at 10m in EPSG:3857.
        s2_l1c = conn.load_collection(
            "SENTINEL2_L1C",
            spatial_extent=TEST_AOI,
            temporal_extent=TEST_TEMPORAL,
            bands=S2_L1C_BANDS,
        ).resample_spatial(resolution=10, projection="EPSG:3857")

        # Load L2A SCL band for cloud classification
        s2_scl = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=TEST_AOI,
            temporal_extent=TEST_TEMPORAL,
            bands=["SCL"],
        ).resample_spatial(resolution=10, projection="EPSG:3857")

        # Merge L1C spectral + SCL into one datacube (14 bands)
        # Keep temporal dimension — no reduce, no median
        merged = s2_l1c.merge_cubes(s2_scl)

        # --- Step 3: Submit batch job ---
        print("\n3. Submitting batch job (multi-temporal stack)...")
        job = merged.create_job(
            title=f"Solar PV e2e — {TEST_CHIP_ID} L1C+SCL stack",
            out_format="netCDF",
        )
        job.start_job()
        print(f"   Job ID: {job.job_id}")

        # --- Step 4: Wait for completion ---
        print("\n4. Waiting for job completion...")
        t0 = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - t0
            print(f"   [{elapsed:5.0f}s] {status}")

            if status == "finished":
                break
            elif status in ("error", "canceled"):
                try:
                    logs = job.logs()
                    for log in logs[-10:]:
                        msg = log.get("message", str(log)) if isinstance(log, dict) else str(log)
                        print(f"   LOG: {msg}")
                except Exception:
                    pass
                raise RuntimeError(f"Job failed: {status}")

            time.sleep(15)

        # --- Step 5: Download results ---
        print("\n5. Downloading results...")
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Clean previous netCDF outputs
        for old in TEST_OUTPUT_DIR.glob("*.nc"):
            old.unlink()

        results = job.get_results()
        results.download_files(TEST_OUTPUT_DIR)

        nc_files = list(TEST_OUTPUT_DIR.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError("No netCDF files in results")

        downloaded = nc_files[0]
        if downloaded != stack_path:
            downloaded.rename(stack_path)
        print(f"   Downloaded: {stack_path} ({stack_path.stat().st_size / 1e6:.1f} MB)")

    # --- Step 6: Load multi-temporal stack and apply mosaic UDF ---
    print("\n6. Loading multi-temporal stack...")
    import xarray as xr

    ds = xr.open_dataset(stack_path)
    print(f"   Dataset variables: {list(ds.data_vars)}")
    print(f"   Dimensions: {dict(ds.dims)}")

    # Extract arrays — shape depends on CDSE output format
    # Expected: each band as a variable with dims (t, y, x) or similar
    band_vars = [v for v in ds.data_vars if v in S2_L1C_BANDS or v == "SCL"]
    print(f"   Band variables found: {band_vars}")

    if len(band_vars) >= 14:
        # Each band is a separate variable
        spectral_bands = []
        for bname in S2_L1C_BANDS:
            if bname in ds:
                spectral_bands.append(ds[bname].values)
            else:
                raise ValueError(f"Missing band {bname} in dataset")

        # Stack spectral: list of (T, H, W) → (T, 13, H, W)
        spectral_stack = np.stack(spectral_bands, axis=1).astype(np.float32)
        scl_stack = ds["SCL"].values.astype(np.int32)  # (T, H, W)
    else:
        # Bands might be in a single variable with a bands dimension
        # Try to find the main data variable
        for varname in ds.data_vars:
            var = ds[varname]
            if len(var.dims) >= 3:
                data = var.values
                print(f"   Using variable '{varname}': shape {data.shape}, dims {var.dims}")
                break
        else:
            raise ValueError(f"Cannot find suitable data variable in {list(ds.data_vars)}")

        # Figure out dimension order and split spectral/SCL
        if data.ndim == 4:
            # (t, bands, y, x) or some permutation
            dims = list(var.dims)
            t_dim = next((d for d in dims if d in ("t", "time")), dims[0])
            b_dim = next((d for d in dims if d in ("bands", "band")), dims[1])
            t_idx = dims.index(t_dim)
            b_idx = dims.index(b_dim)

            # Transpose to (t, bands, y, x)
            target_order = [t_idx, b_idx]
            spatial = [i for i in range(4) if i not in target_order]
            data = np.transpose(data, target_order + spatial).astype(np.float32)

            spectral_stack = data[:, :13, :, :]
            if data.shape[1] >= 14:
                scl_stack = data[:, 13, :, :].astype(np.int32)
            else:
                print("   [WARN] No SCL band in stack, falling back to median")
                scl_stack = np.full(data[:, 0, :, :].shape, 4, dtype=np.int32)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    ds.close()

    t_scenes, n_bands, h_px, w_px = spectral_stack.shape
    print(f"   Spectral stack: {t_scenes} scenes × {n_bands} bands × {h_px}×{w_px}")
    print(f"   SCL stack: {scl_stack.shape}")

    # Apply SLIC-based temporal mosaic UDF
    print("\n7. Applying SLIC-based temporal mosaic UDF...")
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic

    t0 = time.time()
    composite, mosaic_info = create_temporal_mosaic(spectral_stack, scl_stack)
    elapsed = time.time() - t0

    print(f"   Mosaic shape: {composite.shape} (C, H, W)")
    print(f"   Clusters: {mosaic_info['n_clusters']}")
    print(f"   Scenes used: {mosaic_info['n_scenes_used']}")
    rescue_pct = mosaic_info["fill_mode"].sum() / mosaic_info["fill_mode"].size * 100
    print(f"   Rescue-filled: {rescue_pct:.1f}%")
    print(f"   Time: {elapsed:.1f}s")

    # Save mosaic as GeoTIFF for inspection
    print(f"\n   Saving mosaic to {mosaic_path}...")
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    # Use EPSG:3857 chip bounds (matches training grid exactly)
    transform = from_bounds(
        CHIP_XMIN, CHIP_YMIN, CHIP_XMAX, CHIP_YMAX,
        composite.shape[2], composite.shape[1],
    )
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": composite.shape[2],
        "height": composite.shape[1],
        "count": composite.shape[0],
        "crs": CRS.from_epsg(3857),
        "transform": transform,
    }
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(composite)
    print(f"   Saved: {mosaic_path}")

    # Validate
    data = composite
    n_bands = data.shape[0]
    print(f"\n   Bands: {n_bands}")
    for i, name in enumerate(S2_L1C_BANDS[:n_bands]):
        band = data[i]
        valid = band[np.isfinite(band) & (band != 0)]
        if valid.size > 0:
            print(f"   {name}: min={valid.min():.0f}, mean={valid.mean():.0f}, max={valid.max():.0f}")

    if n_bands < 13:
        raise ValueError(f"Only {n_bands} bands — model requires 13 (L1C)")

    image_hwc = np.transpose(data[:13], (1, 2, 0))  # (H, W, 13)
    print(f"   Image shape for model: {image_hwc.shape}")

    # --- Step 8: Load ground truth ---
    print("\n8. Loading ground truth mask...")
    ground_truth = load_ground_truth()

    # --- Step 9: Run local inference ---
    print("\n9. Running UDF inference locally...")
    from openeo_udp.udf.solar_pv_inference import (
        _get_model_and_stats,
        normalize_zscore,
    )

    model, band_stats, registry = _get_model_and_stats()
    threshold = float(registry.get("threshold", 0.80))

    h, w, c = image_hwc.shape
    all_probs = np.zeros((h, w), dtype=np.float32)
    chip_count = 0
    for y0 in range(0, h, 256):
        for x0 in range(0, w, 256):
            chip = image_hwc[y0:y0 + 256, x0:x0 + 256, :]
            ph, pw = chip.shape[0], chip.shape[1]
            if ph < 256 or pw < 256:
                padded = np.zeros((256, 256, c), dtype=np.float32)
                padded[:ph, :pw, :] = chip
                chip = padded
            chip_norm = normalize_zscore(chip[np.newaxis], band_stats)
            pred = model.predict(chip_norm, verbose=0)
            all_probs[y0:y0 + ph, x0:x0 + pw] = pred[0, :ph, :pw, 0]
            chip_count += 1

    binary = (all_probs > threshold).astype(np.uint8)
    pos_pixels = int(np.sum(binary))
    total_pixels = binary.size
    pos_rate = pos_pixels / total_pixels * 100

    print(f"   Chips processed: {chip_count}")
    print(f"   Probability range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(f"   Threshold: {threshold}")
    print(f"   Positive pixels: {pos_pixels} / {total_pixels} ({pos_rate:.2f}%)")

    # Compute Dice against ground truth if available
    if ground_truth is not None:
        gt = ground_truth
        # Resize detection to ground truth size for comparison
        gh, gw = gt.shape
        if gh != h or gw != w:
            print(f"   [NOTE] Mosaic {h}x{w} vs ground truth {gh}x{gw} — resizing for comparison")
            from scipy.ndimage import zoom
            binary_resized = zoom(binary.astype(np.float32), (gh / h, gw / w), order=0)
            binary_resized = (binary_resized > 0.5).astype(np.float32)
        else:
            binary_resized = binary.astype(np.float32)

        gt_bin = (gt > 0.5).astype(np.float32)
        intersection = np.sum(gt_bin * binary_resized)
        denom = np.sum(gt_bin) + np.sum(binary_resized)
        dice = (2.0 * intersection + 1.0) / (denom + 1.0)
        print(f"   Dice vs ground truth: {dice:.4f}")

    # --- Step 10: Save results ---
    print("\n10. Saving results...")

    # Binary mask GeoTIFF
    mask_path = TEST_OUTPUT_DIR / "detection_mask.tif"
    p = profile.copy()
    p.update(count=1, dtype="uint8")
    with rasterio.open(mask_path, "w", **p) as dst:
        dst.write(binary, 1)
    print(f"   Binary mask: {mask_path}")

    # Probability map GeoTIFF
    prob_path = TEST_OUTPUT_DIR / "detection_probabilities.tif"
    p.update(count=1, dtype="float32")
    with rasterio.open(prob_path, "w", **p) as dst:
        dst.write(all_probs, 1)
    print(f"   Probability map: {prob_path}")

    # 3-panel quickview
    print("\n11. Generating quickview...")
    quickview_path = TEST_OUTPUT_DIR / "quickview_e2e.png"
    generate_quickview(
        image_hwc=image_hwc,
        ground_truth=ground_truth,
        detection=binary,
        probs=all_probs,
        output_path=quickview_path,
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("END-TO-END TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nTest chip: {TEST_CHIP_ID} (test split, unseen during training)")
    print(f"Artifacts in: {TEST_OUTPUT_DIR}/")
    print(f"  test_stack_multitemporal.nc   — multi-temporal L1C+SCL stack")
    print(f"  test_mosaic.tif              — 13-band SLIC mosaic composite")
    print(f"  detection_mask.tif           — binary detection")
    print(f"  detection_probabilities.tif  — sigmoid probabilities")
    print(f"  quickview_e2e.png            — 3-panel: RGB | truth | detection")

    if ground_truth is not None and dice > 0.5:
        print(f"\n[PASS] Dice {dice:.4f} — strong agreement with ground truth")
    elif pos_rate > 0 and pos_rate < 50:
        print(f"\n[PASS] Pipeline produces plausible detections ({pos_rate:.1f}%)")
    elif pos_rate == 0:
        print("\n[WARN] No detections — inspect quickview")
    else:
        print(f"\n[WARN] High detection rate ({pos_rate:.1f}%) — inspect quickview")


if __name__ == "__main__":
    main()
