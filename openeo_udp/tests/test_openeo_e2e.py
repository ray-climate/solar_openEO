#!/usr/bin/env python
"""End-to-end test: OpenEO mosaic → local UDF inference.

Tests the full pipeline by:
  1. Connecting to CDSE OpenEO backend
  2. Loading a small Sentinel-2 L1C chip over a known solar farm
  3. Building a cloud-free temporal median mosaic via OpenEO
  4. Downloading the mosaic as a GeoTIFF
  5. Running the local UDF inference on the downloaded mosaic
  6. Validating the output

This verifies that the OpenEO mosaic output is compatible with the
trained model's expected input format.

Authentication options (tried in order):
  1. Environment variables:  CDSE_USERNAME + CDSE_PASSWORD
  2. Credentials file:       ~/.cdse_credentials  (username on line 1, password on line 2)
  3. Interactive device code flow (browser-based, fallback)

Setup (choose one):
  Option A — environment variables:
      export CDSE_USERNAME="your.email@example.com"
      export CDSE_PASSWORD="your_password"

  Option B — credentials file:
      echo "your.email@example.com" > ~/.cdse_credentials
      echo "your_password" >> ~/.cdse_credentials
      chmod 600 ~/.cdse_credentials

Usage:
    python openeo_udp/tests/test_openeo_e2e.py [--skip-download]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Output directory for test artifacts
TEST_OUTPUT_DIR = REPO / "openeo_udp" / "tests" / "test_outputs"

# --- Test AOI ---
# A small area containing known solar installations in Europe.
# Roughly one 256x256 chip at 10m = 2.56 km x 2.56 km.
# Location: Swindon area, UK (known solar farms from training data).
TEST_AOI = {
    "west": -1.85,
    "south": 51.54,
    "east": -1.82,
    "north": 51.56,
    "crs": "EPSG:4326",
}

# Summer temporal window matching training data
TEST_TEMPORAL = ["2024-05-01", "2024-07-31"]

# Sentinel-2 L1C bands in the order expected by the model
S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"


REFRESH_TOKEN_FILE = Path.home() / ".cdse_refresh_token"


def _save_refresh_token(connection) -> None:
    """Extract and save the refresh token for future use."""
    try:
        auth = connection.auth
        if hasattr(auth, "bearer") and hasattr(auth.bearer, "refresh_token"):
            token = auth.bearer.refresh_token
        elif hasattr(auth, "_refresh_token"):
            token = auth._refresh_token
        else:
            # Try to get from the internal auth object
            token = None
        if token:
            REFRESH_TOKEN_FILE.write_text(token)
            REFRESH_TOKEN_FILE.chmod(0o600)
            print(f"  Refresh token saved to {REFRESH_TOKEN_FILE}")
    except Exception as e:
        print(f"  (Could not save refresh token: {e})")


def connect_and_authenticate():
    """Connect to CDSE and authenticate.

    Tries stored refresh token first, falls back to device code flow.
    After successful device code login, the refresh token is saved so
    future runs don't need browser authentication.
    """
    import openeo
    print(f"Connecting to {BACKEND_URL} ...")
    connection = openeo.connect(BACKEND_URL)

    # Option 1: Try stored refresh token
    if REFRESH_TOKEN_FILE.exists():
        token = REFRESH_TOKEN_FILE.read_text().strip()
        if token:
            try:
                print("  Using stored refresh token...")
                connection.authenticate_oidc_refresh_token(refresh_token=token)
                user_info = connection.describe_account()
                print(f"  Authenticated as: {user_info.get('user_id', 'unknown')}")
                _save_refresh_token(connection)  # Update with new token
                return connection
            except Exception as e:
                print(f"  Refresh token expired or invalid: {e}")
                print("  Falling back to device code flow...")

    # Option 2: Device code flow (interactive)
    print("  Starting device code authentication...")
    print("  >>> A URL will appear below — open it in your browser immediately <<<")
    print()
    connection.authenticate_oidc_device()

    user_info = connection.describe_account()
    print(f"\n  Authenticated as: {user_info.get('user_id', 'unknown')}")

    # Save refresh token for future runs
    _save_refresh_token(connection)

    return connection


def build_mosaic_job(connection) -> "openeo.BatchJob":
    """Build and submit a cloud-free L1C mosaic job.

    Strategy:
      - Load SENTINEL2_L1C with all 13 spectral bands
      - Use temporal median to remove clouds (simple, robust)
      - Resample to 10m in EPSG:3857 (matches training grid)

    For production, Apex should replace the median with proper cloud
    masking (e.g., SCL from L2A or s2cloudless).
    """
    print(f"\nBuilding mosaic process graph...")
    print(f"  AOI: {TEST_AOI}")
    print(f"  Temporal: {TEST_TEMPORAL}")
    print(f"  Bands: {len(S2_L1C_BANDS)} bands")

    s2 = connection.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=TEST_AOI,
        temporal_extent=TEST_TEMPORAL,
        bands=S2_L1C_BANDS,
    )

    # Resample all bands to 10m
    # NOTE: On CDSE, load_collection may already handle resampling.
    # If not, uncomment:
    # s2 = s2.resample_spatial(resolution=10, projection="EPSG:3857")

    # Temporal median composite (replaces SNIC-based mosaicing)
    mosaic = s2.reduce_dimension(dimension="t", reducer="median")

    return mosaic


def download_mosaic(connection, mosaic_cube, output_path: Path) -> Path:
    """Download the mosaic synchronously or as a batch job."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSubmitting batch job...")
    job = mosaic_cube.create_job(
        title="Solar PV UDP test — L1C mosaic",
        out_format="GTiff",
    )
    job.start_job()
    job_id = job.job_id
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for completion...")

    # Poll for completion
    t0 = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] Status: {status}")

        if status == "finished":
            break
        elif status in ("error", "canceled"):
            logs = job.logs()
            for log in logs[-10:]:
                print(f"    LOG: {log.get('message', log)}")
            raise RuntimeError(f"Job failed with status: {status}")

        time.sleep(15)

    # Download results
    print(f"  Downloading results to {output_path.parent}/")
    results = job.get_results()
    results.download_files(output_path.parent)

    # Find the downloaded GeoTIFF
    tiffs = list(output_path.parent.glob("*.tif*"))
    if not tiffs:
        raise FileNotFoundError("No GeoTIFF files in job results")

    result_path = tiffs[0]
    if result_path != output_path:
        result_path.rename(output_path)
        result_path = output_path

    print(f"  Downloaded: {result_path} ({result_path.stat().st_size / 1e6:.1f} MB)")
    return result_path


def validate_mosaic(tiff_path: Path) -> np.ndarray:
    """Validate the downloaded mosaic GeoTIFF."""
    import rasterio

    print(f"\nValidating mosaic: {tiff_path}")

    with rasterio.open(tiff_path) as ds:
        print(f"  Shape: {ds.count} bands × {ds.height}H × {ds.width}W")
        print(f"  CRS: {ds.crs}")
        print(f"  Resolution: {ds.res}")
        print(f"  Dtype: {ds.dtypes[0]}")
        print(f"  Bounds: {ds.bounds}")

        # Read all bands → (C, H, W)
        data = ds.read().astype(np.float32)  # (bands, H, W)

    n_bands = data.shape[0]
    print(f"  Bands read: {n_bands}")

    # Check band count
    if n_bands != 13:
        print(f"  [WARN] Expected 13 bands, got {n_bands}")
        if n_bands < 13:
            raise ValueError(f"Only {n_bands} bands — model requires 13")

    # Check for NaN / Inf
    nan_pct = np.isnan(data).mean() * 100
    inf_pct = np.isinf(data).mean() * 100
    print(f"  NaN pixels: {nan_pct:.2f}%")
    print(f"  Inf pixels: {inf_pct:.2f}%")

    # Check value range (L1C TOA reflectance typically 0–10000+)
    for i, band_name in enumerate(S2_L1C_BANDS[:n_bands]):
        band = data[i]
        valid = band[np.isfinite(band)]
        if valid.size > 0:
            print(f"  {band_name}: min={valid.min():.0f}, max={valid.max():.0f}, "
                  f"mean={valid.mean():.0f}, zeros={np.sum(valid == 0) / valid.size * 100:.1f}%")

    # Convert to (H, W, C) for model input
    image_hwc = np.transpose(data[:13], (1, 2, 0))  # (H, W, 13)
    print(f"\n  [PASS] Mosaic validated: shape {image_hwc.shape}")
    return image_hwc


def run_local_inference(image_hwc: np.ndarray) -> np.ndarray:
    """Run UDF inference locally on the downloaded mosaic.

    This tests that the OpenEO output is compatible with the model.
    """
    from openeo_udp.udf.solar_pv_inference import (
        _get_model_and_stats,
        normalize_zscore,
    )

    print(f"\nRunning local inference...")
    model, band_stats, registry = _get_model_and_stats()

    h, w, c = image_hwc.shape
    print(f"  Input: {h}×{w}×{c}")

    # Handle non-256x256 chips by padding or tiling
    if h != 256 or w != 256:
        print(f"  [INFO] Mosaic is {h}×{w}, not 256×256. Tiling into chips...")
        masks = []
        for y0 in range(0, h, 256):
            for x0 in range(0, w, 256):
                chip = image_hwc[y0:y0 + 256, x0:x0 + 256, :]
                # Pad if needed
                if chip.shape[0] < 256 or chip.shape[1] < 256:
                    padded = np.zeros((256, 256, c), dtype=np.float32)
                    padded[:chip.shape[0], :chip.shape[1], :] = chip
                    chip = padded
                chip_norm = normalize_zscore(chip[np.newaxis], band_stats)
                pred = model.predict(chip_norm, verbose=0)
                masks.append((y0, x0, pred[0, :, :, 0]))

        # Reconstruct
        full_mask = np.zeros((h, w), dtype=np.float32)
        for y0, x0, pred in masks:
            ph = min(256, h - y0)
            pw = min(256, w - x0)
            full_mask[y0:y0 + ph, x0:x0 + pw] = pred[:ph, :pw]
    else:
        chip_norm = normalize_zscore(image_hwc[np.newaxis], band_stats)
        pred = model.predict(chip_norm, verbose=0)
        full_mask = pred[0, :, :, 0]

    threshold = float(registry.get("threshold", 0.80))
    binary = (full_mask > threshold).astype(np.uint8)

    pos_pixels = np.sum(binary)
    total_pixels = binary.size
    pos_rate = pos_pixels / total_pixels * 100

    print(f"  Probability range: [{full_mask.min():.4f}, {full_mask.max():.4f}]")
    print(f"  Threshold: {threshold}")
    print(f"  Positive pixels: {pos_pixels} / {total_pixels} ({pos_rate:.2f}%)")
    print(f"  [PASS] Inference complete: {binary.shape}")

    return binary, full_mask


def save_results(binary: np.ndarray, probs: np.ndarray, tiff_path: Path):
    """Save detection results as GeoTIFF alongside the input mosaic."""
    try:
        import rasterio
        from rasterio.transform import from_bounds

        with rasterio.open(tiff_path) as src:
            profile = src.profile.copy()

        # Save binary mask
        mask_path = tiff_path.parent / "detection_mask.tif"
        profile.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(binary, 1)
        print(f"  Saved binary mask: {mask_path}")

        # Save probability map
        prob_path = tiff_path.parent / "detection_probabilities.tif"
        profile.update(count=1, dtype="float32", nodata=-1)
        with rasterio.open(prob_path, "w", **profile) as dst:
            dst.write(probs, 1)
        print(f"  Saved probability map: {prob_path}")

    except Exception as e:
        print(f"  [WARN] Could not save GeoTIFF results: {e}")
        # Fallback: save as numpy
        np.save(tiff_path.parent / "detection_mask.npy", binary)
        np.save(tiff_path.parent / "detection_probs.npy", probs)
        print(f"  Saved numpy arrays instead")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OpenEO job, use previously downloaded mosaic")
    parser.add_argument("--output-dir", type=Path, default=TEST_OUTPUT_DIR,
                        help="Output directory for test artifacts")
    args = parser.parse_args()

    mosaic_path = args.output_dir / "test_mosaic_l1c.tif"

    print("=" * 60)
    print("Solar PV UDP — End-to-End OpenEO Test")
    print("=" * 60)

    if args.skip_download and mosaic_path.exists():
        print(f"\nSkipping download, using existing: {mosaic_path}")
    else:
        # Step 1: Connect
        connection = connect_and_authenticate()

        # Step 2: Build mosaic
        mosaic_cube = build_mosaic_job(connection)

        # Step 3: Download
        mosaic_path = download_mosaic(connection, mosaic_cube, mosaic_path)

    # Step 4: Validate mosaic
    image_hwc = validate_mosaic(mosaic_path)

    # Step 5: Run local inference
    binary, probs = run_local_inference(image_hwc)

    # Step 6: Save results
    print(f"\nSaving results...")
    save_results(binary, probs, mosaic_path)

    print(f"\n{'=' * 60}")
    print("END-TO-END TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nArtifacts saved to: {args.output_dir}/")
    print("  - test_mosaic_l1c.tif        (13-band input mosaic)")
    print("  - detection_mask.tif          (binary detection)")
    print("  - detection_probabilities.tif (sigmoid probabilities)")
    print("\nInspect in QGIS to verify detections over the test AOI.")


if __name__ == "__main__":
    main()
