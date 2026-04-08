#!/usr/bin/env python
"""Batch detection test: 10 positive + 10 hard-negative TEST chips.

Downloads multi-temporal L1C+SCL stacks from CDSE OpenEO, applies the
SLIC temporal mosaic, runs U-Net inference, and generates 3-panel figures:
    Panel 1: RGB (B4/B3/B2)
    Panel 2: Ground truth binary mask (white = solar, black = background)
    Panel 3: Detection binary mask (white = detected, black = background)

Usage:
    python openeo_udp/tests/test_detection_batch.py [--skip-download] [--skip-mosaic]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
OUTPUT_DIR = REPO / "openeo_udp" / "tests" / "detection_batch_outputs"
MANIFEST_PATH = REPO / "outputs/training_prep/stage1_v3/split_manifest.csv"
NEGATIVES_PATH = REPO / "outputs/stage1/negatives/unique_chip_manifest.csv"
H5_PATH = REPO / "outputs/stage1/stage1_positives.h5"
BAND_STATS_PATH = REPO / "outputs/training_prep/stage1_v3/band_stats.npz"

CELL_SIZE = 2560  # metres per chip in EPSG:3857
TEST_TEMPORAL = ["2024-05-01", "2024-07-31"]

S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


def chip_to_bbox_4326(col: int, row: int) -> dict:
    """Convert chip col/row to EPSG:4326 bounding box."""
    xmin = col * CELL_SIZE
    ymin = row * CELL_SIZE
    xmax = xmin + CELL_SIZE
    ymax = ymin + CELL_SIZE

    def m_to_lon(x):
        return x / 20037508.34 * 180.0

    def m_to_lat(y):
        return math.degrees(2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)

    return {
        "west": m_to_lon(xmin),
        "south": m_to_lat(ymin),
        "east": m_to_lon(xmax),
        "north": m_to_lat(ymax),
        "crs": "EPSG:4326",
    }


def select_positive_chips(n: int = 10, seed: int = 42) -> list[dict]:
    """Select top-N TEST chips by positive pixel count."""
    import h5py

    # Load test split indices
    test_indices = {}
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                test_indices[row["chip_id"]] = {
                    "index": int(row["index"]),
                    "col": int(row["col"]),
                    "row": int(row["row"]),
                }

    # Count positive pixels per test chip
    chips_with_counts = []
    with h5py.File(H5_PATH, "r") as h5:
        chip_ids = [c.decode() for c in h5["chip_ids"][:]]
        masks = h5["masks"]
        for cid, info in test_indices.items():
            idx = info["index"]
            if idx < len(chip_ids) and chip_ids[idx] == cid:
                pos = int(np.sum(masks[idx] > 0))
                chips_with_counts.append({**info, "chip_id": cid, "pos_pixels": pos})

    # Sort by positive pixels descending, take top N
    chips_with_counts.sort(key=lambda c: c["pos_pixels"], reverse=True)
    selected = chips_with_counts[:n]

    result = []
    for c in selected:
        bbox = chip_to_bbox_4326(c["col"], c["row"])
        result.append({
            "chip_id": c["chip_id"],
            "col": c["col"],
            "row": c["row"],
            "index": c["index"],
            "pos_pixels": c["pos_pixels"],
            "bbox": bbox,
            "category": "positive",
        })
    return result


def select_negative_chips(n: int = 10, seed: int = 42) -> list[dict]:
    """Select N European hard-negative chips (35-55 lat, confirmed 0 panels)."""
    negatives = []
    with open(NEGATIVES_PATH) as f:
        for row in csv.DictReader(f):
            if row["continent"] == "Europe":
                lat = float(row["chip_center_lat"])
                if 35 <= lat <= 55:
                    negatives.append(row)

    random.seed(seed)
    selected = random.sample(negatives, min(n, len(negatives)))

    result = []
    for c in selected:
        col, row = int(c["chip_col"]), int(c["chip_row"])
        bbox = chip_to_bbox_4326(col, row)
        result.append({
            "chip_id": c["chip_id_str"],
            "col": col,
            "row": row,
            "index": -1,
            "pos_pixels": 0,
            "bbox": bbox,
            "category": "negative",
        })
    return result


def download_chip_stack(conn, chip: dict, output_dir: Path) -> Path:
    """Download multi-temporal L1C + SCL stack at 10m EPSG:3857."""
    nc_path = output_dir / f"{chip['chip_id']}_stack.nc"
    if nc_path.exists():
        print(f"      Already downloaded: {nc_path.name}")
        return nc_path

    bbox = chip["bbox"]

    s2_l1c = conn.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=bbox,
        temporal_extent=TEST_TEMPORAL,
        bands=S2_L1C_BANDS,
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    s2_scl = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=TEST_TEMPORAL,
        bands=["SCL"],
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    merged = s2_l1c.merge_cubes(s2_scl)

    job = merged.create_job(
        title=f"det_batch_{chip['chip_id']}",
        out_format="netCDF",
    )
    job.start_job()
    print(f"      Job ID: {job.job_id}")

    t0 = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - t0
        if elapsed > 10 or status != "queued":
            print(f"      [{elapsed:5.0f}s] {status}")
        if status == "finished":
            break
        elif status in ("error", "canceled"):
            try:
                for log in job.logs()[-5:]:
                    msg = log.get("message", str(log)) if isinstance(log, dict) else str(log)
                    print(f"      LOG: {msg}")
            except Exception:
                pass
            raise RuntimeError(f"Job failed: {status}")
        time.sleep(15)

    tmp_dir = output_dir / f"_tmp_{chip['chip_id']}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job.get_results().download_files(tmp_dir)

    nc_files = list(tmp_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No netCDF for {chip['chip_id']}")
    nc_files[0].rename(nc_path)
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"      Downloaded: {nc_path.name} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    return nc_path


def load_stack(nc_path: Path):
    """Load multi-temporal stack from netCDF → (T, 13, H, W) + (T, H, W) + dates."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    band_vars = [v for v in ds.data_vars if v in S2_L1C_BANDS or v == "SCL"]

    if len(band_vars) >= 14:
        spectral_bands = [ds[b].values for b in S2_L1C_BANDS if b in ds]
        spectral = np.stack(spectral_bands, axis=1).astype(np.float32)
        scl = ds["SCL"].values.astype(np.int32)

        t_dim = next((d for d in ds[S2_L1C_BANDS[0]].dims if d in ("t", "time")), None)
        if t_dim and t_dim in ds.coords:
            dates = [str(np.datetime_as_string(t, unit="D")) for t in ds.coords[t_dim].values]
        else:
            dates = [f"scene_{i}" for i in range(spectral.shape[0])]
    else:
        raise ValueError(f"Expected >= 14 band variables, got {len(band_vars)}: {band_vars}")

    ds.close()
    return spectral, scl, dates


def load_ground_truth(chip: dict) -> np.ndarray | None:
    """Load ground truth mask for a positive chip from HDF5."""
    if chip["category"] != "positive" or chip["index"] < 0:
        return None

    import h5py
    with h5py.File(H5_PATH, "r") as h5:
        mask = h5["masks"][chip["index"]].astype(np.float32)
    return mask


def run_inference(image_hwc: np.ndarray, threshold: float = 0.80):
    """Run U-Net inference on a mosaic chip. Returns (binary, probs)."""
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore

    model, band_stats, registry = _get_model_and_stats()
    threshold = float(registry.get("threshold", threshold))

    h, w, c = image_hwc.shape
    all_probs = np.zeros((h, w), dtype=np.float32)

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

    binary = (all_probs > threshold).astype(np.uint8)
    return binary, all_probs, threshold


def make_rgb(data_chw: np.ndarray) -> np.ndarray:
    """(C, H, W) L1C → (H, W, 3) RGB uint8, percentile-stretched."""
    rgb = np.transpose(data_chw[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
    valid = rgb[rgb > 0]
    if valid.size > 0:
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
    else:
        lo, hi = 0, 1
    return np.clip((rgb - lo) / max(hi - lo, 1), 0, 1)


def generate_figure(
    chip: dict,
    rgb: np.ndarray,
    ground_truth: np.ndarray | None,
    detection: np.ndarray,
    dice: float | None,
    output_path: Path,
):
    """3-panel figure: RGB | Ground Truth mask | Detection mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (B4/B3/B2)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Ground truth binary mask
    if ground_truth is not None:
        gt_display = (ground_truth > 0.5).astype(np.float32)
        # Resize if needed
        gh, gw = gt_display.shape
        dh, dw = detection.shape
        if gh != dh or gw != dw:
            from scipy.ndimage import zoom
            gt_display = zoom(gt_display, (dh / gh, dw / gw), order=0)
        axes[1].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth ({int(np.sum(ground_truth > 0.5))} px)", fontsize=11)
    else:
        # No ground truth — show blank (confirmed negative)
        axes[1].imshow(np.zeros_like(detection, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth (0 px — hard negative)", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Detection binary mask
    det_display = detection.astype(np.float32)
    axes[2].imshow(det_display, cmap="gray", vmin=0, vmax=1)
    det_px = int(np.sum(detection > 0))
    title = f"Detection ({det_px} px)"
    if dice is not None:
        title += f" — Dice: {dice:.3f}"
    axes[2].set_title(title, fontsize=11)
    axes[2].axis("off")

    category = chip["category"].upper()
    fig.suptitle(
        f"{chip['chip_id']}  [{category}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    gt_bin = (gt > 0.5).astype(np.float32).ravel()
    pred_bin = (pred > 0.5).astype(np.float32).ravel()
    intersection = np.sum(gt_bin * pred_bin)
    denom = np.sum(gt_bin) + np.sum(pred_bin)
    if denom == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0
    return float(2.0 * intersection / denom)


def process_chip(chip: dict, output_dir: Path) -> dict:
    """Full pipeline for one chip: load stack → mosaic → inference → figure."""
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic

    chip_id = chip["chip_id"]
    nc_path = output_dir / f"{chip_id}_stack.nc"

    if not nc_path.exists():
        print(f"      SKIPPED (no data file)")
        return {"chip_id": chip_id, "status": "skipped"}

    # Load stack
    spectral, scl, dates = load_stack(nc_path)
    t_scenes, n_bands, h, w = spectral.shape
    print(f"      Stack: {t_scenes} scenes × {n_bands} bands × {h}×{w}")

    # Mosaic
    t0 = time.time()
    composite, info = create_temporal_mosaic(spectral, scl)
    mosaic_time = time.time() - t0
    print(f"      Mosaic: {info['n_scenes_used']} scenes, "
          f"{info['fill_mode'].sum() / info['fill_mode'].size * 100:.1f}% rescue, "
          f"{mosaic_time:.1f}s")

    # Prepare for inference
    image_hwc = np.transpose(composite[:13], (1, 2, 0))  # (H, W, 13)

    # Inference
    binary, probs, threshold = run_inference(image_hwc)
    det_px = int(np.sum(binary))
    print(f"      Detection: {det_px} px (threshold={threshold})")

    # Ground truth
    gt = load_ground_truth(chip)
    dice = None
    if gt is not None:
        # Resize if needed
        dh, dw = binary.shape
        gh, gw = gt.shape
        if gh != dh or gw != dw:
            from scipy.ndimage import zoom
            gt_resized = zoom(gt, (dh / gh, dw / gw), order=0)
        else:
            gt_resized = gt
        dice = compute_dice(gt_resized, binary)
        print(f"      Dice: {dice:.4f} (GT: {int(np.sum(gt > 0.5))} px)")
    else:
        # Negative chip — false positive rate
        fp_rate = det_px / binary.size * 100
        print(f"      False positive rate: {fp_rate:.2f}%")

    # Generate figure
    rgb = make_rgb(composite)
    fig_path = output_dir / f"detection_{chip_id}.png"
    generate_figure(chip, rgb, gt, binary, dice, fig_path)
    print(f"      Figure: {fig_path.name}")

    return {
        "chip_id": chip_id,
        "category": chip["category"],
        "status": "ok",
        "det_px": det_px,
        "gt_px": int(np.sum(gt > 0.5)) if gt is not None else 0,
        "dice": dice,
        "fp_rate": det_px / binary.size * 100 if chip["category"] == "negative" else None,
        "mosaic_scenes": info["n_scenes_used"],
        "rescue_pct": info["fill_mode"].sum() / info["fill_mode"].size * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OpenEO download, use existing .nc files")
    parser.add_argument("--skip-mosaic", action="store_true",
                        help="Skip download + mosaic, only re-run inference on cached mosaics")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Solar PV Detection — Batch Test (10 positive + 10 negative)")
    print("=" * 70)

    # Select chips
    pos_chips = select_positive_chips(10)
    neg_chips = select_negative_chips(10)
    all_chips = pos_chips + neg_chips

    print(f"\n10 POSITIVE test chips (highest solar panel coverage):")
    for c in pos_chips:
        print(f"   {c['chip_id']:25s} {c['pos_pixels']:6d} px")

    print(f"\n10 NEGATIVE test chips (confirmed 0 panels, Europe 35-55°N):")
    for c in neg_chips:
        bbox = c["bbox"]
        print(f"   {c['chip_id']:25s} lat={bbox['south']:.1f} lon={bbox['west']:.1f}")

    # Download
    if not args.skip_download and not args.skip_mosaic:
        import openeo

        print(f"\nConnecting to {BACKEND_URL}...")
        conn = openeo.connect(BACKEND_URL)
        conn.authenticate_oidc_device()
        user = conn.describe_account().get("user_id", "unknown")
        print(f"Authenticated as: {user}")

        print(f"\nDownloading {len(all_chips)} multi-temporal stacks...")
        for i, chip in enumerate(all_chips):
            print(f"\n   [{i + 1}/{len(all_chips)}] {chip['chip_id']} ({chip['category']})")
            try:
                download_chip_stack(conn, chip, OUTPUT_DIR)
            except Exception as e:
                print(f"      [ERROR] {e}")

    # Process each chip
    print(f"\n{'=' * 70}")
    print("Processing chips: mosaic → inference → figures")
    print(f"{'=' * 70}")

    results = []
    for i, chip in enumerate(all_chips):
        print(f"\n[{i + 1}/{len(all_chips)}] {chip['chip_id']} ({chip['category']})")
        try:
            result = process_chip(chip, OUTPUT_DIR)
            results.append(result)
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results.append({"chip_id": chip["chip_id"], "status": "error", "error": str(e)})

    # Summary table
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    pos_results = [r for r in results if r.get("category") == "positive" and r.get("status") == "ok"]
    neg_results = [r for r in results if r.get("category") == "negative" and r.get("status") == "ok"]

    if pos_results:
        print(f"\nPOSITIVE chips (Dice score):")
        print(f"{'Chip ID':>28s}  {'GT px':>7s}  {'Det px':>7s}  {'Dice':>6s}")
        print("-" * 55)
        for r in pos_results:
            d = f"{r['dice']:.3f}" if r['dice'] is not None else "N/A"
            print(f"{r['chip_id']:>28s}  {r['gt_px']:7d}  {r['det_px']:7d}  {d:>6s}")
        dices = [r["dice"] for r in pos_results if r["dice"] is not None]
        if dices:
            print(f"{'Mean Dice':>28s}  {'':>7s}  {'':>7s}  {np.mean(dices):.3f}")

    if neg_results:
        print(f"\nNEGATIVE chips (false positive rate):")
        print(f"{'Chip ID':>28s}  {'Det px':>7s}  {'FP rate':>8s}")
        print("-" * 50)
        for r in neg_results:
            fp = f"{r['fp_rate']:.2f}%" if r["fp_rate"] is not None else "N/A"
            print(f"{r['chip_id']:>28s}  {r['det_px']:7d}  {fp:>8s}")
        fp_rates = [r["fp_rate"] for r in neg_results if r["fp_rate"] is not None]
        if fp_rates:
            print(f"{'Mean FP rate':>28s}  {'':>7s}  {np.mean(fp_rates):.2f}%")

    print(f"\nFigures saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
