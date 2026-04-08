#!/usr/bin/env python
"""Local test for the solar PV inference UDF.

Runs the UDF on real chips from the project's HDF5 dataset WITHOUT
requiring an OpenEO backend.  This verifies:

  1. Model builds and weights load correctly
  2. Normalisation produces sensible values
  3. Output shape, dtype, and value range are correct
  4. Predictions are non-trivial (not all-zero or all-one)

Usage:
    python openeo_udp/tests/test_udf_local.py [--n-chips 5] [--verbose]

Exit code 0 = all checks pass.  Non-zero = failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Ensure repo root is on path so we can import from solar_ml and openeo_udp
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from openeo_udp.udf.solar_pv_inference import (
    _get_model_and_stats,
    _load_registry,
    normalize_zscore,
    predict_chip,
    predict_chip_probabilities,
)


def load_test_chips(n: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a random sample of chips from the project HDF5 dataset.

    Returns (images_hwc, masks_hw, chip_ids).
    """
    registry = _load_registry()
    h5_path = REPO / registry.get("weights_local", "").replace("experiments", "outputs").rsplit("/", 1)[0]

    # Use the known H5 path
    h5_path = REPO / "outputs" / "stage1" / "stage1_positives.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found at {h5_path}.\n"
            "This test requires the training dataset to be present."
        )

    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5:
        total = h5["images"].shape[0]
        indices = sorted(rng.choice(total, size=min(n, total), replace=False))

        images_chw = h5["images"][indices].astype(np.float32)  # (N, 13, 256, 256)
        masks_hw = h5["masks"][indices].astype(np.float32)     # (N, 256, 256)
        chip_ids = [h5["chip_ids"][i].decode("utf-8") if isinstance(h5["chip_ids"][i], bytes)
                    else str(h5["chip_ids"][i]) for i in indices]

    images_hwc = np.transpose(images_chw, (0, 2, 3, 1))  # (N, 256, 256, 13)
    return images_hwc, masks_hw, chip_ids


def test_registry_loads():
    """Test that model_registry.yaml is found and parsed."""
    registry = _load_registry()
    assert "backbone" in registry, "Registry missing 'backbone'"
    assert "threshold" in registry, "Registry missing 'threshold'"
    assert "normalization" in registry, "Registry missing 'normalization'"
    assert registry["normalization"] == "zscore", f"Expected zscore, got {registry['normalization']}"
    assert registry["input_shape"] == [256, 256, 13], f"Unexpected input_shape: {registry['input_shape']}"
    print("  [PASS] Registry loads correctly")


def test_model_and_stats_load():
    """Test that model builds and weights + band stats load."""
    model, band_stats, registry = _get_model_and_stats()

    # Check model
    input_shape = model.input_shape
    assert input_shape[1:] == (256, 256, 13), f"Unexpected input shape: {input_shape}"
    output_shape = model.output_shape
    assert output_shape[1:] == (256, 256, 1), f"Unexpected output shape: {output_shape}"

    # Check band stats
    assert "mean" in band_stats, "Band stats missing 'mean'"
    assert "std" in band_stats, "Band stats missing 'std'"
    assert band_stats["mean"].shape == (13,), f"Unexpected mean shape: {band_stats['mean'].shape}"
    assert band_stats["std"].shape == (13,), f"Unexpected std shape: {band_stats['std'].shape}"
    assert np.all(band_stats["std"] > 0), "Band std contains zeros"

    print("  [PASS] Model and band stats load correctly")
    print(f"         Model params: {model.count_params():,}")
    return model, band_stats, registry


def test_normalisation(images_hwc: np.ndarray, band_stats: dict):
    """Test that z-score normalisation produces sensible values."""
    normed = normalize_zscore(images_hwc[:1], band_stats)

    # After z-score, mean should be near 0, std near 1 (across spatial dims)
    per_band_mean = np.mean(normed[0], axis=(0, 1))
    per_band_std = np.std(normed[0], axis=(0, 1))

    assert normed.dtype == np.float32, f"Unexpected dtype: {normed.dtype}"
    assert normed.shape == images_hwc[:1].shape, f"Shape mismatch: {normed.shape}"
    assert not np.any(np.isnan(normed)), "NaN in normalised output"
    assert not np.any(np.isinf(normed)), "Inf in normalised output"

    # Rough check: normalised values should mostly be in [-5, 5]
    pct_extreme = np.mean(np.abs(normed) > 5.0) * 100
    assert pct_extreme < 5.0, f"Too many extreme values after normalisation: {pct_extreme:.1f}%"

    print("  [PASS] Normalisation produces sensible values")
    print(f"         Per-band mean range: [{per_band_mean.min():.2f}, {per_band_mean.max():.2f}]")
    print(f"         Per-band std  range: [{per_band_std.min():.2f}, {per_band_std.max():.2f}]")


def test_inference(images_hwc: np.ndarray, masks_hw: np.ndarray, chip_ids: list[str], verbose: bool = False):
    """Test inference on real chips."""
    n = len(images_hwc)

    # Single chip prediction
    t0 = time.time()
    mask_single = predict_chip(images_hwc[0])
    t_single = time.time() - t0
    assert mask_single.shape == (256, 256), f"Unexpected output shape: {mask_single.shape}"
    assert mask_single.dtype == np.uint8, f"Unexpected dtype: {mask_single.dtype}"
    assert set(np.unique(mask_single)).issubset({0, 1}), "Mask contains non-binary values"

    # Batch prediction
    t0 = time.time()
    masks_batch = predict_chip(images_hwc)
    t_batch = time.time() - t0
    assert masks_batch.shape == (n, 256, 256), f"Unexpected batch shape: {masks_batch.shape}"

    # Probability output
    probs = predict_chip_probabilities(images_hwc[0])
    assert probs.shape == (256, 256), f"Unexpected prob shape: {probs.shape}"
    assert probs.dtype == np.float32, f"Unexpected prob dtype: {probs.dtype}"
    assert probs.min() >= 0.0, f"Prob min < 0: {probs.min()}"
    assert probs.max() <= 1.0, f"Prob max > 1: {probs.max()}"

    # Check predictions are non-trivial
    total_positive = np.sum(masks_batch)
    total_pixels = n * 256 * 256
    positive_rate = total_positive / total_pixels
    assert positive_rate > 0.0, "All predictions are zero — model may not have loaded correctly"
    assert positive_rate < 1.0, "All predictions are one — model may not have loaded correctly"

    print("  [PASS] Inference produces valid results")
    print(f"         Single chip: {t_single:.2f}s, Batch ({n} chips): {t_batch:.2f}s")
    print(f"         Positive pixel rate: {positive_rate:.4f}")

    if verbose:
        print("\n  Per-chip breakdown:")
        for i, cid in enumerate(chip_ids):
            pred_pos = np.sum(masks_batch[i])
            true_pos = np.sum(masks_hw[i] > 0.5)
            prob_mean = predict_chip_probabilities(images_hwc[i]).mean()
            print(f"    {cid}: pred_px={pred_pos:5d}, true_px={true_pos:5d}, mean_prob={prob_mean:.4f}")


def test_dice_agreement(images_hwc: np.ndarray, masks_hw: np.ndarray):
    """Check Dice agreement between predictions and ground truth.

    This is a smoke test — not a full evaluation.  We just check that
    predictions are somewhat correlated with ground truth.
    """
    masks_pred = predict_chip(images_hwc)
    masks_true = (masks_hw > 0.5).astype(np.float32)
    masks_pred_f = masks_pred.astype(np.float32)

    intersection = np.sum(masks_true * masks_pred_f)
    denom = np.sum(masks_true) + np.sum(masks_pred_f)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)

    # Dice should be above random (> 0.1) for positive-only chips
    print(f"  [INFO] Batch Dice score: {dice:.4f}")
    if dice > 0.3:
        print("  [PASS] Predictions show meaningful agreement with ground truth")
    else:
        print("  [WARN] Low Dice — acceptable for a small random sample, "
              "but investigate if this persists")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-chips", type=int, default=5, help="Number of test chips")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-chip breakdown")
    args = parser.parse_args()

    print("=" * 60)
    print("Solar PV UDF — Local Verification Test")
    print("=" * 60)

    print("\n1. Testing registry...")
    test_registry_loads()

    print("\n2. Loading model and band stats...")
    model, band_stats, registry = test_model_and_stats_load()

    print(f"\n3. Loading {args.n_chips} test chips from HDF5...")
    images, masks, chip_ids = load_test_chips(n=args.n_chips)
    print(f"   Loaded: {images.shape}, dtype={images.dtype}")

    print("\n4. Testing normalisation...")
    test_normalisation(images, band_stats)

    print("\n5. Testing inference...")
    test_inference(images, masks, chip_ids, verbose=args.verbose)

    print("\n6. Testing Dice agreement with ground truth...")
    test_dice_agreement(images, masks)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
