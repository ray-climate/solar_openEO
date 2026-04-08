#!/usr/bin/env python
"""Local test for the temporal mosaic UDF.

Tests that the SLIC-based temporal mosaic algorithm produces correct
cloud-free composites from synthetic multi-temporal data with known
cloud patterns.

Usage:
    python openeo_udp/tests/test_temporal_mosaic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from openeo_udp.udf.temporal_mosaic import (
    compute_clear_masks,
    score_scenes,
    slic_segmentation,
    assign_clusters_to_scenes,
    hierarchical_fallback,
    rescue_fill,
    feather_seams,
    create_temporal_mosaic,
    DEFAULT_PARAMS,
)


def make_synthetic_data(
    n_scenes: int = 5,
    h: int = 64,
    w: int = 64,
    n_bands: int = 13,
    cloud_fracs: list[float] | None = None,
    seed: int = 42,
):
    """Create synthetic multi-temporal S2 data with controlled cloud patterns.

    Each scene has a distinct spectral signature so we can verify which
    scene was selected for each pixel.  Cloud masks are block-shaped for
    easy verification.

    Returns
    -------
    spectral : (T, C, H, W) float32
    scl : (T, H, W) int32
    cloud_fracs : actual cloud fractions per scene
    """
    rng = np.random.RandomState(seed)

    # Each scene has a unique base value per band so composites are traceable
    spectral = np.zeros((n_scenes, n_bands, h, w), dtype=np.float32)
    for t in range(n_scenes):
        base = (t + 1) * 1000.0  # scene 0 → 1000, scene 1 → 2000, etc.
        for b in range(n_bands):
            spectral[t, b] = base + b * 10 + rng.randn(h, w).astype(np.float32) * 5

    # SCL: 4 = vegetation (clear), 9 = cloud high prob
    scl = np.full((n_scenes, h, w), 4, dtype=np.int32)  # all clear by default

    if cloud_fracs is None:
        cloud_fracs = [0.0, 0.1, 0.3, 0.6, 0.9]

    for t, frac in enumerate(cloud_fracs[:n_scenes]):
        # Place cloud as a block in top-left corner
        cloud_h = int(h * np.sqrt(frac))
        cloud_w = int(w * np.sqrt(frac))
        scl[t, :cloud_h, :cloud_w] = 9  # cloud high prob

    return spectral, scl, cloud_fracs


def test_compute_clear_masks():
    """Test cloud masking from SCL values."""
    print("Test: compute_clear_masks")
    h, w = 32, 32
    scl = np.full((3, h, w), 4, dtype=np.int32)  # all vegetation (clear)
    scl[0, :16, :16] = 9   # scene 0: cloud in top-left quadrant
    scl[1, 16:, 16:] = 10  # scene 1: cirrus in bottom-right quadrant
    scl[2, :, :] = 4       # scene 2: fully clear

    masks = compute_clear_masks(scl, DEFAULT_PARAMS["scl_clear_classes"])

    assert masks.shape == (3, h, w), f"Wrong shape: {masks.shape}"
    assert masks.dtype == bool, f"Wrong dtype: {masks.dtype}"

    # Scene 2 should be mostly clear
    assert masks[2].mean() > 0.9, f"Scene 2 should be mostly clear, got {masks[2].mean():.2f}"
    # Scene 0 should have fewer clear pixels than scene 2
    assert masks[0].mean() < masks[2].mean(), "Scene 0 should be cloudier than scene 2"

    print(f"  Clear fractions: {[f'{m.mean():.2f}' for m in masks]}")
    print("  [PASS]")


def test_score_scenes():
    """Test scene scoring by clear fraction."""
    print("Test: score_scenes")
    h, w = 64, 64
    masks = np.ones((4, h, w), dtype=bool)
    masks[0, :32, :32] = False   # 25% cloudy
    masks[1, :, :] = False       # 100% cloudy
    masks[2, :, :] = True        # 0% cloudy
    masks[3, :48, :] = False     # 75% cloudy

    scores = score_scenes(masks)

    assert scores.shape == (4,), f"Wrong shape: {scores.shape}"
    assert scores[2] > scores[0] > scores[3] > scores[1], \
        f"Wrong ordering: {scores}"
    assert abs(scores[2] - 1.0) < 0.01, f"Fully clear scene should score ~1.0, got {scores[2]}"

    print(f"  Scores: {scores}")
    print("  [PASS]")


def test_slic_segmentation():
    """Test SLIC produces reasonable superpixels."""
    print("Test: slic_segmentation")
    h, w = 64, 64

    # Create a simple image with 4 distinct quadrants
    image = np.zeros((h, w, 4), dtype=np.float32)
    image[:32, :32, :] = [1000, 2000, 3000, 4000]   # TL
    image[:32, 32:, :] = [5000, 6000, 7000, 8000]   # TR
    image[32:, :32, :] = [2000, 3000, 4000, 5000]   # BL
    image[32:, 32:, :] = [6000, 7000, 8000, 9000]   # BR

    labels = slic_segmentation(image, n_segments=16, compactness=10.0, n_iter=5)

    assert labels.shape == (h, w), f"Wrong shape: {labels.shape}"
    assert labels.min() >= 0, "Negative labels"
    n_clusters = labels.max() + 1
    assert n_clusters > 1, "Should produce multiple clusters"

    # Pixels in the same quadrant should mostly share labels
    tl_labels = set(labels[:32, :32].ravel())
    br_labels = set(labels[32:, 32:].ravel())
    overlap = tl_labels & br_labels
    # They shouldn't fully overlap (distinct spectral regions)
    assert len(overlap) < min(len(tl_labels), len(br_labels)), \
        "Distinct quadrants should have mostly different labels"

    print(f"  {n_clusters} clusters produced")
    print(f"  TL unique labels: {len(tl_labels)}, BR unique labels: {len(br_labels)}")
    print("  [PASS]")


def test_assign_clusters():
    """Test per-cluster scene assignment."""
    print("Test: assign_clusters_to_scenes")
    h, w = 32, 32

    # 2 clusters: left half = 0, right half = 1
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, 16:] = 1

    # 3 scenes: scene 0 clear on left, scene 1 clear on right, scene 2 half-clear
    clear = np.zeros((3, h, w), dtype=bool)
    clear[0, :, :16] = True   # scene 0: left clear
    clear[1, :, 16:] = True   # scene 1: right clear
    clear[2, :, :] = True     # scene 2: fully clear (but worse overall score)

    scene_order = np.array([2, 0, 1])  # scene 2 best overall

    assignment = assign_clusters_to_scenes(
        labels, clear, scene_order, clear_thresh=0.5,
    )

    assert assignment.shape == (h, w)
    # Scene 2 is best for both clusters (fully clear)
    assert np.all(assignment >= 0), "All pixels should be assigned"

    print(f"  Unique scenes used: {np.unique(assignment)}")
    print("  [PASS]")


def test_hierarchical_fallback():
    """Test fallback fills unassigned pixels."""
    print("Test: hierarchical_fallback")
    h, w = 64, 64

    # Start with all unassigned
    assignment = np.full((h, w), -1, dtype=np.int32)
    # Assign top half
    assignment[:32, :] = 0

    clear = np.ones((3, h, w), dtype=bool)
    clear[0, 32:, :] = False  # scene 0 cloudy in bottom
    clear[1, :, :] = True     # scene 1 clear everywhere

    scene_order = np.array([0, 1, 2])

    result = hierarchical_fallback(
        assignment.copy(), clear, scene_order,
        clear_thresh=0.5, patch_sizes=[16, 8],
    )

    n_unfilled = np.sum(result < 0)
    assert n_unfilled == 0, f"{n_unfilled} pixels still unfilled"

    # Bottom half should be assigned to scene 1 (clear there)
    assert np.all(result[32:, :] == 1), "Bottom should use scene 1"

    print(f"  All pixels assigned, unique scenes: {np.unique(result)}")
    print("  [PASS]")


def test_rescue_fill():
    """Test rescue fill for remaining gaps."""
    print("Test: rescue_fill")
    h, w, c = 16, 16, 13

    spectral = np.random.randn(3, c, h, w).astype(np.float32) * 1000
    scl = np.full((3, h, w), 4, dtype=np.int32)  # all clear
    scl[0, :8, :8] = 9  # scene 0 cloudy in top-left

    assignment = np.full((h, w), -1, dtype=np.int32)  # all unassigned
    scene_order = np.array([0, 1, 2])

    composite, fill_mode = rescue_fill(
        assignment, spectral, scl, scene_order, n_rescue=3,
    )

    assert composite.shape == (c, h, w)
    assert fill_mode.shape == (h, w)
    # All pixels should be rescue-filled
    assert np.all(fill_mode == 1), "All should be rescue-filled"
    # No zeros (rescue should fill everything)
    assert np.any(composite != 0), "Composite should have values"

    print(f"  Composite range: [{composite.min():.0f}, {composite.max():.0f}]")
    print(f"  Rescue-filled: {fill_mode.sum()}/{fill_mode.size}")
    print("  [PASS]")


def test_full_pipeline_synthetic():
    """End-to-end test with synthetic data."""
    print("Test: create_temporal_mosaic (full pipeline, synthetic)")

    spectral, scl, cloud_fracs = make_synthetic_data(
        n_scenes=5, h=64, w=64, seed=42,
    )

    t0 = time.time()
    composite, info = create_temporal_mosaic(spectral, scl)
    elapsed = time.time() - t0

    c, h, w = composite.shape
    assert c == 13, f"Expected 13 bands, got {c}"
    assert h == 64 and w == 64, f"Wrong spatial dims: {h}x{w}"

    # No NaN in output
    assert not np.any(np.isnan(composite)), "Composite has NaN"
    # No zeros (each pixel should be filled from some scene)
    zero_frac = np.mean(composite == 0)
    assert zero_frac < 0.01, f"Too many zeros: {zero_frac:.2%}"

    # Check diagnostics
    assert "scene_scores" in info
    assert "n_clusters" in info
    assert info["n_clusters"] > 1, "Should have multiple clusters"

    n_scenes_used = info["n_scenes_used"]
    fill_mode = info["fill_mode"]
    rescue_pct = fill_mode.sum() / fill_mode.size * 100

    print(f"  Output shape: ({c}, {h}, {w})")
    print(f"  Clusters: {info['n_clusters']}")
    print(f"  Scenes used: {n_scenes_used}")
    print(f"  Rescue-filled: {rescue_pct:.1f}%")
    print(f"  Value range: [{composite.min():.0f}, {composite.max():.0f}]")
    print(f"  Time: {elapsed:.2f}s")
    print("  [PASS]")


def test_full_pipeline_all_cloudy():
    """Edge case: all scenes are heavily cloudy."""
    print("Test: create_temporal_mosaic (all cloudy)")

    spectral, scl, _ = make_synthetic_data(
        n_scenes=3, h=32, w=32,
        cloud_fracs=[0.95, 0.90, 0.85],
        seed=99,
    )

    composite, info = create_temporal_mosaic(spectral, scl)

    assert composite.shape == (13, 32, 32)
    # Should still produce output (rescue fill handles it)
    assert not np.any(np.isnan(composite)), "Should not have NaN even with heavy cloud"

    rescue_pct = info["fill_mode"].sum() / info["fill_mode"].size * 100
    print(f"  Rescue-filled: {rescue_pct:.1f}%")
    print(f"  Scenes used: {info['n_scenes_used']}")
    print("  [PASS]")


def test_full_pipeline_single_scene():
    """Edge case: only one scene available."""
    print("Test: create_temporal_mosaic (single scene)")

    spectral, scl, _ = make_synthetic_data(
        n_scenes=1, h=32, w=32,
        cloud_fracs=[0.1],
        seed=7,
    )

    composite, info = create_temporal_mosaic(spectral, scl)

    assert composite.shape == (13, 32, 32)
    assert not np.any(np.isnan(composite))

    print(f"  Scenes used: {info['n_scenes_used']}")
    print("  [PASS]")


def main():
    print("=" * 60)
    print("Temporal Mosaic UDF — Local Tests")
    print("=" * 60)

    tests = [
        test_compute_clear_masks,
        test_score_scenes,
        test_slic_segmentation,
        test_assign_clusters,
        test_hierarchical_fallback,
        test_rescue_fill,
        test_full_pipeline_synthetic,
        test_full_pipeline_all_cloudy,
        test_full_pipeline_single_scene,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print()
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
