"""OpenEO UDF: Temporal Sentinel-2 mosaic with superpixel-based compositing.

Translates the GEE-based mosaic algorithm (mosaic_module/) into a UDF
that runs on any OpenEO backend with numpy + scipy available.

Algorithm (faithful to GEE training pipeline):
  1. Cloud masking via SCL band (approximates s2cloudless prob >= 65)
  2. Scene scoring by clear-pixel fraction over the chip
  3. SLIC superpixel segmentation (scipy, approximates GEE SNIC)
  4. Per-cluster best-scene assignment (highest clear fraction)
  5. Hierarchical fallback for unassigned pixels (32 → 16 → 8 px patches)
  6. Per-pixel rescue fill (lowest cloud probability per pixel)
  7. Spectral stitching from assigned scenes

OpenEO UDF entry point: apply_datacube()

Input datacube dimensions: (t, bands, y, x)
  - bands must include 13 L1C spectral + SCL as the 14th band
Output datacube dimensions: (bands, y, x)
  - 13 spectral L1C bands, cloud-free composite

Runtime requirements (all available on CDSE Python 3.11):
    numpy, scipy, xarray
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters matching training pipeline (extraction_pipeline/config.py)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "snic_size_px": 20,         # superpixel target size (pixels)
    "snic_compactness": 1.0,    # spatial vs spectral weight in SLIC (matches GEE SNIC)
    "clear_thresh": 0.8,        # min clear fraction to assign cluster to scene
    "top_n_scenes": 8,          # max candidate scenes
    "top_n_rescue": 10,         # max rescue scenes
    "patch_sizes": [32, 16, 8], # hierarchical fallback grid sizes
    "feather_px": 0,            # seam feathering (0 = off for 256px chips)
    # SCL classes considered clear (whitelist approach — robust to NaN/missing SCL).
    # 4=vegetation, 5=bare_soil, 6=water, 7=unclassified, 11=snow_ice
    # Everything else (0=no_data, 1=saturated, 2=dark, 3=shadow,
    # 8=cloud_med, 9=cloud_high, 10=cirrus, NaN→int garbage) is masked.
    "scl_clear_classes": [4, 5, 6, 7, 11],
}

# Sentinel-2 L1C band indices (0-12) within the 13-band spectral stack
# B4(3), B3(2), B2(1), B8(7) used for reference image
REF_BAND_INDICES = [3, 2, 1, 7]  # B4, B3, B2, B8


# ===================================================================
# Step 1: Cloud masking from SCL
# ===================================================================

def compute_clear_masks(
    scl_stack: np.ndarray,
    clear_classes: list[int],
) -> np.ndarray:
    """Compute per-scene binary clear masks from SCL.

    Uses a whitelist approach: only pixels whose SCL value is in
    ``clear_classes`` are considered clear.  This is robust to missing
    SCL data (NaN cast to int produces garbage values like -2147483648,
    which would slip through a blacklist but are correctly rejected here).

    Parameters
    ----------
    scl_stack : (T, H, W) array of SCL classification values.
    clear_classes : SCL class values that indicate clear pixels.
        Default: [4, 5, 6, 7, 11] (vegetation, bare, water, unclass, snow).

    Returns
    -------
    clear_masks : (T, H, W) boolean array. True = clear pixel.
    """
    clear = np.isin(scl_stack, clear_classes)

    # Morphological cleaning (approximates GEE focal_min(1) + focal_max(2))
    struct_erode = ndimage.generate_binary_structure(2, 1)   # 3x3 cross
    struct_dilate = np.ones((5, 5), dtype=bool)              # 5x5 square

    for t in range(clear.shape[0]):
        # Invert: work on cloud mask
        cloud = ~clear[t]
        # Erosion removes thin cloud edges (like GEE focal_min(1))
        cloud = ndimage.binary_erosion(cloud, structure=struct_erode)
        # Dilation buffers cloud boundaries (like GEE focal_max(2px))
        cloud = ndimage.binary_dilation(cloud, structure=struct_dilate)
        # Remove small speckle (connected components < 4 pixels)
        labelled, n_features = ndimage.label(cloud)
        if n_features > 0:
            sizes = ndimage.sum(cloud, labelled, range(1, n_features + 1))
            small = np.array([i + 1 for i, s in enumerate(sizes) if s < 4])
            if len(small) > 0:
                cloud[np.isin(labelled, small)] = False
        clear[t] = ~cloud

    return clear


# ===================================================================
# Step 2: Scene scoring
# ===================================================================

def score_scenes(clear_masks: np.ndarray) -> np.ndarray:
    """Score each scene by its clear-pixel fraction over the full chip.

    Uses sum/total (not mean of valid pixels), matching GEE implementation
    which penalises partial-coverage scenes.

    Returns
    -------
    scores : (T,) array of clear fractions in [0, 1].
    """
    total_pixels = clear_masks.shape[1] * clear_masks.shape[2]
    scores = clear_masks.sum(axis=(1, 2)).astype(np.float32) / total_pixels
    return scores


# ===================================================================
# Step 3: SLIC superpixel segmentation (approximates GEE SNIC)
# ===================================================================

def slic_segmentation(
    reference_image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    n_iter: int = 10,
) -> np.ndarray:
    """Simple SLIC superpixel segmentation using scipy only.

    Approximates GEE's ee.Algorithms.Image.Segmentation.SNIC.
    Uses iterative k-means with spatial + spectral distance.

    Parameters
    ----------
    reference_image : (H, W, C) float32 array (cloud-free reference).
    n_segments : target number of superpixels.
    compactness : spatial vs spectral weighting (higher = more compact).
    n_iter : number of k-means iterations.

    Returns
    -------
    labels : (H, W) int32 array of cluster IDs.
    """
    h, w, c = reference_image.shape

    # Initialise cluster centres on a regular grid
    grid_step = max(1, int(np.sqrt(h * w / max(n_segments, 1))))
    ys = np.arange(grid_step // 2, h, grid_step)
    xs = np.arange(grid_step // 2, w, grid_step)
    centres_yx = np.array([(y, x) for y in ys for x in xs], dtype=np.float32)
    n_k = len(centres_yx)
    if n_k == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Spectral values at centres
    centres_spec = np.array([
        reference_image[int(y), int(x)]
        for y, x in centres_yx
    ], dtype=np.float32)

    # Normalise spectral values to [0, 1] for distance computation
    spec_min = reference_image.min()
    spec_range = reference_image.max() - spec_min
    if spec_range < 1e-6:
        spec_range = 1.0
    ref_norm = (reference_image - spec_min) / spec_range
    centres_spec_norm = (centres_spec - spec_min) / spec_range

    # Build pixel coordinate grids
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Spatial normalisation factor
    spatial_scale = compactness / grid_step

    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)

    for _iteration in range(n_iter):
        # Assign each pixel to nearest centre (within 2*grid_step search window)
        for k in range(n_k):
            cy, cx = centres_yx[k]
            cs = centres_spec_norm[k]

            # Search window
            y0 = max(0, int(cy) - 2 * grid_step)
            y1 = min(h, int(cy) + 2 * grid_step + 1)
            x0 = max(0, int(cx) - 2 * grid_step)
            x1 = min(w, int(cx) + 2 * grid_step + 1)

            # Spectral distance
            patch = ref_norm[y0:y1, x0:x1]
            d_spec = np.sum((patch - cs[None, None, :]) ** 2, axis=2)

            # Spatial distance
            d_spatial = (
                (yy[y0:y1, x0:x1] - cy) ** 2
                + (xx[y0:y1, x0:x1] - cx) ** 2
            ) * (spatial_scale ** 2)

            d_total = d_spec + d_spatial

            # Update assignments
            mask = d_total < distances[y0:y1, x0:x1]
            distances[y0:y1, x0:x1] = np.where(mask, d_total, distances[y0:y1, x0:x1])
            labels[y0:y1, x0:x1] = np.where(mask, k, labels[y0:y1, x0:x1])

        # Update centres
        for k in range(n_k):
            members = labels == k
            if np.any(members):
                centres_yx[k, 0] = yy[members].mean()
                centres_yx[k, 1] = xx[members].mean()
                centres_spec_norm[k] = ref_norm[members].mean(axis=0)

    # Fix any unassigned pixels
    if np.any(labels < 0):
        labels[labels < 0] = 0

    return labels


# ===================================================================
# Step 4: Per-cluster best-scene assignment
# ===================================================================

def assign_clusters_to_scenes(
    labels: np.ndarray,
    clear_masks: np.ndarray,
    scene_order: np.ndarray,
    clear_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each superpixel cluster to its best clear scene.

    For each cluster, compute mean clear fraction per candidate scene.
    Assign cluster to the scene with highest clear fraction, provided
    it exceeds clear_thresh.

    Parameters
    ----------
    labels : (H, W) int32 cluster labels.
    clear_masks : (T, H, W) boolean clear masks.
    scene_order : (N,) indices into time axis, sorted by overall score desc.
    clear_thresh : minimum clear fraction for assignment.

    Returns
    -------
    scene_assignment : (H, W) int index into time axis. -1 = unassigned.
    assignment_date : (H, W) same as scene_assignment (for provenance).
    """
    h, w = labels.shape
    scene_assignment = np.full((h, w), -1, dtype=np.int32)

    n_clusters = labels.max() + 1
    candidates = scene_order

    # Precompute cluster membership masks
    cluster_pixels = {}
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            cluster_pixels[k] = mask

    # For each cluster, find the candidate scene with best clear fraction
    for k, member_mask in cluster_pixels.items():
        n_pixels = member_mask.sum()
        best_scene = -1
        best_frac = -1.0

        for scene_idx in candidates:
            clear_in_cluster = clear_masks[scene_idx][member_mask].sum()
            frac = clear_in_cluster / n_pixels
            if frac > best_frac:
                best_frac = frac
                best_scene = scene_idx

        if best_frac >= clear_thresh and best_scene >= 0:
            scene_assignment[member_mask] = best_scene

    return scene_assignment


# ===================================================================
# Step 5: Hierarchical fallback
# ===================================================================

def hierarchical_fallback(
    scene_assignment: np.ndarray,
    clear_masks: np.ndarray,
    scene_order: np.ndarray,
    clear_thresh: float,
    patch_sizes: list[int],
) -> np.ndarray:
    """Fill unassigned pixels using progressively finer grid patches.

    Mirrors GEE hierarchical_split_and_assign().

    For each patch size (e.g. 32, 16, 8), divide the image into a grid
    of that size. For each patch, compute clear fraction per candidate
    and assign the patch to the best scene.
    """
    h, w = scene_assignment.shape

    for patch_size in patch_sizes:
        for y0 in range(0, h, patch_size):
            for x0 in range(0, w, patch_size):
                y1 = min(y0 + patch_size, h)
                x1 = min(x0 + patch_size, w)

                patch_slice = (slice(y0, y1), slice(x0, x1))
                unassigned = scene_assignment[patch_slice] < 0

                if not np.any(unassigned):
                    continue

                # Find best scene for this patch
                n_pixels = unassigned.sum()
                best_scene = -1
                best_frac = -1.0

                for scene_idx in scene_order:
                    clear_patch = clear_masks[scene_idx][y0:y1, x0:x1]
                    clear_in_unassigned = clear_patch[unassigned].sum()
                    frac = clear_in_unassigned / n_pixels
                    if frac > best_frac:
                        best_frac = frac
                        best_scene = scene_idx

                if best_frac >= clear_thresh and best_scene >= 0:
                    # Only assign the unassigned pixels in this patch
                    assign_patch = scene_assignment[patch_slice]
                    assign_patch[unassigned] = best_scene
                    scene_assignment[patch_slice] = assign_patch

    # Last resort: per-pixel best clear scene
    still_unassigned = scene_assignment < 0
    if np.any(still_unassigned):
        unassigned_ys, unassigned_xs = np.where(still_unassigned)
        for y, x in zip(unassigned_ys, unassigned_xs):
            for scene_idx in scene_order:
                if clear_masks[scene_idx, y, x]:
                    scene_assignment[y, x] = scene_idx
                    break

    return scene_assignment


# ===================================================================
# Step 6: Rescue fill
# ===================================================================

def rescue_fill(
    scene_assignment: np.ndarray,
    spectral_stack: np.ndarray,
    scl_stack: np.ndarray,
    scene_order: np.ndarray,
    n_rescue: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill remaining gaps using per-pixel lowest cloud probability.

    Mirrors GEE _build_rescue_fill(): for each unfilled pixel, select
    the observation with the lowest SCL cloud indicator.

    Parameters
    ----------
    scene_assignment : (H, W) int, -1 = still unassigned.
    spectral_stack : (T, C, H, W) spectral data.
    scl_stack : (T, H, W) SCL values.
    scene_order : indices sorted by overall score.
    n_rescue : max number of rescue candidates.

    Returns
    -------
    composite : (C, H, W) filled spectral composite.
    fill_mode : (H, W) 0 = primary assignment, 1 = rescue filled.
    """
    t, c, h, w = spectral_stack.shape
    composite = np.zeros((c, h, w), dtype=np.float32)
    fill_mode = np.zeros((h, w), dtype=np.uint8)

    # Build composite from primary assignments (vectorised)
    assigned_mask = scene_assignment >= 0
    fill_mode[~assigned_mask] = 1
    if np.any(assigned_mask):
        ys, xs = np.where(assigned_mask)
        ss = scene_assignment[assigned_mask]
        # spectral_stack[ss, :, ys, xs] → (N, C), need (C, N) for composite[:, ys, xs]
        composite[:, ys, xs] = spectral_stack[ss, :, ys, xs].T

    # Rescue: for unfilled pixels, pick the scene with lowest cloud indicator
    unfilled = scene_assignment < 0
    if np.any(unfilled):
        rescue_candidates = scene_order[:n_rescue]
        uf_ys, uf_xs = np.where(unfilled)

        # For each unfilled pixel, find rescue scene with lowest "cloud score"
        # Use SCL value as proxy: lower non-cloud SCL = better
        # SCL 4=vegetation, 5=bare, 6=water are best (low values = clear)
        for y, x in zip(uf_ys, uf_xs):
            best_scene = rescue_candidates[0]  # fallback
            best_cloud = 999
            for s in rescue_candidates:
                scl_val = scl_stack[s, y, x]
                # Prefer clear classes (4, 5, 6, 7, 11)
                cloud_score = scl_val if scl_val in (0, 1, 3, 8, 9, 10) else 0
                if cloud_score < best_cloud:
                    best_cloud = cloud_score
                    best_scene = s
            composite[:, y, x] = spectral_stack[best_scene, :, y, x]
            fill_mode[y, x] = 1

    return composite, fill_mode


# ===================================================================
# Step 7: Seam feathering
# ===================================================================

def feather_seams(
    composite: np.ndarray,
    scene_assignment: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    """Apply light feathering at scene assignment boundaries.

    Mirrors GEE feather_and_blend(): detect boundaries where adjacent
    pixels come from different scenes, then smooth within a buffer zone.
    """
    if feather_px <= 0:
        return composite

    # Detect boundaries
    shifted_h = np.roll(scene_assignment, 1, axis=0)
    shifted_w = np.roll(scene_assignment, 1, axis=1)
    boundary = (scene_assignment != shifted_h) | (scene_assignment != shifted_w)

    # Dilate boundary zone
    struct = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), dtype=bool)
    boundary_zone = ndimage.binary_dilation(boundary, structure=struct)

    # Smooth within boundary zone
    kernel_size = max(1, feather_px // 2)
    smoothed = ndimage.uniform_filter(composite.astype(np.float32),
                                       size=(1, kernel_size, kernel_size))

    result = composite.copy()
    for b in range(composite.shape[0]):
        result[b][boundary_zone] = smoothed[b][boundary_zone]

    return result


# ===================================================================
# Main compositing function
# ===================================================================

def create_temporal_mosaic(
    spectral_stack: np.ndarray,
    scl_stack: np.ndarray,
    params: dict | None = None,
) -> Tuple[np.ndarray, dict]:
    """Create a cloud-free temporal mosaic from a multi-temporal stack.

    This is the main entry point, equivalent to GEE create_temporal_mosaic().

    Parameters
    ----------
    spectral_stack : (T, C, H, W) float32
        Multi-temporal Sentinel-2 L1C spectral data. C=13 bands.
    scl_stack : (T, H, W) float32 or int
        SCL classification for each timestep.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    composite : (C, H, W) float32, cloud-free 13-band composite.
    info : dict with diagnostics (fill_mode, scene_assignment, etc.).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    t, c, h, w = spectral_stack.shape
    logger.info("Mosaic input: %d scenes, %d bands, %dx%d", t, c, h, w)

    # --- Step 1: Cloud masking ---
    scl_int = scl_stack.astype(np.int32)
    clear_masks = compute_clear_masks(scl_int, p["scl_clear_classes"])
    logger.info("Clear masks computed")

    # --- Step 2: Scene scoring ---
    scores = score_scenes(clear_masks)
    scene_order = np.argsort(-scores)  # descending
    top_n = min(p["top_n_scenes"], t)
    candidates = scene_order[:top_n]
    logger.info("Scene scores: %s", dict(zip(scene_order[:top_n].tolist(),
                                              scores[scene_order[:top_n]].tolist())))

    # --- Step 3: Build reference image for segmentation ---
    # Median of clear pixels from top candidates (B4, B3, B2, B8)
    ref_bands = []
    for band_idx in REF_BAND_INDICES:
        stack = []
        for s in candidates:
            masked = np.where(clear_masks[s], spectral_stack[s, band_idx], np.nan)
            stack.append(masked)
        ref_bands.append(np.nanmedian(stack, axis=0))
    reference = np.stack(ref_bands, axis=-1)  # (H, W, 4)

    # Add NDVI as 5th channel (matches GEE reference)
    b8 = reference[:, :, 3].astype(np.float64)
    b4 = reference[:, :, 0].astype(np.float64)
    ndvi = np.where((b8 + b4) > 0, (b8 - b4) / (b8 + b4 + 1e-6), 0)
    reference = np.concatenate([reference, ndvi[:, :, None]], axis=-1)

    # Fill NaN in reference with 0 (fallback for fully cloudy pixels)
    reference = np.nan_to_num(reference, nan=0.0).astype(np.float32)

    # --- Step 4: SLIC superpixel segmentation ---
    n_segments = max(1, (h * w) // (p["snic_size_px"] ** 2))
    labels = slic_segmentation(
        reference,
        n_segments=n_segments,
        compactness=p["snic_compactness"],
        n_iter=5,
    )
    n_clusters = labels.max() + 1
    logger.info("SLIC segmentation: %d clusters (target %d)", n_clusters, n_segments)

    # --- Step 5: Per-cluster scene assignment ---
    scene_assignment = assign_clusters_to_scenes(
        labels=labels,
        clear_masks=clear_masks,
        scene_order=candidates,
        clear_thresh=p["clear_thresh"],
    )
    n_assigned = np.sum(scene_assignment >= 0)
    logger.info("Cluster assignment: %d/%d pixels assigned (%.1f%%)",
                n_assigned, h * w, n_assigned / (h * w) * 100)

    # --- Step 6: Hierarchical fallback ---
    scene_assignment = hierarchical_fallback(
        scene_assignment=scene_assignment,
        clear_masks=clear_masks,
        scene_order=candidates,
        clear_thresh=p["clear_thresh"],
        patch_sizes=p["patch_sizes"],
    )
    n_assigned_after = np.sum(scene_assignment >= 0)
    logger.info("After fallback: %d/%d pixels (%.1f%%)",
                n_assigned_after, h * w, n_assigned_after / (h * w) * 100)

    # --- Step 7: Rescue fill ---
    n_rescue = min(p["top_n_rescue"], t)
    composite, fill_mode = rescue_fill(
        scene_assignment=scene_assignment,
        spectral_stack=spectral_stack,
        scl_stack=scl_int,
        scene_order=scene_order[:n_rescue],
        n_rescue=n_rescue,
    )
    n_rescued = np.sum(fill_mode > 0)
    logger.info("Rescue fill: %d pixels (%.1f%%)",
                n_rescued, n_rescued / (h * w) * 100)

    # --- Step 8: Seam feathering ---
    composite = feather_seams(composite, scene_assignment, p["feather_px"])

    info = {
        "scene_scores": scores,
        "scene_order": scene_order,
        "scene_assignment": scene_assignment,
        "fill_mode": fill_mode,
        "n_clusters": n_clusters,
        "n_scenes_used": len(np.unique(scene_assignment[scene_assignment >= 0])),
    }

    return composite, info


# ===================================================================
# OpenEO UDF entry point
# ===================================================================

def apply_datacube(cube, context: dict) -> "XarrayDataCube":
    """OpenEO UDF entry point for temporal mosaicing.

    Expects a datacube with dimensions (t, bands, y, x) where bands
    includes 13 L1C spectral bands + SCL as the last band.

    The SCL band must be loaded from SENTINEL2_L2A and merged into the
    L1C datacube before calling this UDF.

    Parameters
    ----------
    cube : openeo.udf.XarrayDataCube
        Input datacube with shape (T, 14, H, W) — 13 spectral + SCL.
    context : dict
        Optional parameter overrides (clear_thresh, top_n_scenes, etc.).

    Returns
    -------
    openeo.udf.XarrayDataCube
        Single-timestep datacube with 13 spectral bands.
    """
    import xarray as xr
    from openeo.udf import XarrayDataCube

    array = cube.get_array()  # xarray.DataArray
    data = array.values  # numpy array

    # Determine dimension order and reshape to (T, bands, H, W)
    dims = list(array.dims)

    # Find dimension names
    t_dim = next((d for d in dims if d in ("t", "time")), None)
    b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)

    if t_dim is None or b_dim is None:
        raise ValueError(f"Expected (t, bands, y, x) dimensions, got {dims}")

    # Transpose to (t, bands, y, x)
    spatial_dims = [d for d in dims if d not in (t_dim, b_dim)]
    data = array.transpose(t_dim, b_dim, *spatial_dims).values

    t, n_bands, h, w = data.shape

    # Split spectral (first 13) and SCL (last band)
    if n_bands >= 14:
        spectral = data[:, :13, :, :].astype(np.float32)
        scl = data[:, 13, :, :]  # (T, H, W)
    elif n_bands == 13:
        # No SCL provided — fall back to simple median
        logger.warning("No SCL band provided, falling back to median composite")
        composite = np.nanmedian(spectral, axis=0)
        result = xr.DataArray(composite, dims=[b_dim, *spatial_dims])
        return XarrayDataCube(result)
    else:
        raise ValueError(f"Expected 13 or 14 bands, got {n_bands}")

    # Merge context parameters
    params = {**DEFAULT_PARAMS}
    for key in DEFAULT_PARAMS:
        if key in context:
            params[key] = context[key]

    # Run mosaic algorithm
    composite, info = create_temporal_mosaic(spectral, scl, params)

    # Build output DataArray (bands, y, x)
    y_dim = spatial_dims[0]
    x_dim = spatial_dims[1]

    result = xr.DataArray(
        composite,
        dims=[b_dim, y_dim, x_dim],
        coords={
            y_dim: array.coords[y_dim] if y_dim in array.coords else np.arange(h),
            x_dim: array.coords[x_dim] if x_dim in array.coords else np.arange(w),
        },
    )

    return XarrayDataCube(result)
