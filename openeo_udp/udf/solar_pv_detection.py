"""OpenEO UDF: Solar PV detection (SLIC temporal mosaic + ONNX inference).

This UDF performs the full per-chunk pipeline in a single executor call:
    1. SLIC-based cloud-free temporal mosaic of a multi-temporal Sentinel-2
       L1C + SCL stack (mirrors the GEE SNIC pipeline used to build the
       training chips).
    2. Training-aligned percentile normalization (matches
       ``solar_ml.data.normalize_batch`` with ``mode='percentile'``).
    3. ONNX U-Net inference (fixed 256x256 / 13-band input).

Merging both steps avoids an intermediate cube materialisation/shuffle
between two ``apply_neighborhood`` calls.

Invocation
----------
Call via ``apply_neighborhood`` on chunks of shape (T, 14, 256, 256)
where the 14th band is SCL. Use ``size=192`` + ``overlap=32`` so each
chunk is exactly 256x256 (the model input size) and adjacent chunks
share a 32 px halo for seam-free inference.

Dependency archives (job options ``udf-dependency-archives``)::

    onnx_deps/    -> onnxruntime python package
    onnx_models/  -> solar_pv.onnx + band_stats.npz

Output bands (2):
    - solar_pv               (uint8 binary)
    - solar_pv_probability   (float32)
    - pre_norm_mean          (float32, mean across 13 bands pre-norm)
    - post_norm_mean         (float32, mean across 13 bands post-norm)

Context overrides::

    {
        "threshold": 0.80,             # detection threshold
        "clear_thresh": 0.8,           # mosaic cluster clear-fraction threshold
        "top_n_scenes": 8,             # max mosaic candidates
        "top_n_rescue": 10,            # max rescue scenes
        "snic_size_px": 20,            # SLIC superpixel size
        "snic_compactness": 1.0,       # SLIC compactness
    }
"""

import functools
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from scipy import ndimage

from openeo.metadata import CollectionMetadata

# ---------------------------------------------------------------------------
# Make UDF dependency archives importable.
# ---------------------------------------------------------------------------
sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402

sys.path.append("onnx_models")

logger = logging.getLogger(__name__)


# ===========================================================================
# Constants
# ===========================================================================
NUM_THREADS = 2

DEFAULT_MODEL_NAME = "solar_pv.onnx"
DEFAULT_THRESHOLD = 0.80

MODEL_DIR = "onnx_models/solar_pv_rui"
BAND_STATS_FILENAME = "band_stats.npz"

S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Mosaic defaults match extraction_pipeline/config.py
DEFAULT_MOSAIC_PARAMS = {
    "snic_size_px": 20,
    "snic_compactness": 1.0,
    "clear_thresh": 0.8,
    "top_n_scenes": 8,
    "top_n_rescue": 10,
    "patch_sizes": [32, 16, 8],
    "feather_px": 0,
    # 4=vegetation, 5=bare, 6=water, 7=unclassified, 11=snow_ice
    "scl_clear_classes": [4, 5, 6, 7, 11],
}

# Band indices (0-12) of the 13-band L1C stack used to build the reference
# image for SLIC segmentation: B4, B3, B2, B8.
REF_BAND_INDICES = [3, 2, 1, 7]


# ===========================================================================
# ONNX session + band stats loaders (cached per executor)
# ===========================================================================

def _ort_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = NUM_THREADS
    so.inter_op_num_threads = NUM_THREADS
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    return so


@functools.lru_cache(maxsize=1)
def _load_session() -> ort.InferenceSession:
    """Load the ONNX model from the dependency archive."""
    model_root = Path(MODEL_DIR)
    model_path = model_root / DEFAULT_MODEL_NAME

    if not model_path.exists():
        by_name = list(model_root.rglob(DEFAULT_MODEL_NAME))
        if len(by_name) == 1:
            model_path = by_name[0]
        elif len(by_name) > 1:
            raise FileNotFoundError(
                f"Multiple ONNX model matches for '{DEFAULT_MODEL_NAME}' under {model_root}."
            )
        else:
            all_onnx = list(model_root.rglob("*.onnx"))
            if len(all_onnx) == 1:
                model_path = all_onnx[0]
                logger.info("Auto-selected ONNX model: %s", model_path)
            else:
                raise FileNotFoundError(
                    f"ONNX model not found under {model_root}. Found: {all_onnx}"
                )

    logger.info("Loading ONNX model: %s", model_path)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )

    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))
    return session


@functools.lru_cache(maxsize=1)
def _load_band_stats() -> dict:
    """Load per-band stats (mean, std, p2, p98) from band_stats.npz."""
    model_root = Path(MODEL_DIR)
    stats_path = model_root / BAND_STATS_FILENAME

    if not stats_path.exists():
        matches = list(model_root.rglob(BAND_STATS_FILENAME))
        if len(matches) >= 1:
            stats_path = matches[0]
            if len(matches) > 1:
                logger.warning(
                    "Multiple %s files; using %s", BAND_STATS_FILENAME, stats_path,
                )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"band_stats.npz not found under {model_root}."
        )

    data = np.load(stats_path)
    if "p2" not in data.files or "p98" not in data.files:
        raise ValueError(
            f"band_stats.npz at {stats_path} missing 'p2'/'p98' "
            f"(found {list(data.files)})."
        )
    stats = {
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
        "p2": data["p2"].astype(np.float32),
        "p98": data["p98"].astype(np.float32),
    }
    for key in ("mean", "std", "p2", "p98"):
        if stats[key].shape != (13,):
            raise ValueError(
                f"band_stats[{key}] shape {stats[key].shape} != (13,)"
            )
    logger.info(
        "Loaded band_stats: p2=%s p98=%s", stats["p2"], stats["p98"],
    )
    return stats


# ===========================================================================
# Mosaic helpers (ported from openeo_udp/udf/temporal_mosaic.py)
# ===========================================================================

def _compute_clear_masks(scl_stack: np.ndarray, clear_classes: list[int]) -> np.ndarray:
    """Per-scene binary clear masks from SCL (whitelist + morph cleaning)."""
    clear = np.isin(scl_stack, clear_classes)
    struct_erode = ndimage.generate_binary_structure(2, 1)
    struct_dilate = np.ones((5, 5), dtype=bool)

    for t in range(clear.shape[0]):
        cloud = ~clear[t]
        cloud = ndimage.binary_erosion(cloud, structure=struct_erode)
        cloud = ndimage.binary_dilation(cloud, structure=struct_dilate)
        labelled, n_features = ndimage.label(cloud)
        if n_features > 0:
            sizes = ndimage.sum(cloud, labelled, range(1, n_features + 1))
            small = np.array([i + 1 for i, s in enumerate(sizes) if s < 4])
            if len(small) > 0:
                cloud[np.isin(labelled, small)] = False
        clear[t] = ~cloud
    return clear


def _score_scenes(clear_masks: np.ndarray) -> np.ndarray:
    total = clear_masks.shape[1] * clear_masks.shape[2]
    return clear_masks.sum(axis=(1, 2)).astype(np.float32) / total


def _slic_segmentation(
    reference_image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    n_iter: int = 5,
) -> np.ndarray:
    """Simple SLIC superpixels (k-means with spatial+spectral distance)."""
    h, w, _c = reference_image.shape
    grid_step = max(1, int(np.sqrt(h * w / max(n_segments, 1))))
    ys = np.arange(grid_step // 2, h, grid_step)
    xs = np.arange(grid_step // 2, w, grid_step)
    centres_yx = np.array([(y, x) for y in ys for x in xs], dtype=np.float32)
    n_k = len(centres_yx)
    if n_k == 0:
        return np.zeros((h, w), dtype=np.int32)

    centres_spec = np.array(
        [reference_image[int(y), int(x)] for y, x in centres_yx],
        dtype=np.float32,
    )

    spec_min = reference_image.min()
    spec_range = reference_image.max() - spec_min
    if spec_range < 1e-6:
        spec_range = 1.0
    ref_norm = (reference_image - spec_min) / spec_range
    centres_spec_norm = (centres_spec - spec_min) / spec_range

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    spatial_scale = compactness / grid_step

    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)

    for _ in range(n_iter):
        for k in range(n_k):
            cy, cx = centres_yx[k]
            cs = centres_spec_norm[k]
            y0 = max(0, int(cy) - 2 * grid_step)
            y1 = min(h, int(cy) + 2 * grid_step + 1)
            x0 = max(0, int(cx) - 2 * grid_step)
            x1 = min(w, int(cx) + 2 * grid_step + 1)

            patch = ref_norm[y0:y1, x0:x1]
            d_spec = np.sum((patch - cs[None, None, :]) ** 2, axis=2)
            d_spatial = (
                (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
            ) * (spatial_scale ** 2)
            d_total = d_spec + d_spatial

            mask = d_total < distances[y0:y1, x0:x1]
            distances[y0:y1, x0:x1] = np.where(mask, d_total, distances[y0:y1, x0:x1])
            labels[y0:y1, x0:x1] = np.where(mask, k, labels[y0:y1, x0:x1])

        for k in range(n_k):
            members = labels == k
            if np.any(members):
                centres_yx[k, 0] = yy[members].mean()
                centres_yx[k, 1] = xx[members].mean()
                centres_spec_norm[k] = ref_norm[members].mean(axis=0)

    if np.any(labels < 0):
        labels[labels < 0] = 0
    return labels


def _assign_clusters_to_scenes(
    labels: np.ndarray,
    clear_masks: np.ndarray,
    scene_order: np.ndarray,
    clear_thresh: float,
) -> np.ndarray:
    h, w = labels.shape
    scene_assignment = np.full((h, w), -1, dtype=np.int32)
    n_clusters = labels.max() + 1

    for k in range(n_clusters):
        member_mask = labels == k
        n_pixels = member_mask.sum()
        if n_pixels == 0:
            continue
        best_scene = -1
        best_frac = -1.0
        for scene_idx in scene_order:
            frac = clear_masks[scene_idx][member_mask].sum() / n_pixels
            if frac > best_frac:
                best_frac = frac
                best_scene = scene_idx
        if best_frac >= clear_thresh and best_scene >= 0:
            scene_assignment[member_mask] = best_scene
    return scene_assignment


def _hierarchical_fallback(
    scene_assignment: np.ndarray,
    clear_masks: np.ndarray,
    scene_order: np.ndarray,
    clear_thresh: float,
    patch_sizes: list[int],
) -> np.ndarray:
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
                n_pixels = unassigned.sum()
                best_scene = -1
                best_frac = -1.0
                for scene_idx in scene_order:
                    clear_patch = clear_masks[scene_idx][y0:y1, x0:x1]
                    frac = clear_patch[unassigned].sum() / n_pixels
                    if frac > best_frac:
                        best_frac = frac
                        best_scene = scene_idx
                if best_frac >= clear_thresh and best_scene >= 0:
                    assign_patch = scene_assignment[patch_slice]
                    assign_patch[unassigned] = best_scene
                    scene_assignment[patch_slice] = assign_patch

    still = scene_assignment < 0
    if np.any(still):
        ys_u, xs_u = np.where(still)
        for y, x in zip(ys_u, xs_u):
            for scene_idx in scene_order:
                if clear_masks[scene_idx, y, x]:
                    scene_assignment[y, x] = scene_idx
                    break
    return scene_assignment


def _rescue_fill(
    scene_assignment: np.ndarray,
    spectral_stack: np.ndarray,
    scl_stack: np.ndarray,
    scene_order: np.ndarray,
    n_rescue: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _t, c, h, w = spectral_stack.shape
    composite = np.zeros((c, h, w), dtype=np.float32)
    fill_mode = np.zeros((h, w), dtype=np.uint8)

    assigned_mask = scene_assignment >= 0
    fill_mode[~assigned_mask] = 1
    if np.any(assigned_mask):
        ys, xs = np.where(assigned_mask)
        ss = scene_assignment[assigned_mask]
        composite[:, ys, xs] = spectral_stack[ss, :, ys, xs].T

    unfilled = scene_assignment < 0
    if np.any(unfilled):
        rescue_candidates = scene_order[:n_rescue]
        uf_ys, uf_xs = np.where(unfilled)
        for y, x in zip(uf_ys, uf_xs):
            best_scene = rescue_candidates[0]
            best_cloud = 999
            for s in rescue_candidates:
                scl_val = scl_stack[s, y, x]
                cloud_score = scl_val if scl_val in (0, 1, 3, 8, 9, 10) else 0
                if cloud_score < best_cloud:
                    best_cloud = cloud_score
                    best_scene = s
            composite[:, y, x] = spectral_stack[best_scene, :, y, x]
            fill_mode[y, x] = 1
    return composite, fill_mode


def _create_temporal_mosaic(
    spectral_stack: np.ndarray,
    scl_stack: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Cloud-free composite. spectral=(T,13,H,W), scl=(T,H,W) -> (13,H,W)."""
    t, _c, h, w = spectral_stack.shape

    scl_int = scl_stack.astype(np.int32)
    clear_masks = _compute_clear_masks(scl_int, params["scl_clear_classes"])
    scores = _score_scenes(clear_masks)
    scene_order = np.argsort(-scores)

    top_n = min(params["top_n_scenes"], t)
    candidates = scene_order[:top_n]

    # Reference image (median of clear B4,B3,B2,B8 across top candidates + NDVI)
    ref_bands = []
    for band_idx in REF_BAND_INDICES:
        stack = []
        for s in candidates:
            masked = np.where(clear_masks[s], spectral_stack[s, band_idx], np.nan)
            stack.append(masked)
        ref_bands.append(np.nanmedian(stack, axis=0))
    reference = np.stack(ref_bands, axis=-1)
    b8 = reference[:, :, 3].astype(np.float64)
    b4 = reference[:, :, 0].astype(np.float64)
    ndvi = np.where((b8 + b4) > 0, (b8 - b4) / (b8 + b4 + 1e-6), 0)
    reference = np.concatenate([reference, ndvi[:, :, None]], axis=-1)
    reference = np.nan_to_num(reference, nan=0.0).astype(np.float32)

    n_segments = max(1, (h * w) // (params["snic_size_px"] ** 2))
    labels = _slic_segmentation(
        reference,
        n_segments=n_segments,
        compactness=params["snic_compactness"],
        n_iter=5,
    )

    scene_assignment = _assign_clusters_to_scenes(
        labels=labels,
        clear_masks=clear_masks,
        scene_order=candidates,
        clear_thresh=params["clear_thresh"],
    )
    scene_assignment = _hierarchical_fallback(
        scene_assignment=scene_assignment,
        clear_masks=clear_masks,
        scene_order=candidates,
        clear_thresh=params["clear_thresh"],
        patch_sizes=params["patch_sizes"],
    )

    n_rescue = min(params["top_n_rescue"], t)
    composite, _fill_mode = _rescue_fill(
        scene_assignment=scene_assignment,
        spectral_stack=spectral_stack,
        scl_stack=scl_int,
        scene_order=scene_order[:n_rescue],
        n_rescue=n_rescue,
    )
    return composite


# ===========================================================================
# Inference helpers
# ===========================================================================

def _normalize_training(image_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Percentile normalization, mirroring solar_ml.data.normalize_batch."""
    stats = _load_band_stats()
    mean = stats["mean"]
    p2 = stats["p2"]
    p98 = stats["p98"]

    invalid_mask = ~np.isfinite(image_hwc).all(axis=-1)

    image = image_hwc.astype(np.float32, copy=True)
    for b in range(image.shape[-1]):
        band = image[:, :, b]
        nan_mask = ~np.isfinite(band)
        if nan_mask.any():
            band[nan_mask] = float(mean[b])

    denom = np.maximum(p98 - p2, 1.0)
    image_norm = np.clip((image - p2[None, None, :]) / denom[None, None, :], 0.0, 1.0)
    image_norm = np.nan_to_num(image_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return image_norm.astype(np.float32), invalid_mask


def _run_inference(
    session: ort.InferenceSession,
    image_hwc: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX inference. Returns (binary, probs)."""
    h, w = image_hwc.shape[:2]

    image_norm, invalid_mask = _normalize_training(image_hwc.astype(np.float32))

    if bool(invalid_mask.all()):
        return (
            np.zeros((h, w), dtype=np.uint8),
            np.zeros((h, w), dtype=np.float32),
        )

    batch = image_norm[np.newaxis, ...]  # (1, H, W, 13)

    logger.info(
        "NORMALIZED input: shape=%s min=%.4f max=%.4f mean=%.4f",
        batch.shape, float(batch.min()), float(batch.max()), float(batch.mean()),
    )

    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: batch})[0]
    probs = np.squeeze(probs, axis=(0, -1)).astype(np.float32)

    logger.info(
        "MODEL OUTPUT: shape=%s min=%.4f max=%.4f mean=%.4f",
        probs.shape, float(probs.min()), float(probs.max()), float(probs.mean()),
    )

    probs[invalid_mask] = 0.0
    binary = (probs > threshold).astype(np.uint8)
    return binary, probs


# ===========================================================================
# OpenEO UDF entry points
# ===========================================================================

def _resolve_mosaic_params(context: dict) -> dict:
    params = {**DEFAULT_MOSAIC_PARAMS}
    for key in DEFAULT_MOSAIC_PARAMS:
        if key in context:
            params[key] = context[key]
    return params


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Declare the 2-band output schema."""
    return metadata.rename_labels(
        dimension="bands",
        target=["solar_pv", "solar_pv_probability"],
    )


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """Main UDF entry point: mosaic -> normalize -> ONNX -> 2-band output.

    Input cube dims: (t, bands, y, x) with 14 bands (13 L1C + SCL).
    Output cube dims: (bands, y, x) with 2 bands.
    """
    threshold = float(context.get("threshold", DEFAULT_THRESHOLD))
    mosaic_params = _resolve_mosaic_params(context)

    dims = list(cube.dims)
    t_dim = next((d for d in dims if d in ("t", "time")), None)
    b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)
    if t_dim is None or b_dim is None:
        raise ValueError(f"Expected (t, bands, y, x) dimensions, got {dims}")
    spatial_dims = [d for d in dims if d not in (t_dim, b_dim)]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")
    y_dim, x_dim = spatial_dims

    # Transpose to (t, bands, y, x)
    data = cube.transpose(t_dim, b_dim, y_dim, x_dim).values
    _t, n_bands, h, w = data.shape

    if n_bands < 14:
        raise ValueError(
            f"Expected >=14 bands (13 L1C + SCL), got {n_bands}. "
            f"Did you forget to merge SCL into the L1C cube?"
        )

    spectral = data[:, :13, :, :].astype(np.float32)
    scl = data[:, 13, :, :]

    # --- Mosaic ---
    composite = _create_temporal_mosaic(spectral, scl, mosaic_params)  # (13, H, W)
    logger.info(
        "Mosaic done: shape=%s finite_pct=%.1f%%",
        composite.shape,
        100.0 * np.isfinite(composite).mean(),
    )

    # --- Inference ---
    image_hwc = np.transpose(composite, (1, 2, 0))  # (H, W, 13)
    session = _load_session()
    binary, probs = _run_inference(session, image_hwc, threshold)

    stacked = np.stack(
        [
            binary.astype(np.float32),
            probs.astype(np.float32),
        ],
        axis=0,
    )

    coords: dict = {}
    if y_dim in cube.coords:
        coords[y_dim] = cube.coords[y_dim]
    if x_dim in cube.coords:
        coords[x_dim] = cube.coords[x_dim]

    return xr.DataArray(
        stacked,
        dims=("bands", y_dim, x_dim),
        coords=coords,
    )
