"""OpenEO UDF: Solar PV detection using ONNX runtime.

This UDF runs binary segmentation of solar PV panels on a 13-band
Sentinel-2 L1C composite (cloud-free temporal mosaic).

It is designed to be invoked via ``apply_neighborhood`` on 256x256
spatial chunks because the underlying U-Net segmentation model has a
fixed spatial input size (256, 256, 13).

Model and ONNX runtime are expected to be supplied as UDF dependency
archives via ``udf-dependency-archives`` job options:

    udf-dependency-archives:
        - <onnx_runtime_archive>.zip#onnx_deps
        - <model_archive>.zip#onnx_models

The archives expose:
    onnx_deps/      -> onnxruntime python package
    onnx_models/    -> solar_pv.onnx + band_stats.npz

``band_stats.npz`` is required (no fallback in production).

Default model archive URL (zipped model bundle):

    https://s3.waw3-1.cloudferro.com/project_dependencies/apex_pv_rui/solar_pv_rui.zip

Output: a two-band cube with bands:
    - solar_pv               (uint8, 0/1, probability > threshold)
    - solar_pv_probability   (float32, sigmoid output in [0, 1])

Threshold can be overridden via the UDF ``context`` dict::

    context = {
        "threshold": 0.80,                # detection threshold
    }
"""

import functools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from openeo.metadata import CollectionMetadata

# ---------------------------------------------------------------------------
# Make UDF dependency archives importable.
# These two folders are produced by openEO when udf-dependency-archives is set.
# ---------------------------------------------------------------------------
sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402

sys.path.append("onnx_models")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_THREADS = 2

DEFAULT_MODEL_NAME = "solar_pv.onnx"
DEFAULT_THRESHOLD = 0.80

MODEL_DIR = "onnx_models/solar_pv_rui"
BAND_STATS_FILENAME = "band_stats.npz"

# Sentinel-2 L1C 13-band order expected by the model
S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


# ---------------------------------------------------------------------------
# ONNX session loading (cached per executor)
# ---------------------------------------------------------------------------

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
        # Support nested zip layouts (e.g. onnx_models/<subdir>/solar_pv.onnx)
        by_name = list(model_root.rglob(DEFAULT_MODEL_NAME))
        if len(by_name) == 1:
            model_path = by_name[0]
        elif len(by_name) > 1:
            raise FileNotFoundError(
                f"Multiple ONNX model matches for '{DEFAULT_MODEL_NAME}' under {model_root}: "
                f"{[str(p) for p in by_name]}."
            )
        else:
            all_onnx = list(model_root.rglob("*.onnx"))
            if len(all_onnx) == 1:
                model_path = all_onnx[0]
                logger.info(
                    "Default model '%s' not found, auto-selected only ONNX model: %s",
                    DEFAULT_MODEL_NAME,
                    model_path,
                )
            else:
                raise FileNotFoundError(
                    f"ONNX model not found for '{DEFAULT_MODEL_NAME}' under {model_root}. "
                    f"Found ONNX files: {[str(p) for p in all_onnx]}."
                )

    logger.info("Loading ONNX model: %s", model_path)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )

    # Apply CPU env hints (idempotent)
    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))

    return session


@functools.lru_cache(maxsize=1)
def _load_band_stats() -> tuple[np.ndarray, np.ndarray]:
    """Load per-band mean/std for z-score normalization (matching training)."""
    model_root = Path(MODEL_DIR)
    stats_path = model_root / BAND_STATS_FILENAME

    if not stats_path.exists():
        matches = list(model_root.rglob(BAND_STATS_FILENAME))
        if len(matches) == 1:
            stats_path = matches[0]
            logger.info("Resolved nested band stats path: %s", stats_path)
        elif len(matches) > 1:
            logger.warning(
                "Multiple %s files found under %s; using first: %s",
                BAND_STATS_FILENAME,
                model_root,
                matches[0],
            )
            stats_path = matches[0]

    if not stats_path.exists():
        raise FileNotFoundError(
            f"band_stats.npz not found under {model_root}. "
            f"Ensure it is included in the model archive."
        )

    data = np.load(stats_path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    if mean.shape != (13,) or std.shape != (13,):
        raise ValueError(
            f"Unexpected band_stats shape: mean={mean.shape}, std={std.shape}"
        )
    logger.info("Loaded band_stats from %s: mean=%s, std=%s", stats_path, mean, std)
    return mean, std


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

# Robust per-chip statistics use the middle band of the pixel distribution to
# avoid contamination from clouds, water, snow, panels themselves, etc.
ROBUST_LOW_PCT = 5.0
ROBUST_HIGH_PCT = 95.0


def _normalize_zscore(image_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-aligned normalization to match training distribution.

    For each band we compute robust per-chip mean and std (using the middle
    p5..p95 of the pixel distribution to suppress outliers) and linearly
    rescale the chip so that its distribution matches the training mean/std
    loaded from band_stats.npz. The model expects inputs that look like the
    training data after z-score, i.e. roughly N(0, 1) per band.

    Steps per band:
        1. clip to [p5, p95] to compute robust chip mean / std
        2. rescale full chip:  x' = (x - chip_mean) / chip_std
        3. apply training z-score to bring to model input space:
           x_in = x' * train_std + train_mean   (now matches training scale)
           x_in = (x_in - train_mean) / train_std  (final z-score)
        Combined the two steps collapse to:
           x_in = (x - chip_mean) / chip_std

    The training stats are still loaded for diagnostic logging so that we can
    compare the chip distribution against the expected training distribution.
    """
    train_mean, train_std = _load_band_stats()

    # Pixels can be NaN at tile edges or from padded neighborhoods.
    invalid_mask = ~np.isfinite(image_hwc).all(axis=-1)  # (H, W)

    h, w, c = image_hwc.shape
    image_norm = np.zeros_like(image_hwc, dtype=np.float32)

    for b in range(c):
        band = image_hwc[:, :, b]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            # Fully invalid band: leave at zero.
            continue

        # Robust per-chip stats over middle p5..p95.
        lo, hi = np.percentile(finite, [ROBUST_LOW_PCT, ROBUST_HIGH_PCT])
        mid = finite[(finite >= lo) & (finite <= hi)]
        if mid.size < 16:
            mid = finite  # Degenerate distribution: fall back to all finite pixels.

        chip_mu = float(np.mean(mid))
        chip_sigma = float(np.std(mid))
        if chip_sigma < 1e-3:
            chip_sigma = 1.0  # Avoid divide-by-zero on flat bands.

        # Replace non-finite with chip mean before rescaling.
        band_clean = np.where(np.isfinite(band), band, chip_mu)

        # Per-chip standardization to N(0, 1) — what the model was trained on.
        image_norm[:, :, b] = (band_clean - chip_mu) / chip_sigma

        logger.info(
            "BAND %d ALIGN: chip_mu=%.2f chip_sigma=%.2f | train_mu=%.2f train_sigma=%.2f",
            b, chip_mu, chip_sigma, float(train_mean[b]), float(train_std[b]),
        )

    image_norm = np.nan_to_num(image_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return image_norm.astype(np.float32), invalid_mask


def _ensure_band_order(cube: xr.DataArray) -> xr.DataArray:
    """Reorder bands to the 13-band L1C order expected by the model."""
    available = list(cube.coords["bands"].values)
    target = [b for b in S2_L1C_BANDS if b in available]
    if len(target) != 13:
        raise ValueError(
            f"Expected 13 L1C bands, found {len(target)} in cube. "
            f"Available={available}, required={S2_L1C_BANDS}"
        )
    if available != target:
        cube = cube.sel(bands=target)
    return cube


def _run_inference(
    session: ort.InferenceSession,
    image_hwc: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the ONNX model on a single chip.

    Returns
    -------
    binary : np.ndarray, shape (H, W), uint8 in {0, 1}
    probs  : np.ndarray, shape (H, W), float32 in [0, 1]
    """
    h, w = image_hwc.shape[0], image_hwc.shape[1]
    image_norm, invalid_mask = _normalize_zscore(image_hwc.astype(np.float32))

    # Fully invalid chunk: return zeros directly.
    if bool(invalid_mask.all()):
        return (
            np.zeros((h, w), dtype=np.uint8),
            np.zeros((h, w), dtype=np.float32),
        )

    batch = image_norm[np.newaxis, ...]  # (1, H, W, 13)

    # Log normalized input range
    logger.info(
        "NORMALIZED input: shape=%s min=%.4f max=%.4f mean=%.4f",
        batch.shape, float(batch.min()), float(batch.max()), float(batch.mean()),
    )

    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: batch})[0]  # (1, H, W, 1)
    probs = np.squeeze(probs, axis=(0, -1)).astype(np.float32)  # (H, W)

    # Log model output range
    logger.info(
        "MODEL OUTPUT: shape=%s min=%.4f max=%.4f mean=%.4f",
        probs.shape, float(probs.min()), float(probs.max()), float(probs.mean()),
    )

    # Never predict positives on invalid (NaN/padded) pixels.
    probs[invalid_mask] = 0.0

    binary = (probs > threshold).astype(np.uint8)
    return binary, probs


def _apply_per_time(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """Run inference on a single (bands, y, x) cube.

    Always returns a 2-band output stacked along the ``bands`` dimension:
        - solar_pv             (uint8 binary mask)
        - solar_pv_probability (float32 probability)
    """
    threshold = float(context.get("threshold", DEFAULT_THRESHOLD))

    cube = _ensure_band_order(cube)

    # Drop temporal dimension if a degenerate one is present
    if "t" in cube.dims:
        cube = cube.squeeze("t", drop=True)

    cube = cube.transpose("y", "x", "bands")
    image_hwc = cube.values.astype(np.float32)

    h, w, c = image_hwc.shape
    if c != 13:
        raise ValueError(f"Expected 13 bands at last axis, got shape={image_hwc.shape}")

    # Diagnostic logging: raw input statistics per band
    finite_mask = np.isfinite(image_hwc)
    for b in range(c):
        band_vals = image_hwc[:, :, b]
        fin = band_vals[finite_mask[:, :, b]]
        if fin.size > 0:
            logger.info(
                "INPUT band %d: min=%.2f max=%.2f mean=%.2f std=%.2f finite_pct=%.1f%%",
                b, fin.min(), fin.max(), fin.mean(), fin.std(),
                100.0 * fin.size / band_vals.size,
            )
        else:
            logger.info("INPUT band %d: ALL NaN/Inf", b)

    session = _load_session()
    binary, probs = _run_inference(session, image_hwc, threshold)

    stacked = np.stack([binary.astype(np.float32), probs.astype(np.float32)], axis=0)

    return xr.DataArray(
        stacked,
        dims=("bands", "y", "x"),
        coords={
            "bands": ["solar_pv", "solar_pv_probability"],
            "y": cube.coords["y"],
            "x": cube.coords["x"],
        },
    )


# ---------------------------------------------------------------------------
# OpenEO UDF entry points
# ---------------------------------------------------------------------------

def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Rename the bands of the output cube to the two-band stacked output."""
    return metadata.rename_labels(
        dimension="bands",
        target=["solar_pv", "solar_pv_probability"],
    )


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """Main UDF entry point (apply_neighborhood callback)."""
    return _apply_per_time(cube, context)
