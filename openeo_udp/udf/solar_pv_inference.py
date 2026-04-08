"""OpenEO UDF: Solar PV panel detection using U-Net.

This UDF is designed to be called within an OpenEO process graph via
`run_udf()`.  It receives a Sentinel-2 L1C datacube (13 bands, 256x256
spatial), normalises inputs using pre-computed band statistics, runs
inference through a trained U-Net model, and returns a binary detection
mask.

The model architecture and weights are decoupled via model_registry.yaml
so that the UDF code never needs to change when a new model version is
deployed.  Only the registry file is updated.

OpenEO UDF entry point: apply_datacube()

Runtime requirements (see requirements.txt):
    tensorflow >= 2.15
    numpy
    h5py
    pyyaml
    requests
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache directory for downloaded model artifacts.  On OpenEO backends this
# is typically /tmp.  Artifacts are downloaded once and reused across chunks.
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(tempfile.gettempdir()) / "solar_pv_model_cache"

# Expected Sentinel-2 L1C band order (must match training data).
S2_L1C_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


# ===================================================================
# Registry loading
# ===================================================================

def _find_registry() -> dict[str, Any]:
    """Locate and parse model_registry.yaml.

    Search order:
      1. Same directory as this UDF file
      2. Parent directory (openeo_udp/)
    """
    udf_dir = Path(__file__).resolve().parent
    for candidate in [udf_dir / "model_registry.yaml",
                      udf_dir.parent / "model_registry.yaml"]:
        if candidate.exists():
            with candidate.open("r") as f:
                return yaml.safe_load(f)["active_model"]
    raise FileNotFoundError(
        "model_registry.yaml not found. Expected next to UDF or in openeo_udp/."
    )


def _load_registry(context: dict | None = None) -> dict[str, Any]:
    """Load model configuration from registry, with optional context overrides.

    OpenEO can pass parameters via the ``context`` dict in run_udf().
    Supported context overrides:
        - threshold: float (override default inference threshold)
        - weights_url: str (override model weights URL)
        - band_stats_url: str (override band stats URL)
    """
    registry = _find_registry()
    if context:
        for key in ("threshold", "weights_url", "band_stats_url"):
            if key in context:
                registry[key] = context[key]
    return registry


# ===================================================================
# Artifact download / cache
# ===================================================================

def _url_cache_path(url: str, suffix: str = "") -> Path:
    """Deterministic local cache path for a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    name = url.rsplit("/", 1)[-1] if "/" in url else url_hash
    return _CACHE_DIR / f"{url_hash}_{name}{suffix}"


def _download_if_needed(url: str, local_fallback: str | None = None) -> Path:
    """Download a file from URL, or use local fallback if available."""
    # Try local fallback first (development / testing on JASMIN)
    if local_fallback:
        repo_root = Path(__file__).resolve().parent.parent.parent
        local_path = repo_root / local_fallback
        if local_path.exists():
            logger.info("Using local artifact: %s", local_path)
            return local_path

    cached = _url_cache_path(url)
    if cached.exists():
        logger.info("Using cached artifact: %s", cached)
        return cached

    logger.info("Downloading: %s", url)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import requests
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    tmp = cached.with_suffix(".tmp")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp.rename(cached)
    logger.info("Downloaded to: %s", cached)
    return cached


# ===================================================================
# Band statistics and normalisation
# ===================================================================

def _load_band_stats(path: Path) -> dict[str, np.ndarray]:
    """Load per-band mean/std from .npz file."""
    data = np.load(path)
    return {
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
    }


def normalize_zscore(images: np.ndarray, band_stats: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score normalisation: (pixel - mean) / std, per band.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W, C) or (H, W, C), raw Sentinel-2 reflectance.
    band_stats : dict
        Must contain 'mean' and 'std' arrays of shape (C,).

    Returns
    -------
    Normalised array, same shape and dtype as input.
    """
    mean = band_stats["mean"]
    std = band_stats["std"]
    if images.ndim == 4:
        return (images - mean[None, None, None, :]) / std[None, None, None, :]
    return (images - mean[None, None, :]) / std[None, None, :]


# ===================================================================
# Model building and weight loading
# ===================================================================

def _build_model(registry: dict[str, Any]):
    """Build U-Net model from registry parameters.

    Imports tensorflow lazily so the module can be parsed without TF
    installed (e.g. for linting or registry inspection).
    """
    import tensorflow as tf

    # Import from project if available, otherwise inline minimal builder
    try:
        import sys
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from solar_ml.model import build_unet_model
        model, _ = build_unet_model(
            input_shape=tuple(registry["input_shape"]),
            backbone_name=registry["backbone"],
            pretrained=True,
            decoder_filters=tuple(registry["decoder_filters"]),
            decoder_dropout=float(registry["decoder_dropout"]),
            attention=bool(registry.get("attention", False)),
            se=bool(registry.get("se", False)),
        )
        return model
    except ImportError:
        raise ImportError(
            "solar_ml package not found. Ensure the repo root is on PYTHONPATH "
            "or install solar_ml as a package."
        )


def _load_weights_compat(model, weights_path: Path) -> None:
    """Load weights from a Keras ModelCheckpoint H5 file.

    Keras ModelCheckpoint saves weights using class-based H5 group names
    (e.g. 'conv2d', 'batch_normalization') rather than layer instance names
    (e.g. 'rgb_adapter').  model.load_weights() fails when rebuilding a
    fresh model because it looks for instance names.  This function reads
    the H5 directly, maps each layer to its group by class-name + sequential
    counter, and sets weights explicitly.
    """
    import tensorflow as tf

    def _h5_class_key(layer: tf.keras.layers.Layer) -> str:
        name = type(layer).__name__
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        return name.lower()

    def _assign_from_group(layers, h5_group, class_ctr=None):
        if class_ctr is None:
            class_ctr = {}
        for layer in layers:
            cls_key = _h5_class_key(layer)
            cnt = class_ctr.get(cls_key, 0)
            h5_key = cls_key if cnt == 0 else f"{cls_key}_{cnt}"
            class_ctr[cls_key] = cnt + 1
            if h5_key not in h5_group:
                continue
            grp = h5_group[h5_key]
            if "layers" in grp:
                _assign_from_group(layer.layers, grp["layers"])
            if layer.weights and "vars" in grp and len(grp["vars"]) > 0:
                n = len(grp["vars"])
                values = [grp["vars"][str(i)][:] for i in range(n)]
                layer.set_weights(values)

    with h5py.File(str(weights_path), "r") as f:
        _assign_from_group(model.layers, f["layers"])


# ===================================================================
# Singleton model holder (avoids reloading on every chunk)
# ===================================================================

_MODEL = None
_BAND_STATS = None
_REGISTRY = None


def _get_model_and_stats(context: dict | None = None):
    """Load model and band stats once, reuse across invocations."""
    global _MODEL, _BAND_STATS, _REGISTRY

    if _MODEL is not None and _BAND_STATS is not None:
        return _MODEL, _BAND_STATS, _REGISTRY

    registry = _load_registry(context)
    _REGISTRY = registry

    # Download / locate artifacts
    weights_path = _download_if_needed(
        registry["weights_url"],
        local_fallback=registry.get("weights_local"),
    )
    stats_path = _download_if_needed(
        registry["band_stats_url"],
        local_fallback=registry.get("band_stats_local"),
    )

    # Build model and load weights
    logger.info("Building model: backbone=%s", registry["backbone"])
    model = _build_model(registry)
    logger.info("Loading weights from: %s", weights_path)
    _load_weights_compat(model, weights_path)

    # Load band stats
    band_stats = _load_band_stats(stats_path)
    logger.info("Band stats loaded: %d bands", len(band_stats["mean"]))

    _MODEL = model
    _BAND_STATS = band_stats
    return _MODEL, _BAND_STATS, _REGISTRY


# ===================================================================
# Inference
# ===================================================================

def predict_chip(
    image: np.ndarray,
    model=None,
    band_stats: dict | None = None,
    registry: dict | None = None,
    context: dict | None = None,
) -> np.ndarray:
    """Run inference on a single chip or batch.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 13) for single chip, or (N, H, W, 13) for batch.
        Raw Sentinel-2 L1C reflectance values (not normalised).
    model, band_stats, registry :
        Pre-loaded resources. If None, loaded from registry.
    context : dict
        OpenEO context overrides (threshold, weights_url, etc.).

    Returns
    -------
    np.ndarray
        Binary mask, shape (H, W) or (N, H, W). Values 0 or 1.
    """
    if model is None or band_stats is None or registry is None:
        model, band_stats, registry = _get_model_and_stats(context)

    threshold = float(registry.get("threshold", 0.80))
    single = image.ndim == 3
    if single:
        image = image[np.newaxis, ...]

    # Normalise
    image_norm = normalize_zscore(image.astype(np.float32), band_stats)

    # Predict
    probs = model.predict(image_norm, verbose=0)  # (N, H, W, 1)
    mask = (probs[..., 0] > threshold).astype(np.uint8)  # (N, H, W)

    return mask[0] if single else mask


def predict_chip_probabilities(
    image: np.ndarray,
    model=None,
    band_stats: dict | None = None,
    registry: dict | None = None,
    context: dict | None = None,
) -> np.ndarray:
    """Run inference and return sigmoid probabilities (no thresholding).

    Returns
    -------
    np.ndarray
        Probability map, shape (H, W) or (N, H, W). Values in [0, 1].
    """
    if model is None or band_stats is None or registry is None:
        model, band_stats, registry = _get_model_and_stats(context)

    single = image.ndim == 3
    if single:
        image = image[np.newaxis, ...]

    image_norm = normalize_zscore(image.astype(np.float32), band_stats)
    probs = model.predict(image_norm, verbose=0)
    result = probs[..., 0].astype(np.float32)
    return result[0] if single else result


# ===================================================================
# OpenEO UDF entry point
# ===================================================================

def apply_datacube(cube, context: dict) -> "XarrayDataCube":
    """OpenEO UDF entry point.

    Receives a datacube with dimensions (bands, y, x) representing a
    single 256x256 chip of Sentinel-2 L1C data (13 bands).

    The datacube is normalised, passed through the model, and the output
    is a single-band binary detection mask.

    Parameters
    ----------
    cube : openeo.udf.XarrayDataCube
        Input datacube. Expected shape: (13, 256, 256) — bands first.
    context : dict
        OpenEO context dict, may contain overrides:
            threshold (float): Detection threshold override.

    Returns
    -------
    openeo.udf.XarrayDataCube
        Single-band datacube with binary detection mask.
    """
    import xarray as xr
    from openeo.udf import XarrayDataCube

    array = cube.get_array()  # xarray.DataArray

    # Determine array layout and convert to (N, H, W, C) numpy
    dims = list(array.dims)

    if "t" in dims or "time" in dims:
        # If there's a time dimension, squeeze or iterate
        time_dim = "t" if "t" in dims else "time"
        if array.sizes[time_dim] == 1:
            array = array.squeeze(time_dim)
            dims = list(array.dims)

    # Identify band and spatial dimensions
    band_dim = None
    for candidate in ("bands", "band", "spectral"):
        if candidate in dims:
            band_dim = candidate
            break

    if band_dim is None:
        raise ValueError(f"Cannot find band dimension in {dims}. "
                         f"Expected one of: bands, band, spectral.")

    # Transpose to (y, x, bands) = (H, W, C)
    spatial_dims = [d for d in dims if d != band_dim]
    data = array.transpose(*spatial_dims, band_dim).values  # (H, W, C) or (T, H, W, C)

    if data.ndim == 3:
        # Single chip: (H, W, C)
        image = data.astype(np.float32)
    elif data.ndim == 4:
        # Batch: (N, H, W, C)
        image = data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected data shape after transpose: {data.shape}")

    # Run inference
    model, band_stats, registry = _get_model_and_stats(context)
    threshold = float(context.get("threshold", registry.get("threshold", 0.80)))

    single = image.ndim == 3
    if single:
        image = image[np.newaxis, ...]

    image_norm = normalize_zscore(image, band_stats)
    probs = model.predict(image_norm, verbose=0)  # (N, H, W, 1)
    mask = (probs[..., 0] > threshold).astype(np.float32)  # (N, H, W)

    if single:
        mask = mask[0]  # (H, W)

    # Build output xarray DataArray
    y_dim = spatial_dims[0]
    x_dim = spatial_dims[1] if len(spatial_dims) > 1 else spatial_dims[0]

    result = xr.DataArray(
        mask,
        dims=[y_dim, x_dim] if single else ["batch", y_dim, x_dim],
        coords={
            y_dim: array.coords[y_dim] if y_dim in array.coords else np.arange(mask.shape[-2]),
            x_dim: array.coords[x_dim] if x_dim in array.coords else np.arange(mask.shape[-1]),
        },
    )

    return XarrayDataCube(result)
