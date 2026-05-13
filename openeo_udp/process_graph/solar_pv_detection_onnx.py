"""Process graph helper: solar PV detection via ONNX UDF on CDSE.

Builds a Sentinel-2 L1C cloud-masked temporal mosaic and applies the
solar PV ONNX inference UDF in 256x256 spatial chunks via
``apply_neighborhood``.

Example
-------
    import openeo
    from openeo_udp.process_graph.solar_pv_detection_onnx import (
        build_solar_pv_detection_onnx,
        DEFAULT_JOB_OPTIONS,
    )

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()

    cube = build_solar_pv_detection_onnx(
        connection=conn,
        spatial_extent={"west": -1.85, "south": 51.54,
                        "east": -1.82, "north": 51.56, "crs": "EPSG:4326"},
        temporal_extent=["2024-05-01", "2024-07-31"],
        threshold=0.80,
    )
    cube.execute_batch(
        outputfile="solar_pv.tif",
        out_format="GTiff",
        job_options=DEFAULT_JOB_OPTIONS,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import openeo

# ---------------------------------------------------------------------------
# Default UDF dependency archives.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/apex_pv_rui/solar_pv_rui.zip#onnx_models"
)
DEFAULT_ONNX_DEPS_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/onnx_deps_python311.zip#onnx_deps"
)

DEFAULT_JOB_OPTIONS: dict = {
    "udf-dependency-archives": [
        DEFAULT_ONNX_DEPS_ARCHIVE_URL,
        DEFAULT_MODEL_ARCHIVE_URL,
    ],
    # Reasonable defaults for a 256x256 / 13-band U-Net (~50M params)
    "executor-memory": "4g",
    "executor-memoryOverhead": "2g",
    "python-memory": "disable",
    "soft-errors": 0.1,
}

S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

UDF_PATH = Path(__file__).resolve().parent.parent / "udf" / "solar_pv_inference_onnx.py"

# Expected output band order from the ONNX UDF.
UDF_OUTPUT_BANDS = [
    "solar_pv",
    "solar_pv_probability",
    "pre_norm_mean",
    "post_norm_mean",
]


def _load_l1c_cube(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    target_resolution: int = 10,
    target_crs: str = "EPSG:3857",
) -> openeo.DataCube:
    """Load 13-band Sentinel-2 L1C cube at 10 m in EPSG:3857."""
    s2 = connection.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=S2_L1C_BANDS,
        max_cloud_cover =  75
    )
    s2 = s2.resample_spatial(resolution=target_resolution, projection=target_crs)
    return s2


def _build_temporal_median_mosaic(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
) -> openeo.DataCube:
    """Cloud-masked temporal-median composite (SCL-based mask).

    Robust starter mosaic that works on any backend exposing SCL. For a
    closer match to the GEE training pipeline, swap this with the SLIC
    UDF in ``openeo_udp/udf/temporal_mosaic.py``.
    """
    l1c = _load_l1c_cube(connection, spatial_extent, temporal_extent)

    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover =  75
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    mask = scl.process(
        "to_scl_dilation_mask", 
        data=scl,
        kernel1_size=17,
        kernel2_size=77,
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3
    )
    
    # Create a cloud-free mosaic
    masked = l1c.mask(mask)

    return masked.reduce_dimension(dimension="t", reducer="median")


THRESHOLD = 0.80


def _extract_band_vector(result: Any, *, label: str) -> list[float]:
    """Extract a 13-value band vector from an openEO execute() result."""
    if isinstance(result, dict):
        if all(b in result for b in S2_L1C_BANDS):
            return [float(result[b]) for b in S2_L1C_BANDS]
        raise ValueError(
            f"Unsupported {label} result keys: {list(result.keys())}. "
            f"Expected band keys {S2_L1C_BANDS}."
        )

    if isinstance(result, (list, tuple)) and len(result) == 13:
        return [float(v) for v in result]

    raise ValueError(
        f"Unsupported {label} result type: {type(result)} value={result!r}"
    )


def _compute_global_band_stats_from_mosaic(
    mosaic: openeo.DataCube,
) -> tuple[list[float], list[float]]:
    """Compute ROI-wide per-band mean/std from the same mosaic used for inference."""
    mean_cube = mosaic.reduce_dimension(dimension="x", reducer="mean").reduce_dimension(
        dimension="y", reducer="mean"
    )
    mean_sq_cube = (mosaic * mosaic).reduce_dimension(
        dimension="x", reducer="mean"
    ).reduce_dimension(
        dimension="y", reducer="mean"
    )

    mean_raw = mean_cube.execute()
    mean_sq_raw = mean_sq_cube.execute()

    global_mean = _extract_band_vector(mean_raw, label="global_mean")
    global_mean_sq = _extract_band_vector(mean_sq_raw, label="global_mean_sq")

    mean_arr = np.asarray(global_mean, dtype=np.float64)
    mean_sq_arr = np.asarray(global_mean_sq, dtype=np.float64)
    var_arr = np.maximum(mean_sq_arr - (mean_arr ** 2), 1e-12)
    global_std = np.sqrt(var_arr).astype(np.float64).tolist()
    return global_mean, global_std


def _gaussian_kernel(size: int, sigma: float) -> list[list[float]]:
    """Build a normalized 2D Gaussian kernel (odd size).

    Returns a plain Python nested list so the value is JSON-serializable
    when sent to the openEO backend via ``apply_kernel``.
    """
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    return k.astype(float).tolist()


def _smooth_probability_band(
    cube: openeo.DataCube,
    prob_band: str,
    binary_band: str,
    threshold: float,
    prob_band_index: int = 1,
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> openeo.DataCube:
    """Apply Gaussian smoothing to the probability band, re-derive binary.

    Pattern adapted from masolele/WAC ``smooth_probabilities_gaussian``:
    convolve the float probability output with a small Gaussian kernel via
    backend ``apply_kernel`` to reduce edge artifacts at chunk boundaries
    and small-scale speckle. The binary mask is then re-thresholded from
    the smoothed probabilities so the two output bands stay consistent.
    """
    kernel = _gaussian_kernel(kernel_size, sigma)

    # Use positional selection because some backends/clients keep source
    # collection metadata (B01..B12) after apply_neighborhood even if the UDF
    # emits new band labels.
    prob = cube.band(prob_band_index)

    prob = prob.apply_kernel(kernel=kernel)
    binary = (prob > threshold) * 1.0  # float band, 0.0 / 1.0

    # Re-stack as a 2-band cube with the original band names.
    smoothed = binary.merge_cubes(prob)
    smoothed = smoothed.rename_labels(
        dimension="bands",
        target=[binary_band, prob_band],
    )

    # Preserve optional diagnostic bands emitted by the UDF.
    for idx in (2, 3):
        try:
            extra = cube.band(idx)
        except ValueError:
            continue
        smoothed = smoothed.merge_cubes(extra)

    try:
        smoothed = smoothed.rename_labels(
            dimension="bands",
            target=UDF_OUTPUT_BANDS,
        )
    except ValueError:
        # If debug bands are absent, keep the 2-band names already assigned.
        pass
    return smoothed


def build_solar_pv_detection_onnx(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    udf_path: Optional[str] = None,
    normalization: str = "percentile",
) -> openeo.DataCube:
    """Build the full mosaic + ONNX inference cube.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    spatial_extent : dict with west/south/east/north[/crs]
    temporal_extent : [start, end] dates (ISO strings)
    udf_path : override the default UDF source path
    normalization : "percentile" (default, matches winning training run) or "zscore".
        Both modes use the band statistics in band_stats.npz from the model archive.
    """
    mosaic = _build_temporal_median_mosaic(connection, spatial_extent, temporal_extent)

    if normalization not in ("percentile", "zscore"):
        raise ValueError(
            f"Unknown normalization mode '{normalization}'. "
            f"Supported: 'percentile', 'zscore'."
        )

    udf_src_path = Path(udf_path) if udf_path else UDF_PATH
    udf_code = udf_src_path.read_text(encoding="utf-8")

    # apply_neighborhood feeds the UDF a chunk of shape (size + 2*overlap).
    # Model expects fixed 256x256, so size=192 + 2*32 = 256 ✓.
    udf_context: dict = {
        "threshold": THRESHOLD,
        "normalization": normalization,
    }

    inferred = mosaic.apply_neighborhood(
        process=openeo.UDF(
            udf_code,
            runtime="Python",
            context=udf_context,
        ),
        size=[
            {"dimension": "x", "value": 192, "unit": "px"},
            {"dimension": "y", "value": 192, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 32, "unit": "px"},
            {"dimension": "y", "value": 32, "unit": "px"},
        ],
    )

    # Debug mode simplification: skip Gaussian smoothing and return direct
    # UDF output to avoid downstream band-selection complexity.
    return inferred
