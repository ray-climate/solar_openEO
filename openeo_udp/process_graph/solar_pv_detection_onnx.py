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
from typing import Optional

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
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    scl_mask = (
        (scl.band("SCL") != 0)
        & (scl.band("SCL") != 1)
        & (scl.band("SCL") != 3)
        & (scl.band("SCL") != 8)
        & (scl.band("SCL") != 9)
        & (scl.band("SCL") != 10)
    )
    masked = l1c.mask(scl_mask)
    return masked.reduce_dimension(dimension="t", reducer="median")


THRESHOLD = 0.80


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

    prob = cube.band(prob_band).apply_kernel(kernel=kernel)
    binary = (prob > threshold) * 1.0  # float band, 0.0 / 1.0

    # Re-stack as a 2-band cube with the original band names.
    smoothed = binary.merge_cubes(prob)
    smoothed = smoothed.rename_labels(
        dimension="bands",
        target=[binary_band, prob_band],
    )
    return smoothed


def build_solar_pv_detection_onnx(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    udf_path: Optional[str] = None,
) -> openeo.DataCube:
    """Build the full mosaic + ONNX inference cube.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    spatial_extent : dict with west/south/east/north[/crs]
    temporal_extent : [start, end] dates (ISO strings)
    udf_path : override the default UDF source path

    Returns
    -------
    openeo.DataCube with two bands:
      - solar_pv (binary mask)
      - solar_pv_probability (float probability)
    """
    mosaic = _build_temporal_median_mosaic(connection, spatial_extent, temporal_extent)

    udf_src_path = Path(udf_path) if udf_path else UDF_PATH
    udf_code = udf_src_path.read_text(encoding="utf-8")

    # apply_neighborhood feeds the UDF a chunk of shape (size + 2*overlap).
    # Model expects fixed 256x256, so size=192 + 2*32 = 256 ✓.
    inferred = mosaic.apply_neighborhood(
        process=openeo.UDF(
            udf_code,
            runtime="Python",
            context={
                "threshold": THRESHOLD,
            },
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

    # Soft Gaussian smoothing on the probability band to suppress residual
    # tile-edge artifacts and salt-and-pepper noise (pattern from masolele/WAC).
    # We smooth probabilities (continuous) and re-derive the binary mask from
    # the smoothed probabilities so both output bands stay consistent.
    smoothed = _smooth_probability_band(
        inferred,
        prob_band="solar_pv_probability",
        binary_band="solar_pv",
        threshold=THRESHOLD,
        kernel_size=9,
        sigma=2.0,
    )
    return smoothed
