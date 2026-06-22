"""Process graph: solar PV detection (single merged mosaic+ONNX UDF).

Loads Sentinel-2 L1C + L2A SCL, stacks them into a 14-band cube, and runs
a single ``apply_neighborhood`` with the merged UDF
(``openeo_udp/udf/solar_pv_detection.py``) which performs SLIC mosaicing
and ONNX inference in one executor call. This avoids the intermediate
cube shuffle between two chained ``apply_neighborhood`` calls.

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

import openeo
import yaml

# ---------------------------------------------------------------------------
# UDF dependency archives.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ARCHIVE_URL = (
    "https://github.com/ray-climate/solar_openEO/releases/download/"
    "v2.0.0/openeo_dependencies.zip#onnx_models"
)
DEFAULT_ONNX_DEPS_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/onnx_deps_python311.zip#onnx_deps"
)

S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

UDF_PATH = Path(__file__).resolve().parent.parent / "udf" / "solar_pv_detection.py"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "model_registry.yaml"

# Chunk geometry: model needs exactly 256x256. size=192 + overlap=32 gives
# a 256-pixel chunk to the UDF and a 32 px halo for seam-free inference.
CHUNK_INNER_PX = 192
CHUNK_OVERLAP_PX = 32

# Temporal neighborhood in apply_neighborhood. This lets the UDF receive
CHUNK_TEMPORAL_WINDOW = "P90D"


def _load_active_model_config(registry_path: Path = REGISTRY_PATH) -> dict:
    with registry_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw or "active_model" not in raw:
        raise ValueError(f"Missing 'active_model' in {registry_path}")
    return raw["active_model"]


def _build_job_options(model_config: Optional[dict] = None) -> dict:
    """Build default job options with registry-aware dependency archives.

    If active_model.model_archive_url is set, use it as a fallback for legacy
    archive-based runs. Otherwise we rely on UDF context URLs for ONNX and
    band stats retrieval.
    """
    active_model = dict(model_config or _load_active_model_config())
    archives = [DEFAULT_ONNX_DEPS_ARCHIVE_URL]

    # Model artifact archive: prefer registry-specified URL, fall back to default.
    model_archive_url = active_model.get("model_archive_url") or DEFAULT_MODEL_ARCHIVE_URL
    archives.append(str(model_archive_url))

    return {
        "udf-dependency-archives": archives,
        # Merged UDF: mosaic (SLIC + numpy) + ONNX U-Net per 256x256 chunk.
        # Allow more headroom than a pure-inference run.
        "executor-memory": "6g",
        "executor-memoryOverhead": "3g",
        "python-memory": "disable",
        "soft-errors": 0.1,
    }


DEFAULT_JOB_OPTIONS: dict = _build_job_options()


def _build_udf_context(model_config: Optional[dict] = None, threshold: Optional[float] = None) -> dict:
    active_model = dict(model_config or _load_active_model_config())
    context = {
        "model_config": {
            "name": active_model.get("name"),
            "version": active_model.get("version"),
            "normalization": active_model.get("normalization", "zscore"),
            "threshold": float(active_model.get("threshold", 0.80)),
        }
    }
    if threshold is not None:
        context["threshold"] = float(threshold)
    return context


def _load_l1c_cube(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    target_resolution: int = 10,
    target_crs: str = "EPSG:3857",
) -> openeo.DataCube:
    s2 = connection.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=S2_L1C_BANDS,
        max_cloud_cover=75,
    )
    return s2.resample_spatial(resolution=target_resolution, projection=target_crs)


def _load_scl_cube(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
) -> openeo.DataCube:
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=75,
    )
    return scl.resample_spatial(resolution=10, projection="EPSG:3857")


def build_solar_pv_detection_onnx(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    udf_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> openeo.DataCube:
    """Build the merged mosaic+ONNX inference cube.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    spatial_extent : dict with west/south/east/north[/crs]
    temporal_extent : [start, end] ISO date strings
    udf_path : optional override for the merged UDF source path
    """
    l1c = _load_l1c_cube(connection, spatial_extent, temporal_extent)
    scl = _load_scl_cube(connection, spatial_extent, temporal_extent)

    # 14-band cube: 13 L1C spectral + SCL.
    merged = l1c.merge_cubes(scl)

    udf_src_path = Path(udf_path) if udf_path else UDF_PATH
    udf_code = udf_src_path.read_text(encoding="utf-8")
    udf_context = _build_udf_context(threshold=threshold)

    # Single apply_neighborhood: chunk = inner (192) + 2*overlap (32) = 256.
    # The merged UDF mosaics the temporal stack and runs ONNX inference
    # in one executor call, avoiding the intermediate cube shuffle that
    # two chained apply_neighborhood calls would introduce.
    detected = merged.apply_neighborhood(
        process=openeo.UDF(
            udf_code,
            runtime="Python",
            context=udf_context,
        ),
        size=[
            {"dimension": "x", "value": CHUNK_INNER_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_INNER_PX, "unit": "px"},
            {"dimension": "t", "value": CHUNK_TEMPORAL_WINDOW},
        ],
        overlap=[
            {"dimension": "x", "value": CHUNK_OVERLAP_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_OVERLAP_PX, "unit": "px"},
            {"dimension": "t", "value": 0},
        ],
    )
    return detected
