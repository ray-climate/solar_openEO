"""I/O helpers for Earth Engine initialization and exports."""

from __future__ import annotations

import logging
import re
from typing import Optional

import ee

LOGGER = logging.getLogger(__name__)

DEFAULT_DRIVE_FOLDER = "solar_openEO_temporal_mosaics"


def initialize_ee(project: Optional[str] = None) -> None:
    """Initialize Earth Engine for the active authenticated account."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "Failed to initialize Earth Engine. Authenticate first with "
            "`earthengine authenticate` and verify API access."
        ) from exc


def sanitize_export_name(export_name: str, max_len: int = 95) -> str:
    """Normalize an export name to Earth-Engine-safe characters."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", export_name).strip("_")
    if not clean:
        clean = "s2_temporal_mosaic"
    return clean[:max_len]


def default_export_name(
    center_lat: float,
    center_lon: float,
    start_date: str,
    end_date: str,
) -> str:
    """Create a deterministic export name for the mosaic."""
    return sanitize_export_name(
        f"s2_temporal_mosaic_{start_date}_{end_date}_{center_lat:.5f}_{center_lon:.5f}"
    )


def export_composite(
    image: ee.Image,
    aoi: ee.Geometry,
    export_target: str,
    export_name: str,
    scale: int,
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    crs: str = "EPSG:3857",
) -> ee.batch.Task:
    """Start an export task for a composite image.

    Parameters
    ----------
    image:
        Image to export.
    aoi:
        Export region.
    export_target:
        Export destination. v1 supports ``drive``.
    export_name:
        Task description and file prefix.
    scale:
        Pixel size in meters.
    drive_folder:
        Google Drive folder name for exports.
    """
    target = export_target.lower().strip()
    safe_name = sanitize_export_name(export_name)

    if target != "drive":
        raise NotImplementedError(
            "v1 currently supports export_target='drive' only."
        )

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=safe_name,
        folder=drive_folder,
        fileNamePrefix=safe_name,
        region=aoi,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
    )
    task.start()
    LOGGER.info(
        "Started Drive export task: %s (folder=%s)", safe_name, drive_folder
    )
    return task
