"""High-level API for temporal Sentinel-2 mosaicing."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import ee

from .assignment import assign_clusters, hierarchical_split_and_assign
from .cloud_mask import attach_cloud_prob, compute_clear_mask, load_sentinel_l1c
from .io_helpers import (
    DEFAULT_DRIVE_FOLDER,
    default_export_name,
    export_composite,
    initialize_ee,
)
from .resample_align import S2_L1C_13_BANDS, align_l1c_bands
from .scene_scoring import score_scenes
from .segmentation import build_reference_image, run_snic
from .stitching import build_composite_from_assignments, feather_and_blend

LOGGER = logging.getLogger(__name__)


def create_aoi(
    center_lat: float,
    center_lon: float,
    half_size_km: float = 5.0,
) -> ee.Geometry:
    """Create a square AOI centered at lat/lon with side length 2*half_size_km."""
    proj_3857 = ee.Projection("EPSG:3857")
    center = ee.Geometry.Point([center_lon, center_lat])
    aoi_3857 = center.buffer(
        distance=half_size_km * 1000,
        proj=proj_3857,
    ).bounds(proj=proj_3857)
    return aoi_3857


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if end <= start:
        raise ValueError("end_date must be greater than start_date.")


def _build_rescue_fill(
    scored: ee.ImageCollection,
    aoi: ee.Geometry,
    out_scale: int,
    rescue_count: int,
) -> Dict[str, object]:
    """Build per-pixel rescue fill from lowest cloud probability candidates."""
    rescue_candidates = scored.limit(rescue_count, "score", False)
    rescue_list = [ee.Image(rescue_candidates.toList(rescue_count).get(i)) for i in range(rescue_count)]

    layers = []
    for scene in rescue_list:
        aligned = align_l1c_bands(
            scene,
            aoi=aoi,
            out_scale=out_scale,
            out_crs="EPSG:3857",
        )
        cloud_prob = (
            scene.select("cloud_prob")
            .clip(aoi)
            .resample("bilinear")
            .reproject(crs="EPSG:3857", scale=out_scale)
            .rename("rescue_cloud_prob")
        )
        layers.append(
            cloud_prob.multiply(-1).rename("cp_score")
            .addBands(cloud_prob)
            .addBands(aligned)
        )

    best = ee.ImageCollection(layers).qualityMosaic("cp_score")
    return {
        "rescue_spectral": best.select(S2_L1C_13_BANDS),
        "rescue_cloud_prob": best.select("rescue_cloud_prob"),
        "rescue_count": rescue_count,
    }


def create_temporal_mosaic(
    center_lat: float,
    center_lon: float,
    start_date: str,
    end_date: str,
    aoi_size_km: float = 10.0,
    out_scale: int = 10,
    patch_size_px: int = 64,
    snic_size_px: int = 40,
    clear_thresh: float = 0.8,
    top_n_scenes: int = 8,
    top_n_scenes_rescue: int = 10,
    feather_px: int = 12,
    min_patch_px: int = 8,
    export_target: str = "drive",
    export_name: Optional[str] = None,
    max_scene_cloud_pct: int | None = None,
    use_shadow_mask: bool = False,
    drive_folder: Optional[str] = None,
    export_rgb: bool = True,
    aoi_bounds_3857: tuple | None = None,
) -> Dict[str, object]:
    """Create and export a temporal Sentinel-2 L1C mosaic.

    Outputs:
    - 13 spectral L1C bands at 10 m
    - ``source_scene_id`` and ``source_date`` QA bands

    Parameters
    ----------
    aoi_bounds_3857:
        Optional (xmin, ymin, xmax, ymax) in EPSG:3857.  When provided the AOI
        is built directly from these bounds, guaranteeing pixel-grid alignment
        and exact output dimensions.  Takes precedence over center_lat/lon +
        aoi_size_km.
    """
    _validate_dates(start_date, end_date)

    initialize_ee()

    if aoi_bounds_3857 is not None:
        xmin, ymin, xmax, ymax = aoi_bounds_3857
        aoi = ee.Geometry.Rectangle(
            [xmin, ymin, xmax, ymax],
            proj=ee.Projection("EPSG:3857"),
            evenOdd=False,
        )
        LOGGER.info("Using exact chip bounds AOI: %s", aoi_bounds_3857)
    else:
        if aoi_size_km <= 0:
            raise ValueError("aoi_size_km must be > 0.")
        LOGGER.info("Building AOI around (%s, %s).", center_lat, center_lon)
        aoi = create_aoi(
            center_lat=center_lat,
            center_lon=center_lon,
            half_size_km=aoi_size_km / 2.0,
        )

    LOGGER.info("Loading Sentinel-2 L1C and cloud probability collections.")
    s2 = load_sentinel_l1c(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_scene_cloud_pct=max_scene_cloud_pct,
    )
    s2 = attach_cloud_prob(s2)
    s2 = s2.map(lambda img: compute_clear_mask(img, aoi=aoi, use_shadow=use_shadow_mask))

    scored = score_scenes(
        collection=s2,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
    )

    scene_count = int(scored.size().getInfo())
    if scene_count == 0:
        raise RuntimeError("No Sentinel-2 scenes found for the requested AOI/time window.")

    n = min(top_n_scenes, scene_count)
    rescue_count = min(max(1, top_n_scenes_rescue), scene_count)
    LOGGER.info("Scored %d scenes; using top %d candidates.", scene_count, n)
    candidates = scored.limit(n, "score", False)
    candidate_list = [ee.Image(candidates.toList(n).get(i)) for i in range(n)]

    LOGGER.info("Running SNIC segmentation (size=%d px).", snic_size_px)
    reference = build_reference_image(candidates=candidates, aoi=aoi)
    clusters = run_snic(reference=reference, size_px=snic_size_px).clip(aoi)

    LOGGER.info("Assigning scenes to SNIC clusters (clear_thresh=%s).", clear_thresh)
    initial_assign = assign_clusters(
        clusters=clusters,
        candidate_list=candidate_list,
        clear_thresh=clear_thresh,
    )

    LOGGER.info(
        "Running hierarchical fallback split (patch=%d -> min=%d px).",
        patch_size_px,
        min_patch_px,
    )
    final_assign = hierarchical_split_and_assign(
        source_scene_id=initial_assign["source_scene_id"],
        source_date=initial_assign["source_date"],
        unassigned_mask=initial_assign["unassigned_mask"],
        candidate_list=candidate_list,
        aoi=aoi,
        patch_size_px=patch_size_px,
        min_patch_px=min_patch_px,
        clear_thresh=clear_thresh,
        out_scale=out_scale,
    )

    source_scene_id = (
        final_assign["source_scene_id"]
        .clip(aoi)
        .reproject(crs="EPSG:3857", scale=out_scale)
        .toInt16()
    )
    source_date = (
        final_assign["source_date"]
        .clip(aoi)
        .reproject(crs="EPSG:3857", scale=out_scale)
        .toInt32()
    )

    LOGGER.info("Stitching spectral bands and applying seam feathering.")
    spectral, assigned_valid = build_composite_from_assignments(
        candidate_list=candidate_list,
        source_scene_id=source_scene_id,
        aoi=aoi,
        out_scale=out_scale,
    )
    spectral = feather_and_blend(
        composite=spectral,
        source_scene_id=source_scene_id,
        out_scale=out_scale,
        feather_px=feather_px,
    )

    primary_valid = spectral.mask().reduce(ee.Reducer.min()).rename("primary_valid")
    rescue_needed = primary_valid.Not().rename("rescue_needed")
    rescue = _build_rescue_fill(
        scored=scored,
        aoi=aoi,
        out_scale=out_scale,
        rescue_count=rescue_count,
    )
    spectral = spectral.unmask(rescue["rescue_spectral"])
    fill_mode = rescue_needed.toFloat().rename("fill_mode")
    rescue_cloud_prob = (
        ee.Image(rescue["rescue_cloud_prob"])
        .updateMask(rescue_needed)
        .unmask(-1)
        .rename("rescue_cloud_prob")
    )

    output = (
        spectral.toFloat()
        .addBands(source_scene_id.toFloat().rename("source_scene_id"))
        .addBands(source_date.toFloat().rename("source_date"))
        .addBands(assigned_valid.toFloat().rename("assigned_valid_mask"))
        .addBands(fill_mode)
        .addBands(rescue_cloud_prob.toFloat())
        .clip(aoi)
    )

    final_export_name = export_name or default_export_name(
        center_lat=center_lat,
        center_lon=center_lon,
        start_date=start_date,
        end_date=end_date,
    )
    _drive_folder = drive_folder if drive_folder is not None else DEFAULT_DRIVE_FOLDER

    # When exact bounds are provided, pin the export to the chip pixel grid so
    # the output is always exactly (xmax-xmin)/scale × (ymax-ymin)/scale pixels.
    _crs_transform, _dimensions = None, None
    if aoi_bounds_3857 is not None:
        xmin, ymin, xmax, ymax = aoi_bounds_3857
        _crs_transform = [out_scale, 0, xmin, 0, -out_scale, ymax]
        w = round((xmax - xmin) / out_scale)
        h = round((ymax - ymin) / out_scale)
        _dimensions = f"{w}x{h}"

    task = export_composite(
        image=output,
        aoi=aoi,
        export_target=export_target,
        export_name=final_export_name,
        scale=out_scale,
        drive_folder=_drive_folder,
        crs="EPSG:3857",
        crs_transform=_crs_transform,
        dimensions=_dimensions,
    )
    task_status = task.status()
    result = {
        "task": task,
        "task_id": task_status.get("id"),
        "task_state": task_status.get("state"),
        "export_name": final_export_name,
        "drive_folder": _drive_folder,
        "candidate_scene_count": n,
        "rescue_scene_count": rescue_count,
        "aoi_size_km": aoi_size_km,
        "max_scene_cloud_pct": max_scene_cloud_pct,
        "use_shadow_mask": use_shadow_mask,
        "aoi": aoi,
    }

    if export_rgb:
        rgb_vis = (
            spectral.select(["B4", "B3", "B2"])
            .unitScale(0, 3000)
            .clamp(0, 1)
            .multiply(255)
            .toUint8()
        )
        rgb_task = export_composite(
            image=rgb_vis,
            aoi=aoi,
            export_target=export_target,
            export_name=f"{final_export_name}_rgb",
            scale=out_scale,
            drive_folder=_drive_folder,
            crs="EPSG:3857",
        )
        rgb_task_status = rgb_task.status()
        result.update({
            "rgb_task": rgb_task,
            "rgb_task_id": rgb_task_status.get("id"),
            "rgb_task_state": rgb_task_status.get("state"),
        })

    return result
