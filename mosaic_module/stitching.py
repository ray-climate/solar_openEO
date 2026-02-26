"""Composite reconstruction and seam handling."""

from __future__ import annotations

from typing import Sequence, Tuple

import ee

from .resample_align import align_l1c_bands


def build_composite_from_assignments(
    candidate_list: Sequence[ee.Image],
    source_scene_id: ee.Image,
    aoi: ee.Geometry,
    out_scale: int = 10,
) -> Tuple[ee.Image, ee.Image]:
    """Build final spectral composite from assigned scene IDs.

    Returns
    -------
    composite:
        13-band spectral composite.
    assigned_valid_mask:
        Mask where assignment + clear pixels were available before fallback fill.
    """
    assigned_scene_id = (
        source_scene_id.clip(aoi)
        .reproject(crs="EPSG:3857", scale=out_scale)
        .toInt16()
    )

    assigned_pieces = []
    clear_median_inputs = []

    for idx, scene in enumerate(candidate_list):
        aligned = align_l1c_bands(
            scene,
            aoi=aoi,
            out_scale=out_scale,
            out_crs="EPSG:3857",
        )
        clear = (
            scene.select("clear_mask")
            .clip(aoi)
            .reproject(crs="EPSG:3857", scale=out_scale)
            .gt(0)
        )
        clear_median_inputs.append(aligned.updateMask(clear))

        piece = aligned.updateMask(assigned_scene_id.eq(idx)).updateMask(clear)
        assigned_pieces.append(piece)

    assigned_only = ee.ImageCollection(assigned_pieces).mosaic()
    assigned_valid_mask = assigned_only.mask().reduce(ee.Reducer.min()).rename(
        "assigned_valid_mask"
    )
    median_fill = ee.ImageCollection(clear_median_inputs).median()
    composite = assigned_only.unmask(median_fill).clip(aoi)
    return composite, assigned_valid_mask


def feather_and_blend(
    composite: ee.Image,
    source_scene_id: ee.Image,
    out_scale: int = 10,
    feather_px: int = 12,
) -> ee.Image:
    """Light seam feathering on assignment boundaries."""
    if feather_px <= 0:
        return composite

    src = (
        source_scene_id.reproject(crs="EPSG:3857", scale=out_scale)
        .toInt16()
    )
    boundaries = src.focal_max(1, "square", "pixels").neq(
        src.focal_min(1, "square", "pixels")
    )
    boundary_zone = boundaries.focal_max(feather_px, "square", "pixels")

    smooth = composite.focal_mean(max(1, feather_px // 2), "square", "pixels")
    return composite.where(boundary_zone, smooth)
