"""Scene assignment logic for clusters and fallback patch grids."""

from __future__ import annotations

from typing import Dict, List, Sequence

import ee


def compute_clear_fraction_per_cluster(scene: ee.Image, clusters: ee.Image) -> ee.Image:
    """Compute mean clear mask fraction for each connected cluster."""
    clear = scene.select("clear_mask").unmask(0).rename("clear_mask")
    labels = clusters.rename("clusters").toInt64()
    frac = clear.addBands(labels).reduceConnectedComponents(
        reducer=ee.Reducer.mean(),
        labelBand="clusters",
        maxSize=4096,
    )
    return frac.rename("cluster_clear_frac")


def _scene_quality_image(
    scene: ee.Image,
    scene_idx: int,
    labels: ee.Image,
    clear_thresh: float,
) -> ee.Image:
    cluster_clear = compute_clear_fraction_per_cluster(scene, labels).rename("score")
    date_int = ee.Number.parse(scene.date().format("YYYYMMdd"))
    out = (
        cluster_clear
        .addBands(ee.Image.constant(scene_idx).toInt16().rename("source_scene_id"))
        .addBands(ee.Image.constant(date_int).toInt32().rename("source_date"))
        .updateMask(cluster_clear.gte(clear_thresh))
    )
    return out


def assign_clusters(
    clusters: ee.Image,
    candidate_list: Sequence[ee.Image],
    clear_thresh: float,
) -> Dict[str, ee.Image]:
    """Assign each SNIC cluster to the best-scoring scene."""
    if not candidate_list:
        raise ValueError("candidate_list is empty.")

    quality_layers = [
        _scene_quality_image(scene, idx, clusters, clear_thresh)
        for idx, scene in enumerate(candidate_list)
    ]
    winner = ee.ImageCollection(quality_layers).qualityMosaic("score")

    assigned_mask = winner.select("score").mask().unmask(0).gt(0)
    source_scene_id = winner.select("source_scene_id").unmask(-1).toInt16()
    source_date = winner.select("source_date").unmask(0).toInt32()

    return {
        "source_scene_id": source_scene_id.rename("source_scene_id"),
        "source_date": source_date.rename("source_date"),
        "best_cluster_clear_fraction": winner.select("score")
        .unmask(0)
        .rename("best_cluster_clear_fraction"),
        "unassigned_mask": assigned_mask.Not().rename("unassigned_mask"),
    }


def _build_grid_labels(aoi: ee.Geometry, patch_size_px: int, out_scale: int) -> ee.Image:
    patch_size_m = ee.Number(patch_size_px).multiply(out_scale)
    grid_proj = ee.Projection("EPSG:3857").atScale(out_scale)
    coords = ee.Image.pixelCoordinates(grid_proj)
    x_id = coords.select("x").divide(patch_size_m).floor().toInt64()
    y_id = coords.select("y").divide(patch_size_m).floor().toInt64()
    labels = x_id.multiply(10_000_000).add(y_id).toInt64().rename("clusters")
    return labels.clip(aoi)


def _patch_sizes(start_px: int, min_px: int) -> List[int]:
    start_px = max(start_px, min_px)
    sizes = []
    curr = start_px
    while curr >= min_px:
        sizes.append(curr)
        if curr == min_px:
            break
        next_px = max(min_px, curr // 2)
        if next_px == curr:
            break
        curr = next_px
    return sizes


def _best_available_pixel_assignment(
    candidate_list: Sequence[ee.Image],
) -> ee.Image:
    layers = []
    for idx, scene in enumerate(candidate_list):
        date_int = ee.Number.parse(scene.date().format("YYYYMMdd"))
        score = scene.select("clear_mask").unmask(0).rename("score")
        layers.append(
            score.addBands(
                [
                    ee.Image.constant(idx).toInt16().rename("source_scene_id"),
                    ee.Image.constant(date_int).toInt32().rename("source_date"),
                ]
            )
        )
    return ee.ImageCollection(layers).qualityMosaic("score")


def hierarchical_split_and_assign(
    source_scene_id: ee.Image,
    source_date: ee.Image,
    unassigned_mask: ee.Image,
    candidate_list: Sequence[ee.Image],
    aoi: ee.Geometry,
    patch_size_px: int = 64,
    min_patch_px: int = 8,
    clear_thresh: float = 0.9,
    out_scale: int = 10,
) -> Dict[str, ee.Image]:
    """Hierarchical fallback using progressively finer grid patches."""
    current_scene_id = source_scene_id.toInt16()
    current_source_date = source_date.toInt32()
    current_unassigned = unassigned_mask.unmask(0).gt(0).rename("unassigned_mask")

    for size_px in _patch_sizes(patch_size_px, min_patch_px):
        labels = _build_grid_labels(aoi=aoi, patch_size_px=size_px, out_scale=out_scale)
        candidates = [
            _scene_quality_image(scene, idx, labels, clear_thresh)
            for idx, scene in enumerate(candidate_list)
        ]
        winner = ee.ImageCollection(candidates).qualityMosaic("score")
        winner_mask = winner.select("score").mask().unmask(0).gt(0)

        can_fill = current_unassigned.And(winner_mask)
        current_scene_id = current_scene_id.where(
            can_fill, winner.select("source_scene_id")
        )
        current_source_date = current_source_date.where(
            can_fill, winner.select("source_date")
        )
        current_unassigned = current_unassigned.And(can_fill.Not())

    # Last resort: fill remaining cells with best per-pixel clear scene.
    best_any = _best_available_pixel_assignment(candidate_list)
    can_fill_any = current_unassigned.And(
        best_any.select("source_scene_id").mask().unmask(0).gt(0)
    )
    current_scene_id = current_scene_id.where(
        can_fill_any, best_any.select("source_scene_id")
    )
    current_source_date = current_source_date.where(
        can_fill_any, best_any.select("source_date")
    )
    current_unassigned = current_unassigned.And(can_fill_any.Not())

    return {
        "source_scene_id": current_scene_id.rename("source_scene_id"),
        "source_date": current_source_date.rename("source_date"),
        "unassigned_mask": current_unassigned.rename("unassigned_mask"),
    }
