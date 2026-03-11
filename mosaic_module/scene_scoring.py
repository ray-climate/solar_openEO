"""Candidate scene scoring functions."""

from __future__ import annotations

import ee


def score_scenes(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Score scenes using AOI clear fraction only.

    Uses sum/total_px instead of mean so that partial-coverage scenes (where
    the Sentinel-2 tile only covers part of the chip) are penalised.
    reduceRegion(mean) divides by valid-pixel count, giving score=1.0 for a
    scene that covers 25% of the chip but is cloud-free within that 25%.
    reduceRegion(sum)/total_px instead divides by the full chip area.
    """
    # Total number of 10 m pixels in the chip AOI — computed once, reused per scene.
    total_px = ee.Number(
        ee.Image.constant(1)
        .reproject(crs="EPSG:3857", scale=10)
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=10,
            maxPixels=1e8,
        )
        .get("constant")
    )

    def _score(img: ee.Image) -> ee.Image:
        clear_sum = img.select("clear_mask").unmask(0).reproject(
            crs="EPSG:3857", scale=10,
        ).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            crs="EPSG:3857",
            maxPixels=1e9,
            tileScale=4,
        ).get("clear_mask")
        clear_frac = ee.Number(ee.Algorithms.If(clear_sum, clear_sum, 0)).divide(total_px)
        return img.set(
            {
                "clear_fraction": clear_frac,
                "temporal_score": ee.Number(0),
                "score": clear_frac,
            }
        )

    return collection.map(_score).sort("score", False)
