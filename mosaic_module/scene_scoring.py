"""Candidate scene scoring functions."""

from __future__ import annotations

import ee


def score_scenes(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Score scenes using AOI clear fraction only."""

    def _score(img: ee.Image) -> ee.Image:
        clear_frac_raw = img.select("clear_mask").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=4,
        ).get("clear_mask")
        clear_frac = ee.Number(ee.Algorithms.If(clear_frac_raw, clear_frac_raw, 0))
        score = clear_frac
        return img.set(
            {
                "clear_fraction": clear_frac,
                "temporal_score": ee.Number(0),
                "score": score,
            }
        )

    return collection.map(_score).sort("score", False)
