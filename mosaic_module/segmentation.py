"""SNIC-based segmentation helpers."""

from __future__ import annotations

import ee


def build_reference_image(
    candidates: ee.ImageCollection,
    aoi: ee.Geometry,
) -> ee.Image:
    """Build a cloud-masked reference image for segmentation."""

    def _prep(img: ee.Image) -> ee.Image:
        reflectance = img.select(["B4", "B3", "B2", "B8"]).divide(10000)
        ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return (
            reflectance.addBands(ndvi)
            .updateMask(img.select("clear_mask"))
            .clip(aoi)
        )

    return ee.ImageCollection(candidates.map(_prep)).median().clip(aoi)


def run_snic(
    reference: ee.Image,
    size_px: int,
    compactness: float = 1.0,
) -> ee.Image:
    """Run SNIC and return the integer ``clusters`` label band."""
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=reference,
        size=size_px,
        compactness=compactness,
        connectivity=8,
        neighborhoodSize=size_px * 2,
    )
    return ee.Image(snic).select("clusters").toInt64().rename("clusters")
