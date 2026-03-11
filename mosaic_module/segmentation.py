"""SNIC-based segmentation helpers."""

from __future__ import annotations

import ee


_GRID_CRS = "EPSG:3857"
_GRID_SCALE = 10  # metres — must match out_scale used everywhere else


def build_reference_image(
    candidates: ee.ImageCollection,
    aoi: ee.Geometry,
) -> ee.Image:
    """Build a cloud-masked reference image for segmentation.

    Explicitly reprojected to EPSG:3857 at 10 m so SNIC runs at the correct
    scale.  Without this GEE resolves to its default nominal scale (~111 km in
    EPSG:4326), producing effectively one superpixel per chip.
    """

    def _prep(img: ee.Image) -> ee.Image:
        reflectance = img.select(["B4", "B3", "B2", "B8"]).divide(10000)
        ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return (
            reflectance.addBands(ndvi)
            .updateMask(img.select("clear_mask"))
            .clip(aoi)
            .reproject(crs=_GRID_CRS, scale=_GRID_SCALE)
        )

    return (
        ee.ImageCollection(candidates.map(_prep))
        .median()
        .clip(aoi)
        .reproject(crs=_GRID_CRS, scale=_GRID_SCALE)
    )


def run_snic(
    reference: ee.Image,
    size_px: int,
    compactness: float = 1.0,
) -> ee.Image:
    """Run SNIC and return the integer ``clusters`` label band.

    Output is reprojected to EPSG:3857 at 10 m so downstream
    ``reduceConnectedComponents`` runs at the correct pixel scale.
    """
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=reference,
        size=size_px,
        compactness=compactness,
        connectivity=8,
        neighborhoodSize=size_px * 2,
    )
    return (
        ee.Image(snic)
        .select("clusters")
        .toInt64()
        .rename("clusters")
        .reproject(crs=_GRID_CRS, scale=_GRID_SCALE)
    )
