"""Sentinel-2 loading and cloud/shadow masking utilities."""

from __future__ import annotations

import ee


def load_sentinel_l1c(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_scene_cloud_pct: int | None = None,
) -> ee.ImageCollection:
    """Load Sentinel-2 L1C scenes for AOI/date.

    If ``max_scene_cloud_pct`` is set, applies a coarse scene-level filter using
    image metadata field ``CLOUDY_PIXEL_PERCENTAGE``. If ``None``, no coarse
    scene-level cloud filter is applied.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if max_scene_cloud_pct is not None:
        collection = collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud_pct)
        )
    return collection


def attach_cloud_prob(s2: ee.ImageCollection) -> ee.ImageCollection:
    """Attach s2cloudless cloud probability band as ``cloud_prob``.

    Images are joined by ``system:index``.
    """
    cloud_prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    joined = ee.ImageCollection(
        ee.Join.saveFirst("cloud_prob_img").apply(
            primary=s2,
            secondary=cloud_prob,
            condition=ee.Filter.equals(
                leftField="system:index",
                rightField="system:index",
            ),
        )
    )

    def _attach(img: ee.Image) -> ee.Image:
        prob_img = ee.Image(img.get("cloud_prob_img"))
        with_prob = img.addBands(
            prob_img.select("probability").rename("cloud_prob")
        )
        fallback = img.addBands(
            ee.Image.constant(100).rename("cloud_prob").toUint8()
        )
        return ee.Image(
            ee.Algorithms.If(img.get("cloud_prob_img"), with_prob, fallback)
        ).copyProperties(img, img.propertyNames())

    return joined.map(_attach)


def compute_clear_mask(
    img: ee.Image,
    aoi: ee.Geometry | None = None,
    cloud_prob_th: int = 65,
    dilate_px: int = 2,
    min_cloud_px: int = 4,
    shadow_nir_th: float = 0.15,
    shadow_proj_km: float = 1.5,
    use_shadow: bool = False,
) -> ee.Image:
    """Create and append cloud/shadow masks plus a clear mask.

    Mask logic:
    - cloud from s2cloudless ``cloud_prob >= cloud_prob_th``
    - cloud dilation by ``dilate_px``
    - removes tiny cloud components smaller than ``min_cloud_px``
    - projected shadows from cloud mask + dark NIR pixels
    """
    if aoi is not None:
        img = img.clip(aoi)

    band_names = img.bandNames()
    cloud_prob = ee.Image(
        ee.Algorithms.If(
            band_names.contains("cloud_prob"),
            img.select("cloud_prob"),
            ee.Image.constant(100).rename("cloud_prob"),
        )
    )

    cloud_from_prob = cloud_prob.gte(cloud_prob_th)
    cloud_mask_raw = (
        cloud_from_prob
        .focal_min(1, "square", "pixels")
        .focal_max(radius=dilate_px, units="pixels")
    )
    if min_cloud_px > 1:
        cloud_size = cloud_mask_raw.selfMask().connectedPixelCount(
            maxSize=1024,
            eightConnected=True,
        )
        cloud_mask = (
            cloud_mask_raw.updateMask(cloud_size.gte(min_cloud_px))
            .unmask(0)
            .rename("cloud_mask")
        )
    else:
        cloud_mask = cloud_mask_raw.rename("cloud_mask")

    if use_shadow:
        snow_cls = ee.Image(
            ee.Algorithms.If(
                band_names.contains("MSK_CLASSI_SNOW_ICE"),
                img.select("MSK_CLASSI_SNOW_ICE"),
                ee.Image.constant(0).rename("MSK_CLASSI_SNOW_ICE").toUint8(),
            )
        )
        sun_azimuth = ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        shadow_azimuth = ee.Number(90).subtract(sun_azimuth)
        proj_dist_px = ee.Number(shadow_proj_km).multiply(1000).divide(20)

        projected = (
            cloud_mask.directionalDistanceTransform(shadow_azimuth, proj_dist_px)
            .reproject(crs="EPSG:3857", scale=20)
            .select("distance")
            .mask()
        )
        dark_nir = img.select("B8").lt(int(shadow_nir_th * 10000)).And(snow_cls.eq(0))
        shadow_mask = projected.And(dark_nir).rename("shadow_mask")
    else:
        shadow_mask = ee.Image.constant(0).rename("shadow_mask").toUint8()

    cloud_shadow = cloud_mask.Or(shadow_mask).rename("cloud_shadow_mask")
    clear_mask = cloud_shadow.Not().rename("clear_mask")

    return img.addBands(
        [
            cloud_mask.toUint8(),
            shadow_mask.toUint8(),
            cloud_shadow.toUint8(),
            clear_mask.toUint8(),
        ],
        overwrite=True,
    )
