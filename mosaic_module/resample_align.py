"""Band alignment and resampling helpers."""

from __future__ import annotations

import ee

S2_L1C_13_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]


def align_l1c_bands(
    img: ee.Image,
    aoi: ee.Geometry,
    out_scale: int = 10,
    out_crs: str = "EPSG:3857",
) -> ee.Image:
    """Upsample and align all 13 L1C spectral bands to a common grid."""
    return (
        img.select(S2_L1C_13_BANDS)
        .clip(aoi)
        .reproject(crs=out_crs, scale=out_scale)
        .rename(S2_L1C_13_BANDS)
    )
