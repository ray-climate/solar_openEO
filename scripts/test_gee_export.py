import random
from pathlib import Path

import geopandas as gpd
import ee

# Config
FGB_PATH = Path("data/solar_panels_merged_v4.fgb")
START_DATE = "2024-03-01"
END_DATE = "2024-08-01"  # end is exclusive
DRIVE_FOLDER = "gee_exports"
EXPORT_PREFIX = "solar_panel_test"
CHIP_BUFFER_M = 2560  # ~2.56 km buffer around centroid
SCALE_M = 10

# Rough EU bounding box (includes some non-EU)
EU_BBOX = (-31.0, 34.0, 39.0, 72.0)  # min lon, min lat, max lon, max lat


def pick_random_panel(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("No features in file")
    # Filter to rough EU bbox
    minx, miny, maxx, maxy = EU_BBOX
    gdf = gdf.cx[minx:maxx, miny:maxy]
    if gdf.empty:
        raise ValueError("No features in EU bbox")
    idx = random.randrange(len(gdf))
    geom = gdf.geometry.iloc[idx]
    if geom is None:
        raise ValueError("Selected feature has no geometry")
    return geom


def mask_s2_clouds(image):
    # Use S2 SR Harmonized + SCL to mask clouds, shadows, and snow.
    # SCL classes: 1,2,3,7,8,9,10,11 are typically non-clear.
    scl = image.select("SCL")
    clear = (
        scl.neq(1)
        .And(scl.neq(2))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(clear).divide(10000)


def main():
    ee.Initialize()

    geom = pick_random_panel(FGB_PATH)
    centroid = geom.centroid
    lon, lat = float(centroid.x), float(centroid.y)

    # Use a buffered point to keep the export region small and consistent
    region = ee.Geometry.Point([lon, lat]).buffer(CHIP_BUFFER_M).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_s2_clouds)
        .select(["B4", "B3", "B2"])  # RGB
    )

    composite = collection.median().clip(region)

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"{EXPORT_PREFIX}_{lon:.5f}_{lat:.5f}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{EXPORT_PREFIX}_{lon:.5f}_{lat:.5f}",
        region=region,
        scale=SCALE_M,
        maxPixels=1e13,
    )

    task.start()

    print("Export started.")
    print(f"Centroid (lat, lon): {lat:.6f}, {lon:.6f}")
    print(f"Drive folder: {DRIVE_FOLDER}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("Check GEE Tasks to monitor progress.")


if __name__ == "__main__":
    main()
