"""Prototype: Sentinel-2 temporal mosaic using OpenEO Python client.

This is a PROTOTYPE translation of the GEE-based mosaic algorithm
(mosaic_module/) into OpenEO process graph operations.  It is intended as
a starting point for the Apex team, who will validate and adapt it for
their target backend.

The GEE mosaic uses SNIC superpixel segmentation + per-cluster scene
assignment — which has no direct OpenEO equivalent.  This prototype uses
a simplified but functionally equivalent approach:

  GEE algorithm (mosaic_module/)         OpenEO prototype (this file)
  --------------------------------       --------------------------------
  1. Load S2 L1C + s2cloudless           1. load_collection(SENTINEL2_L1C)
  2. Cloud mask (prob >= 65, morph)      2. Cloud mask (SCL or s2cloudless)
  3. Score scenes by clear fraction      3. (implicit in reduce_dimension)
  4. SNIC segmentation                   -- NOT AVAILABLE in OpenEO --
  5. Per-cluster best-scene assignment   4. Per-pixel best-clear composite
  6. Hierarchical fallback fill          -- folded into step 4 --
  7. Rescue fill (lowest cloud prob)     5. Fallback: temporal median
  8. Seam feathering                     -- not needed (per-pixel) --

NOTE FOR APEX:
  The SNIC-based approach produces spatially coherent mosaics (whole
  superpixels from the same scene).  The OpenEO per-pixel approach may
  produce more inter-pixel spectral inconsistency.  Consider whether
  your backend supports:
    - ard_normalized_radar_backscatter (for compositing)
    - aggregate_spatial (for block-based assignment)
    - Custom UDFs for SNIC-like clustering
  If so, a closer replica of the GEE algorithm may be possible.

Usage (example, not meant to be run as-is):
    python openeo_udp/process_graph/mosaic_prototype.py
"""

from __future__ import annotations

from typing import Optional

import openeo


# Sentinel-2 L1C band names in the order expected by the model.
# NOTE: OpenEO collection band names may differ from GEE names.
# Apex should verify the exact band names on their backend.
S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12",
]

# Mapping from GEE band names (used in training) to OpenEO band names.
# Adjust if the target backend uses different naming conventions.
GEE_TO_OPENEO_BANDS = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04",
    "B5": "B05", "B6": "B06", "B7": "B07", "B8": "B08",
    "B8A": "B8A", "B9": "B09", "B10": "B10", "B11": "B11", "B12": "B12",
}


def create_cloud_mask(
    s2_cube: openeo.DataCube,
    cloud_prob_threshold: int = 65,
) -> openeo.DataCube:
    """Create a cloud-free Sentinel-2 datacube.

    GEE equivalent:
        - attach_cloud_prob() + compute_clear_mask() in cloud_mask.py
        - Uses s2cloudless probability >= 65
        - Morphological operations: focal_min(1) + focal_max(2)
        - Connected component filter >= 4 px

    OpenEO approach:
        Option A (preferred): Use mask_scl_dilation() if SCL band is available
        Option B: Load SENTINEL2_L2A and use SCL classification
        Option C: Custom UDF with s2cloudless (most faithful to training)

    NOTE FOR APEX:
        The model was trained on L1C data masked with s2cloudless (prob >= 65).
        If using SCL-based masking (L2A), there may be slight distributional
        differences.  Option C is the most faithful to training conditions.
    """
    # Option A: SCL-based masking (simplest, works on most backends)
    # This uses the Scene Classification Layer from L2A processing.
    # SCL classes to mask out:
    #   3 = cloud shadow, 8 = cloud medium probability,
    #   9 = cloud high probability, 10 = thin cirrus
    masked = s2_cube.process(
        "mask_scl_dilation",
        data=s2_cube,
        scl_band_name="SCL",
        kernel1_size=3,   # erosion (similar to GEE focal_min(1))
        kernel2_size=5,   # dilation (similar to GEE focal_max(2))
    )
    return masked


def create_temporal_mosaic(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    collection_id: str = "SENTINEL2_L1C",
    cloud_prob_threshold: int = 65,
    target_crs: str = "EPSG:3857",
    target_resolution: int = 10,
) -> openeo.DataCube:
    """Build a cloud-free temporal mosaic from Sentinel-2 data.

    This produces an output comparable to the GEE create_temporal_mosaic()
    function.  The model expects a single cloud-free composite with all 13
    L1C bands at 10m resolution in EPSG:3857.

    Parameters
    ----------
    connection : openeo.Connection
        Authenticated OpenEO connection.
    spatial_extent : dict
        Bounding box: {"west": ..., "south": ..., "east": ..., "north": ...,
                       "crs": "EPSG:4326"}.
        For chip-aligned inference, use EPSG:3857 coordinates matching the
        training grid (origin 0,0, cell size 2560m = 256px * 10m).
    temporal_extent : list[str]
        Date range ["YYYY-MM-DD", "YYYY-MM-DD"].
        Training data used ["2024-05-01", "2024-07-31"] (summer window).
    collection_id : str
        OpenEO collection ID.  Options:
            - "SENTINEL2_L1C" (closest to training data)
            - "SENTINEL2_L2A" (if L1C not available, has SCL band)
    cloud_prob_threshold : int
        Cloud probability threshold.  GEE training used 65.
    target_crs : str
        Output CRS.  Must be EPSG:3857 to match training grid.
    target_resolution : int
        Output resolution in metres.  Must be 10 to match training.

    Returns
    -------
    openeo.DataCube
        Single-timestep datacube with 13 bands at 10m in EPSG:3857.

    GEE algorithm correspondence:
        Step 1-2: load_collection + cloud masking
        Step 3:   scene scoring → implicit in rank/filter
        Step 4-6: SNIC + assignment → simplified to per-pixel composite
        Step 7:   rescue fill → temporal median fallback
    """
    # --- Step 1: Load L1C spectral data (all 13 bands, TOA reflectance) ---
    # The model was trained on L1C. Using L2A would change radiometry.
    s2_l1c = connection.load_collection(
        collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=S2_L1C_BANDS,
    )

    # Resample all bands to 10m in target CRS
    # (GEE equivalent: align_l1c_bands() in resample_align.py)
    s2_l1c = s2_l1c.resample_spatial(resolution=target_resolution, projection=target_crs)

    # --- Step 2: Cloud masking using L2A SCL ---
    # L1C has no cloud classification band.  Load L2A SCL separately and
    # use it to mask clouds in the L1C data.  This mirrors the GEE training
    # pipeline which used s2cloudless probability (≥65) + morphological ops.
    #
    # SCL classes masked out:
    #   0 = no data, 1 = saturated/defective, 3 = cloud shadow,
    #   8 = cloud medium prob, 9 = cloud high prob, 10 = thin cirrus
    #
    # NOTE FOR APEX: If s2cloudless is available as a collection on your
    # backend, using it directly (threshold ≥ 65) would be more faithful
    # to the training pipeline.
    s2_scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
    )

    scl_mask = (
        (s2_scl.band("SCL") != 0)
        & (s2_scl.band("SCL") != 1)
        & (s2_scl.band("SCL") != 3)
        & (s2_scl.band("SCL") != 8)
        & (s2_scl.band("SCL") != 9)
        & (s2_scl.band("SCL") != 10)
    )

    s2_masked = s2_l1c.mask(scl_mask)

    # --- Step 3-6: Temporal compositing ---
    # TWO OPTIONS are provided:
    #
    # Option A (RECOMMENDED): SLIC-based mosaic UDF
    #   Faithful replication of the GEE SNIC algorithm using a UDF.
    #   Requires merging L1C + SCL into a 14-band temporal stack and
    #   applying the UDF via chunk_polygon or apply_neighborhood.
    #   See openeo_udp/udf/temporal_mosaic.py for the full implementation.
    #
    # Option B (FALLBACK): Simple temporal median
    #   Robust but loses spatial coherence. Adequate for quick testing.
    #
    # NOTE FOR APEX: Option A is preferred because it produces spatially
    # coherent mosaics matching the training pipeline. If UDF execution
    # is not feasible, Option B is acceptable but may slightly degrade
    # detection quality due to inter-pixel spectral inconsistency.

    # --- Option A: SLIC-based mosaic via UDF ---
    # Merge L1C spectral + SCL into 14-band cube, keep temporal dimension
    s2_merged = s2_l1c.merge_cubes(s2_scl)

    from pathlib import Path
    udf_code = Path("openeo_udp/udf/temporal_mosaic.py").read_text()
    mosaic = s2_merged.reduce_dimension(
        dimension="t",
        reducer=openeo.UDF(udf_code, runtime="Python"),
    )

    # --- Option B (fallback): Simple temporal median ---
    # Uncomment below (and comment Option A) if UDF is not supported:
    # s2_masked = s2_l1c.mask(scl_mask)
    # mosaic = s2_masked.reduce_dimension(dimension="t", reducer="median")

    return mosaic


def build_solar_pv_detection_graph(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    collection_id: str = "SENTINEL2_L1C",
    threshold: float = 0.80,
    udf_path: str = "openeo_udp/udf/solar_pv_inference.py",
) -> openeo.DataCube:
    """Full detection pipeline: mosaic -> normalise -> infer -> threshold.

    This is the complete process graph that would be registered as a UDP.

    Parameters
    ----------
    connection : openeo.Connection
        Authenticated OpenEO connection.
    spatial_extent : dict
        AOI bounding box.
    temporal_extent : list[str]
        Date range for Sentinel-2 imagery.
    collection_id : str
        Sentinel-2 collection.
    threshold : float
        Detection threshold (default 0.80, from test-set evaluation).
    udf_path : str
        Path to the UDF Python file.

    Returns
    -------
    openeo.DataCube
        Binary detection mask (1 = solar PV panel, 0 = background).
    """
    # Step 1: Build cloud-free mosaic
    mosaic = create_temporal_mosaic(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        collection_id=collection_id,
    )

    # Step 2: Run inference via UDF
    # The UDF handles normalisation (z-score) and model inference internally.
    # It reads model_registry.yaml for weights URL and parameters.
    from pathlib import Path
    udf_code = Path(udf_path).read_text()

    detection_mask = mosaic.apply_neighborhood(
        process=openeo.UDF(udf_code, runtime="Python"),
        size=[
            {"dimension": "x", "value": 256, "unit": "px"},
            {"dimension": "y", "value": 256, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 0, "unit": "px"},
            {"dimension": "y", "value": 0, "unit": "px"},
        ],
        context={"threshold": threshold},
    )

    return detection_mask


# ===================================================================
# Example usage (for reference / testing)
# ===================================================================

def example_usage():
    """Example: detect solar PV panels in a small area near Oxford, UK."""
    # Connect to an OpenEO backend
    connection = openeo.connect("https://openeo.cloud")
    connection.authenticate_oidc()

    # Define AOI (small area for testing)
    spatial_extent = {
        "west": -1.30,
        "south": 51.74,
        "east": -1.25,
        "north": 51.77,
        "crs": "EPSG:4326",
    }

    # Summer window (matches training data temporal range)
    temporal_extent = ["2024-05-01", "2024-07-31"]

    # Run full pipeline
    result = build_solar_pv_detection_graph(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        collection_id="SENTINEL2_L1C",
        threshold=0.80,
    )

    # Download result
    result.download("solar_pv_detection_oxford.tif")
    print("Detection result saved to solar_pv_detection_oxford.tif")


if __name__ == "__main__":
    print("This module defines OpenEO process graph functions.")
    print("See example_usage() for a complete detection pipeline.")
    print("\nRun example_usage() interactively or adapt for your AOI.")
