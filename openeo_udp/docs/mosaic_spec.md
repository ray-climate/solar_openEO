# Solar PV Detection — Mosaic Algorithm Specification

This document describes the temporal mosaic algorithm used to create
cloud-free Sentinel-2 composites for model training.  It is written in
backend-agnostic terms so the Apex team can implement it in OpenEO.

The GEE implementation lives in `mosaic_module/`.  This spec extracts
the algorithm logic independent of GEE API calls.

## Purpose

Create a single cloud-free, temporally consistent composite image from
multiple Sentinel-2 L1C scenes over a defined time window.  The output
is a 13-band image at 10 m resolution in EPSG:3857.

## Training Parameters

These are the exact parameters used to create the training data:

| Parameter              | Value                    |
|------------------------|--------------------------|
| Collection             | COPERNICUS/S2_HARMONIZED (L1C) |
| Temporal window        | 2024-05-01 to 2024-07-31 |
| Cloud probability source | s2cloudless             |
| Cloud probability threshold | 65 (%)              |
| Morphological erosion  | 1 px (focal_min)         |
| Morphological dilation | 2 px (focal_max)         |
| Min cloud component    | 4 px (connected components) |
| Shadow masking         | Disabled                 |
| Top N scenes           | 8 (ranked by clear fraction) |
| Rescue scenes          | 10                       |
| Output CRS             | EPSG:3857                |
| Output resolution      | 10 m                     |
| Chip size              | 256 × 256 px (2560 m)    |

## Algorithm Steps

### Step 1: Load Sentinel-2 Scenes

Load all Sentinel-2 L1C scenes intersecting the AOI within the temporal
window.  No scene-level cloud percentage filter was applied (the
per-pixel cloud mask handles cloud removal).

**OpenEO equivalent:**
```python
s2 = connection.load_collection(
    "SENTINEL2_L1C",
    spatial_extent=bbox,
    temporal_extent=["2024-05-01", "2024-07-31"],
    bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
)
```

### Step 2: Cloud Masking

For each scene, compute a binary clear/cloud mask:

1. **Cloud probability**: From s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY),
   joined by `system:index`.  Pixels with probability >= 65 are cloud.
2. **Morphological cleaning**:
   - Erosion (focal_min, 1 pixel, square kernel) — removes thin cloud edges
   - Dilation (focal_max, 2 pixels, square kernel) — buffers cloud boundaries
3. **Speckle removal**: Connected components with < 4 pixels are removed
   (small noise patches reclassified as clear).
4. **Scene footprint**: Mask out pixels where B4 == 0 (off-swath fill pixels
   that S2 tiles write as zero rather than NoData).
5. **Final clear mask** = NOT(cloud OR shadow) AND scene_footprint

Shadow masking was disabled for training data (`use_shadow: False`).

**OpenEO equivalent (simplified):**
- If using L2A with SCL band: `mask_scl_dilation(kernel1_size=3, kernel2_size=5)`
- If using L1C: load s2cloudless as auxiliary collection and threshold

**Critical note:** The model was trained with s2cloudless masking on L1C
data.  Using SCL from L2A may produce slightly different cloud masks.
For highest fidelity, use s2cloudless if available on the backend.

### Step 3: Scene Scoring

Rank scenes by AOI clear fraction:

```
score(scene) = sum(clear_mask pixels in AOI) / total_pixels_in_AOI
```

Key detail: uses `sum / total_pixels` (not `mean`), so partial-coverage
scenes are penalised.  A scene covering 25% of the chip that is 100%
clear gets score=0.25, not score=1.0.

Select the top 8 scenes by score.

**OpenEO equivalent:**
This step is implicit when using temporal reduction.  If implementing
best-pixel compositing, load cloud probability as a band and use
`reduce_dimension` with a custom reducer that selects the pixel with
lowest cloud probability.

### Step 4: SNIC Superpixel Segmentation

Run SNIC (Simple Non-Iterative Clustering) on a cloud-masked reference
image built from the top candidates:

1. Build reference: median of top candidates (B4, B3, B2, B8, NDVI),
   masked to clear pixels, at 10 m in EPSG:3857
2. Run SNIC with `size=20` (for 256×256 chips) or `size=40` (for larger
   tiles), `compactness=1.0`, `connectivity=8`
3. Output: integer cluster label per pixel

**OpenEO equivalent:**
SNIC is not natively available in OpenEO.  Alternatives:
- **Skip this step** and use per-pixel compositing (simpler, recommended
  for initial implementation)
- Use a custom UDF to implement SNIC
- Use `aggregate_spatial` with a pre-computed grid

### Step 5: Per-Cluster Scene Assignment

For each SNIC cluster, compute the mean clear fraction from each
candidate scene.  Assign the cluster to the scene with the highest mean
clear fraction, provided it exceeds `clear_thresh` (0.80).

**OpenEO equivalent (per-pixel alternative):**
Use `reduce_dimension("t", reducer="min")` on cloud probability to
select the clearest observation per pixel.  Or use temporal median.

### Step 6: Hierarchical Fallback

Pixels not assigned after Step 5 go through progressively finer grids:

1. 64 × 64 px patches → assign to best scene per patch
2. 32 × 32 px patches → assign to best scene per patch
3. 16 × 16 px patches → assign to best scene per patch
4. 8 × 8 px patches → assign to best scene per patch
5. Per-pixel → select the scene with the highest clear_mask value

**OpenEO equivalent:**
Folded into the temporal reduction in Step 5 (median or best-pixel
already handles gap filling at per-pixel level).

### Step 7: Spectral Band Stitching

For each pixel, extract all 13 L1C bands from the assigned scene.
Non-native resolution bands (B1, B5-B7, B8A, B9, B10-B12) are
bilinearly resampled to 10 m.

Apply seam feathering at scene boundaries (focal_mean with radius
`feather_px / 2`).  For 256×256 chips, `feather_px=0` (no feathering).
For larger tiles, `feather_px=12`.

### Step 8: Rescue Fill

Any remaining unfilled pixels (no clear observation in top 8 scenes)
are filled using a "rescue" composite:

1. Select top 10 scenes by overall clear fraction
2. For each pixel, select the scene with the lowest cloud probability
   (quality mosaic on `-cloud_prob`)
3. Fill unfilled pixels with this rescue value
4. Record `fill_mode=1` for rescue-filled pixels

**OpenEO equivalent:**
A temporal median composite naturally handles this (no explicit rescue
step needed).

## Recommended OpenEO Implementation

For the initial UDP, use this simplified pipeline:

```
1. load_collection("SENTINEL2_L1C", bands=13, temporal_extent=3_months)
2. Cloud mask (SCL or s2cloudless)
3. Resample all bands to 10m in EPSG:3857
4. reduce_dimension("t", reducer="median")   ← replaces steps 3-8
```

The temporal median is the most robust and widely-supported approach.
It should produce comparable results to the SNIC-based approach for
the model's purposes, since:
- The model was designed to be robust to slight spectral variations
- Solar panels have stable spectral signatures across time
- The main goal of the mosaic is cloud removal, not temporal consistency

## Band Name Mapping

The GEE collection uses these band names (also used in model code):

| GEE name | OpenEO name (typical) | Index |
|----------|-----------------------|-------|
| B1       | B01                   | 0     |
| B2       | B02                   | 1     |
| B3       | B03                   | 2     |
| B4       | B04                   | 3     |
| B5       | B05                   | 4     |
| B6       | B06                   | 5     |
| B7       | B07                   | 6     |
| B8       | B08                   | 7     |
| B8A      | B8A                   | 8     |
| B9       | B09                   | 9     |
| B10      | B10                   | 10    |
| B11      | B11                   | 11    |
| B12      | B12                   | 12    |

**Verify band names on target backend** — they may differ.
