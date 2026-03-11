# Temporal Mosaic v1

## Purpose
This document describes version-1 of the temporal mosaicing pipeline implemented in `mosaic_module`.

## High-level flow
1. Build a true planar AOI in `EPSG:3857` from either center point + size or exact projected bounds.
2. Load Sentinel-2 L1C scenes (`COPERNICUS/S2_HARMONIZED`) for the date range.
3. Join s2cloudless (`COPERNICUS/S2_CLOUD_PROBABILITY`) by `system:index`.
4. Compute cloud/shadow masks and a `clear_mask`.
5. Score scenes by AOI clear fraction only.
6. Keep top-N candidate scenes.
7. Build a cloud-masked reference image and run SNIC to get superpixels.
8. Assign each superpixel to the best scene by per-cluster clear fraction.
9. Fallback with hierarchical grid splitting (`patch_size_px` down to `min_patch_px`).
10. Stitch assigned scenes into one 13-band composite.
11. Run rescue fill on remaining gaps using lowest `cloud_prob` pixels from top rescue scenes.
12. Feather boundaries and add QA/source bands.
13. Export to Google Drive:
   - full mosaic (13 bands + QA/source layers)
   - optional RGB visualization file (`B4,B3,B2`) scaled to 8-bit
   - when exact bounds are provided, pin export with `crsTransform` + `dimensions` for exact pixel-grid alignment

## Output bands
- 13 L1C spectral bands: `B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12`
- QA / provenance:
  - `source_scene_id` (int index in ranked candidate list)
  - `source_date` (`YYYYMMDD` integer)
  - `assigned_valid_mask` (1 where assignment+clear existed before median fill)
  - `fill_mode` (`1` if pixel came from rescue fill)
  - `rescue_cloud_prob` (cloud probability used for rescue-filled pixels)
- Separate visualization export:
  - `*_rgb` (3-band 8-bit RGB GeoTIFF)

## Module layout
- `mosaic_module/io_helpers.py`: EE init + Drive export
- `mosaic_module/cloud_mask.py`: loading + s2cloudless cloud mask (optional shadow stage)
- `mosaic_module/resample_align.py`: 13-band alignment to 10m
- `mosaic_module/scene_scoring.py`: clear-fraction scene score
- `mosaic_module/segmentation.py`: reference image + SNIC
- `mosaic_module/assignment.py`: cluster assignment + hierarchical fallback
- `mosaic_module/stitching.py`: compose + seam feathering
- `mosaic_module/main.py`: orchestrator and public API

## Public API
```python
from mosaic_module import create_temporal_mosaic

result = create_temporal_mosaic(
    center_lat=51.5074,
    center_lon=-0.1278,
    start_date="2025-11-01",
    end_date="2026-01-31",
    aoi_size_km=10.0,
    out_scale=10,
    patch_size_px=64,
    snic_size_px=40,
    clear_thresh=0.8,
    top_n_scenes=8,
    top_n_scenes_rescue=10,
    feather_px=12,
    min_patch_px=8,
    export_target="drive",
    export_name="s2_mosaic_london_v1",
    max_scene_cloud_pct=None,
    use_shadow_mask=False,
    drive_folder="solar_openEO_temporal_mosaics",
    export_rgb=True,
    aoi_bounds_3857=None,
)
```

Notes:
- `drive_folder` overrides the default Drive destination.
- `export_rgb=False` is used by the Stage-1 chip pipeline to export only the 13-band product.
- `aoi_bounds_3857=(xmin, ymin, xmax, ymax)` forces exact projected chip bounds and exact output dimensions.

## Diagnostics plot
Use `create_mosaic_diagnostic_plot(...)` (or `scripts/plot_mosaic_diagnostics.py`) to generate
a static QA figure with:
- per-date input RGB
- per-date cloud/shadow mask
- per-date clear mask
- final quick-preview mosaic RGB + source scene map + assigned-valid mask

Note: the diagnostics plot uses a lightweight clear-pixel quality-mosaic preview for fast visualization.
The production export still uses the full SNIC + hierarchical assignment pipeline.

## Notes / limitations in v1
- `export_target` currently supports `drive` only.
- Scene scoring divides clear-pixel sum by total AOI pixels, so partial-coverage scenes are penalized instead of scoring as fully clear.
- `clear_mask` is restricted to the real Sentinel-2 scene footprint (`B4 > 0`) to avoid treating out-of-swath zero-fill as valid clear data.
- SNIC reference and cluster outputs are explicitly reprojected to EPSG:3857 at 10 m so segmentation and connected-components run at the intended scale.
- Hierarchical fallback uses regular projected grids (coarse-to-fine), not recursive SNIC splitting.
- Feathering is a light boundary smoothing approximation.
- For very cloudy windows, remaining gaps are filled from clear-masked temporal median.
- Candidate ranking is AOI-clear-fraction driven. Scene-level cloud metadata filtering is optional.
- Cloud mask source is s2cloudless (`cloud_prob`) only; shadow stage is optional and off by default.
- Default mosaic tuning: `clear_thresh=0.8`, `top_n_scenes_rescue=10`.

## Session continuity
- Quick resume reference: `docs/HANDOFF_2026-02-26.md`
- Project running memory: `PROJECT_LOG.md`
