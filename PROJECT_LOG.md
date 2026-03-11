# Solar OpenEO Project Log

## Purpose
This file is the running memory for the project. At the start of each session, read this file to understand prior actions, decisions, and next steps.

## Project Summary
- Goal: Use a CNN to detect solar panels from Sentinel-2 data.
- Scope: Early phase.

## Current Status (as of 2026-02-28)
- Stage-1 pipeline fully updated (Europe-only, size-stratified, chip-direct).
- 5,725 unique chips submitted to GEE; 473 COMPLETED, 2,774 READY/RUNNING, 2,475 SUBMIT_ERROR (GEE queue limit hit).
- SUBMIT_ERROR chips need resubmission once queue drains (see Next Steps).
- Per-chip outputs: `_image.tif` (13-band) + `_mask.tif` (binary) + `_quickview.png` (2-panel local QA).
- No RGB TIF exported via GEE — RGB generated locally from 13-band image.

## Decisions
- Exports are standardized to Google Drive folder: `solar_openEO_temporal_mosaics`.
- Stage-1 extraction exports to Drive folder: `solar_openEO_stage1_mosaics` (single folder, all chips).
- Training chip size: 256×256 px at 10 m/px = 2.56 km per side.
- Global chip grid: EPSG:3857, origin (0,0), cell size 2560 m. No 10km tile intermediate.
- Chip IDs encode grid position: `c{col:+07d}_r{row:+07d}`.
- Grid is globally consistent — same cell boundaries used for training and production inference.
- Stage-1 polygon dataset: 140K polygons, only Europe subset used (44,384 polygons). Fixed temporal window May–Jul 2024.
- Sampling: 5000 polygons, size-stratified (small/medium/large by area quantile, ~1667 each).
- Polygon-to-chip assignment: bbox-overlap (not centroid), so large polygons span multiple chips.
- Stage-1 = positive patches only (overlapping labeled polygons). Stage-2 = negatives.
- Per-chip local outputs: `_image.tif` (13-band float32) + `_mask.tif` (binary uint8) + `_quickview.png` (2-panel PNG).
- NO `_rgb.tif` — RGB is derived on demand from the 13-band image.
- GEE export: `export_rgb=False` — only 13-band mosaic exported to Drive.
- v1 output includes 13 Sentinel-2 L1C spectral bands + source QA bands.
- v1 fallback strategy uses hierarchical projected grids (coarse-to-fine) after SNIC assignment.
- Scene ranking uses AOI clear fraction only (no temporal-center weighting).
- Added configurable AOI size (`aoi_size_km`) for fast tests (e.g., 1x1 km).
- Added diagnostics plotting workflow (input RGB + cloud mask + mosaic QA panels).
- Added optional scene-level cloud metadata filter (`max_scene_cloud_pct`), default off.
- Improved cloud/shadow mask by adding MSK class cloud cues and refined shadow projection.
- Added batch diagnostics script for multi-location QA (e.g., 3x3 km tests).
- Added `use_shadow_mask` switch for quick cloud-only sensitivity testing.
- Increased diagnostics image quality controls (`thumb_px`, `figure_dpi`).
- Added HTML gallery builder for browsing multi-location diagnostics.
- Updated masking to single cloud source: s2cloudless `cloud_prob` only.
- Default mode now uses cloud-only masking (`use_shadow_mask=False`).
- Raised default cloud threshold to 65 and added minimum cloud blob filter to reduce single-pixel noise.
- Added rescue-gap fill using minimum `cloud_prob` from top rescue scenes (`top_n_scenes_rescue=10`).
- Updated default cluster threshold to `clear_thresh=0.8`.
- Updated default batch diagnostics locations to non-urban regions.

## Data & Sources
- `COPERNICUS/S2_HARMONIZED` (Sentinel-2 L1C)
- `COPERNICUS/S2_CLOUD_PROBABILITY` (s2cloudless)

## Experiments / Results
- GEE connectivity and Drive export smoke test passed in `geospatial` environment.

## Open Questions
- Tune cloud/shadow thresholds for different regions/seasons.
- Evaluate SNIC parameters (`snic_size_px`, `compactness`) for rooftop coherence.
- Decide whether to persist intermediates (cluster/source maps) as assets.
- Stage-2: design negative patch sampling strategy.
- CNN architecture selection (U-Net or similar for pixel-level segmentation).

## Next Steps
1. **Resubmit failed chips** (2,475 SUBMIT_ERROR, hit GEE 3000-task queue limit):
   ```
   conda run -n geospatial python scripts/02_launch_gee_exports.py --retry-errors --batch-size 3000
   ```
   Run again as needed until all 5,725 are submitted. Check queue first:
   ```
   conda run -n geospatial python scripts/03_check_export_status.py
   ```
2. **Download mosaics** from Google Drive folder `solar_openEO_stage1_mosaics` → `outputs/stage1/mosaics/`
3. **Extract chips locally**:
   ```
   conda run -n geospatial python scripts/04_extract_chips.py
   ```
   Produces `_image.tif` + `_mask.tif` + `_quickview.png` per chip in `outputs/stage1/chips/`.
4. **QA figures**:
   ```
   conda run -n geospatial python scripts/06_visualize_chips.py
   ```
   Saves `chip_grid.png`, `chip_locations.png`, `chip_stats.png` to `outputs/stage1/figures/`.
5. **Package HDF5**:
   ```
   conda run -n geospatial python scripts/05_package_dataset.py
   ```
   Produces `outputs/stage1/stage1_positives.h5` — images (N,13,256,256) + masks (N,256,256).
6. Design Stage-2 negative patch extraction.
7. CNN architecture selection (U-Net or similar for pixel-level segmentation).

## Session Notes
- 2026-02-02: Created initial `PROJECT_LOG.md` as persistent project memory.
- 2026-02-26: Added `mosaic_module` temporal mosaicing v1 with SNIC assignment, hierarchical fallback, and Drive export workflow.
- 2026-02-26: Added `docs/HANDOFF_2026-02-26.md` with full resume checklist, defaults, workflows, and latest output locations.
- 2026-02-27: Built `extraction_pipeline/` Stage-1 positive sample extraction pipeline. Upgraded geopandas to 1.1.2 (was 0.9.0, incompatible with shapely 2.x). Added `drive_folder` and `export_rgb` params to `create_temporal_mosaic`. Switched from 10km tile approach to chip-direct (each GEE export = one 2.56km chip). Switched from centroid-only to bbox-overlap chip assignment.
- 2026-02-28: Major pipeline updates:
  - Filtered to Europe only (44,384 of 140,945 polygons).
  - Size-stratified sampling: small/medium/large by area quantile, ~1,667 each → 5,000 total.
  - Per-chip outputs changed: removed `_rgb.tif`, added `_quickview.png` (2-panel local PNG).
  - GEE export set to `export_rgb=False` — only 13-band mosaic written to Drive.
  - Added `--retry-errors` flag to `02_launch_gee_exports.py` for resubmitting SUBMIT_ERROR chips.
  - Re-ran step 01: 5,000 polygons → 5,725 unique chips.
  - Submitted all 5,725 chips to GEE; 3,250 accepted (READY), 2,475 rejected (GEE 3000-task queue limit).
  - As of end of session: 473 COMPLETED, 2,774 READY/RUNNING, 2,475 SUBMIT_ERROR.
