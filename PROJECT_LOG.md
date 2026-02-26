# Solar OpenEO Project Log

## Purpose
This file is the running memory for the project. At the start of each session, read this file to understand prior actions, decisions, and next steps.

## Project Summary
- Goal: Use a CNN to detect solar panels from Sentinel-2 data.
- Scope: Early phase.

## Current Status (as of 2026-02-26)
- Implemented temporal mosaicing v1 module using Earth Engine.
- Added Drive export-focused pipeline for 10x10 km AOI mosaics.
- Added consolidated handoff doc: `docs/HANDOFF_2026-02-26.md`.

## Decisions
- Exports are standardized to Google Drive folder: `solar_openEO_temporal_mosaics`.
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

## Next Steps
- Validate mosaic quality across several AOIs and seasons.
- Add quantitative QA metrics (residual cloud fraction, seam density, scene fragmentation).
- Integrate module into training-data and prediction-data workflows.
- Keep `docs/HANDOFF_2026-02-26.md` updated as settings/logic evolve.

## Session Notes
- 2026-02-02: Created initial `PROJECT_LOG.md` as persistent project memory.
- 2026-02-26: Added `mosaic_module` temporal mosaicing v1 with SNIC assignment, hierarchical fallback, and Drive export workflow.
- 2026-02-26: Added `docs/HANDOFF_2026-02-26.md` with full resume checklist, defaults, workflows, and latest output locations.
