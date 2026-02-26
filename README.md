# solar_openEO

Temporal mosaicing and solar panel ML data prep on Sentinel-2.

## Start here (resume)
- Session handoff: `docs/HANDOFF_2026-02-26.md`
- Running memory: `PROJECT_LOG.md`
- Technical method: `docs/temporal_mosaic_v1.md`

## Temporal mosaic v1
- Code: `mosaic_module/`
- Main entrypoint: `mosaic_module.create_temporal_mosaic(...)`
- Test runner: `scripts/test_temporal_mosaic_v1.py`
- Docs:
  - `docs/temporal_mosaic_v1.md`
  - `docs/HANDOFF_2026-02-26.md`
- Export outputs:
  - Full mosaic (13 bands + QA/source layers)
  - RGB visualization (`*_rgb`) in the same Drive folder

### Quick run (geospatial env)
```bash
conda run -n geospatial python scripts/test_temporal_mosaic_v1.py \
  --lat 51.5074 --lon -0.1278 --start 2025-11-01 --end 2026-01-31 \
  --name s2_mosaic_london_v1
```

### Quick 1x1 km test export
```bash
conda run -n geospatial python scripts/test_temporal_mosaic_v1.py \
  --lat 51.5074 --lon -0.1278 --start 2025-11-01 --end 2026-01-31 \
  --aoi-km 1 --name s2_mosaic_london_1km_test \
  --clear-thresh 0.8 --top-n-scenes-rescue 10
```

### Diagnostics plot (input RGB + cloud mask + final mosaic QA)
```bash
conda run -n geospatial python scripts/plot_mosaic_diagnostics.py \
  --lat 51.5074 --lon -0.1278 --start 2025-11-01 --end 2026-01-31 \
  --aoi-km 1 --top-n 6
```
This creates a fast preview plot (quality-mosaic based) for QA.
Default mask mode is cloud-only (`s2cloudless`).
Use `--use-shadow` to enable projected shadow masking.
Use `--thumb-px` and `--figure-dpi` for higher-quality saved images.

### Batch diagnostics for multiple 3x3 km locations
```bash
conda run -n geospatial python scripts/plot_mosaic_diagnostics_batch.py \
  --start 2025-11-01 --end 2026-01-31 --aoi-km 3 --top-n 6 \
  --thumb-px 700 --figure-dpi 180 --out-dir outputs/diagnostics/batch10
```
Default batch locations include 10 non-urban regions.

Notes:
- `top-n` controls how many dates are shown in diagnostics.
- Coarse scene-cloud metadata filtering is optional via `--max-scene-cloud-pct`.
- Cloud mask source is `s2cloudless` (`cloud_prob`) only.
- Production mosaic defaults: `clear_thresh=0.8`, `top_n_scenes_rescue=10`.

### Build / refresh HTML gallery for browsing
```bash
conda run -n geospatial python scripts/build_diagnostics_gallery.py \
  --dir outputs/diagnostics/batch10 --output index.html
```
