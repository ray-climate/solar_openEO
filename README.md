# solar_openEO

Temporal mosaicing and solar panel ML data prep on Sentinel-2.

## Start here (resume)
- Session handoff: `docs/HANDOFF_2026-02-26.md`
- Running memory: `PROJECT_LOG.md`
- Technical method: `docs/temporal_mosaic_v1.md`

## Stage-1: Positive sample extraction
- Code: `extraction_pipeline/`
- Scripts: `scripts/01_*.py` → `scripts/05_*.py`
- Input: `data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg`
- Output: `outputs/stage1/stage1_positives.h5`
- GEE Drive folder: `solar_openEO_stage1_mosaics`

### Run order (geospatial env)
```bash
# Step 1: sample polygons + build chip/tile manifests (no GEE needed)
conda run -n geospatial python scripts/01_sample_and_tile.py

# Step 2: submit GEE exports — test with 1 tile first
conda run -n geospatial python scripts/02_launch_gee_exports.py --batch-size 1
# then submit the rest in batches
conda run -n geospatial python scripts/02_launch_gee_exports.py --batch-size 50

# Step 3: monitor GEE job status
conda run -n geospatial python scripts/03_check_export_status.py

# Step 4: download mosaics from Google Drive → outputs/stage1/mosaics/
# then extract chips (works for whichever tiles are already downloaded)
conda run -n geospatial python scripts/04_extract_chips.py

# Step 5: package chips into HDF5
conda run -n geospatial python scripts/05_package_dataset.py
```

### Key outputs
```
outputs/stage1/
├── sample_5k.gpkg          5K sampled polygons (geographically stratified)
├── chip_manifest.csv       polygon → chip grid assignment
├── unique_chip_manifest.csv unique chips to export (~5725 chips for 5K polygons)
├── export_jobs.csv         GEE task IDs + status
├── mosaics/                downloaded GeoTIFFs (one per chip, 256×256 px)
├── chips/                  extracted 256×256 chips (_image.tif + _mask.tif)
├── chip_metadata.csv       per-chip stats (panel_frac, coverage_ok, continent)
└── stage1_positives.h5     final dataset: images(N,13,256,256) + masks(N,256,256)
```

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
