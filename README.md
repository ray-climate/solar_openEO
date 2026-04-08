# solar_openEO

Temporal mosaicing and solar panel ML data prep on Sentinel-2.

## Start here (resume)
- Session handoff: `docs/HANDOFF_2026-02-26.md`
- Running memory: `PROJECT_LOG.md`
- Technical method: `docs/temporal_mosaic_v1.md`
- Training workflow: `docs/training_unet_stage1.md`

## Stage-1: Positive sample extraction
- Code: `extraction_pipeline/`
- Scripts: `scripts/01_*.py` → `scripts/05_*.py`
- Input: `data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg`
- Output: `outputs/stage1/stage1_positives.h5`
- GEE Drive folder: `solar_openEO_stage1_v3`

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

# Step 4: download mosaics from Google Drive folder solar_openEO_stage1_v3
#         → outputs/stage1/mosaics/
# then extract chips (works for whichever tiles are already downloaded)
conda run -n geospatial python scripts/04_extract_chips.py

# Step 5: package chips into HDF5
conda run -n geospatial python scripts/05_package_dataset.py

# Recovery path: if chip_metadata.csv is missing but extracted chip pairs exist,
# package directly from chips/ and rebuild metadata on the fly
conda run -n geospatial python scripts/05_package_dataset.py --write-scanned-metadata

# End-to-end unattended rebuild from Drive sync through HDF5 QA
bash scripts/run_full_stage1_rebuild.sh
```

Notes:
- `scripts/04_extract_chips.py` can recover the unique chip list directly from `outputs/stage1/mosaics/*.tif` if `chip_manifest.csv` is missing.
- `scripts/05_package_dataset.py` can recover `chip_metadata.csv` by scanning `outputs/stage1/chips/` for `*_image.tif` / `*_mask.tif` pairs.

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

Verified Drive state on 2026-03-11:
- `google:solar_openEO_stage1_v3` exists and contains 4,922 `.tif` mosaics.

## Stage-2: U-Net training
- Code: `solar_ml/`
- Main trainer: `scripts/train_unet_stage1.py`
- Prep script: `scripts/prepare_stage1_training_data.py`
- Experiment configs: `configs/training/experiments/`
- Slurm GPU scripts: `scripts/slurm/train_stage1_gpu.sh`, `scripts/slurm/submit_stage1_backbone_grid.sh`
- Result comparison: `scripts/compare_training_runs.py`
- Threshold / failure review: `scripts/evaluate_training_run.py`

### Current design
- Model: U-Net decoder with pretrained `tf.keras.applications` encoder backbones
- Input: all 13 Sentinel-2 L1C spectral bands
- Backbone adaptation: learnable `1x1` projection from 13 bands to 3 channels
- Default split: deterministic spatial-block split by chip ID
- Current supported backbone configs:
  - `resnet50`
  - `efficientnetb0`
  - `mobilenetv2`
  - `resnet50` with frozen-encoder warmup

### Prepare training inputs
```bash
python scripts/prepare_stage1_training_data.py \
  --h5 outputs/stage1/stage1_positives.h5 \
  --out-dir outputs/training_prep/stage1_v3
```

### Spot-check packaged HDF5 samples
```bash
python scripts/08_visualize_h5_samples.py \
  --h5 outputs/stage1/stage1_positives.h5 \
  --n 6 \
  --out outputs/stage1/figures/h5_random_samples.png
```

### Dry-run the trainer
```bash
conda run -n tf-gpu python scripts/train_unet_stage1.py \
  --config configs/training/experiments/exp_stage1_resnet50.yaml \
  --dry-run -v
```

### Submit a GPU training grid
```bash
# Safe to resume: skips queued or checkpointed experiments by default.
DRY_RUN=1 scripts/slurm/submit_stage1_backbone_grid.sh
scripts/slurm/submit_stage1_backbone_grid.sh
```

Training throughput defaults now include:
- `training.input_backend: tfdata`
- full split preload into host memory with one-time transpose/normalization
- cached image dtype `float16`
- `steps_per_execution: 16`
- `drop_remainder: true`
- `shuffle_buffer: 2048`
- `tfdata_num_parallel_calls: 8`
- `tfdata_prefetch_buffer: 4`
- fallback backend remains available with `training.input_backend: sequence`

### Compare completed runs
```bash
python scripts/compare_training_runs.py
```

### Sweep thresholds and render failure gallery for a completed run
```bash
/home/users/ruisong/mambaforge_new/envs/tf-gpu/bin/python scripts/evaluate_training_run.py \
  --experiment-dir experiments/exp_stage1_resnet101_tfdata \
  --split test
```

This writes under `<experiment-dir>/evaluation_<split>/`:
- `threshold_sweep_<split>.csv`
- `per_chip_<split>_thr_<best>.csv`
- `failure_gallery_<split>_thr_<best>.png`
- `summary_<split>.json`

Current training status:
- local packaged dataset is now rebuilt from the synced full local mirror at `outputs/stage1/stage1_positives.h5` with `images(4922,13,256,256)` and `masks(4922,256,256)`
- refreshed comparison state on 2026-03-13:
  - best clean completed run: `exp_stage1_resnet101_tfdata`
  - highest completed test-Dice run: `exp_stage1_resnet101_bce_dice_tfdata` but its recorded loss is still `NaN`
  - threshold-aware evaluation artifacts now exist for both completed ResNet101 runs under their experiment directories
- use `RESUBMIT_EXISTING=1 scripts/slurm/submit_stage1_backbone_grid.sh` only if you intentionally want duplicate reruns

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
