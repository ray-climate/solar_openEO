# Solar OpenEO Project Log

## Purpose
This file is the running memory for the project. At the start of each session, read this file to understand prior actions, decisions, and next steps.

## Project Summary
- Goal: Use a CNN to detect solar panels from Sentinel-2 data.
- Scope: Early phase.

## Current Status (as of 2026-03-13)
- Stage-1 pipeline fully updated (Europe-only, size-stratified, chip-direct).
- Repo default Drive folder is now `solar_openEO_stage1_v3`.
- Verified on 2026-03-11: `google:solar_openEO_stage1_v3` exists and contains 4,922 `.tif` mosaics.
- Updated later on 2026-03-11: local `outputs/stage1/mosaics/` now also contains all 4,922 synced mosaics.
- Last documented expected total remains 5,725 unique chips, so v3 is the active export destination but should not yet be treated as fully complete without reconciling against the manifest.
- Per-chip outputs: `_image.tif` (13-band) + `_mask.tif` (binary) + `_quickview.png` (2-panel local QA).
- No RGB TIF exported via GEE — RGB generated locally from the 13-band image.
- Stage-2 training stack is now implemented in-repo for HDF5-based training with a 13-band U-Net and pretrained backbone encoder variants.
- Training code is validated on synthetic data, dry-run validated on the rebuilt full HDF5, and real GPU training is now underway on JASMIN.
- Current local labeled dataset is available at `outputs/stage1/stage1_positives.h5` with shape `images(4922,13,256,256)` / `masks(4922,256,256)`.
- Recovery tooling is now in place for partial local state:
  - extraction can recover chip IDs from downloaded mosaic filenames
  - packaging can recover chip metadata from extracted chip pairs
  - HDF5 random-sample QA figures can be generated directly from the packaged dataset
  - `scripts/run_full_stage1_rebuild.sh` can resume sync, extraction, packaging, and HDF5 QA unattended
- Updated on 2026-03-11 later in session:
  - source polygon GPKG is now present locally under `data/`
  - extracted `287` chips from the `287` currently local Stage-1 mosaics
  - built a partial local HDF5 at `outputs/stage1/stage1_positives.h5` with shape `(287, 13, 256, 256)`
  - generated HDF5 QA figure `outputs/stage1/figures/h5_random_samples.png`
  - note: this HDF5 reflects only the currently downloaded mosaics, not the full 4,922-file Drive folder
- Updated again on 2026-03-11 later:
  - resumed the interrupted Stage-1 rebuild
  - completed local Drive sync to `outputs/stage1/mosaics/` with all `4,922` currently exported mosaics
  - restarted `python scripts/04_extract_chips.py` against the full local mosaic set
  - extraction refresh completed successfully for all `4,922` local mosaics
  - rebuilt `outputs/stage1/chip_metadata.csv`
  - rebuilt `outputs/stage1/stage1_positives.h5` with shape `images(4922,13,256,256)` / `masks(4922,256,256)`
  - generated an additional HDF5 spot-check figure from the current partial HDF5:
    `outputs/stage1/figures/h5_random_samples_check_20260311.png`
  - generated a fresh HDF5 spot-check figure from the rebuilt full HDF5:
    `outputs/stage1/figures/h5_random_samples_full_20260311.png`
- Updated on 2026-03-12:
  - baseline GPU job `6591808` (`exp_stage1_resnet50`) remains running on `gpuhost003`
  - resumed the experiment grid by submitting:
    - `6683362` `exp_stage1_efficientnetb0`
    - `6683363` `exp_stage1_mobilenetv2`
    - `6683364` `exp_stage1_resnet50_freeze`
  - `scripts/slurm/submit_stage1_backbone_grid.sh` now skips experiments that are already queued or already have checkpoints unless `RESUBMIT_EXISTING=1`
  - confirmed via TensorFlow logs and live `nvidia-smi` that `6591808` is attached to an A100 GPU, but sampled GPU utilization stayed near `0%` while GPU memory stayed near `39 GB`
  - identified host-side input throughput as a likely epoch-time bottleneck
  - optimized `solar_ml.data.H5SegmentationSequence` to read whole batches from HDF5 instead of one sample at a time
  - updated trainer defaults with `steps_per_execution: 16`, `loader_workers: 4`, and `loader_max_queue_size: 16`
  - local loader benchmark showed about `7x` faster batch assembly after the optimization (`30.6 s/batch` -> `4.4 s/batch`)
- Updated again on 2026-03-12 later:
  - the trainer default input backend is now `tfdata` with full split preload into host memory
  - cached training images are normalized once and stored as `float16` in memory before batching
  - training batches now come from an in-memory `tf.data` pipeline with shuffle, parallel augmentation, fixed-size train batches, and prefetch
  - the old `H5SegmentationSequence` path remains available via `training.input_backend: sequence`
  - local throughput benchmark on a 512-chip train subset showed:
    - `sequence`: about `6.39 s/batch`
    - `tfdata` after preload: about `0.061 s/batch`
    - one-time preload for 512 chips: about `38.6 s`
  - smoke-tested the new path end-to-end through model build and first forward pass with `batch_x=(16,256,256,13)` and `batch_y=(16,256,256,1)`
  - added dedicated rerun config `configs/training/experiments/exp_stage1_resnet50_tfdata.yaml`
  - attempted to submit `exp_stage1_resnet50_tfdata` twice on 2026-03-12 around `23:42 UTC`, but both `sbatch` calls failed with `Unable to contact slurm controller (connect failure)`
  - verified with `scontrol ping` that `Slurmctld(primary) at sc.jc.rl.ac.uk is DOWN` at that time
  - existing running jobs remained visible in `squeue`, but the new `tfdata` rerun was not submitted
- Updated on 2026-03-13:
  - `6925712` (`exp_stage1_resnet50_tfdata`) finished successfully and is now the reference healthy baseline
  - steady-state training after preload ran at about `25-30 s/epoch`
  - best validation Dice reached `0.7610`
  - final test metrics for `6925712` were:
    - Dice `0.7177`
    - IoU `0.6693`
  - cleaned pre-`tfdata` jobs/logs/artifacts to save space before launching the new comparison set
  - active valid overnight comparison jobs are:
    - `6929039` `exp_stage1_resnet50_bce_dice_tfdata`
    - `6931005` `exp_stage1_resnet101_tfdata`
    - `6931039` `exp_stage1_resnet101_freeze_tfdata`
    - `6931072` `exp_stage1_resnet50_freeze_tfdata`
    - `6931085` `exp_stage1_efficientnetb0_tfdata`
  - invalid duplicate runs to ignore in tomorrow's comparison:
    - `6929324`
    - `6929325`
    - `6930992`
  - all three invalid runs fell back to `manual_stage1` / `configs/training/experiments/exp_stage1_resnet50.yaml` because `EXP_ID` and `CONFIG_PATH` were omitted at submission time
  - accuracy-focused follow-up work is now prepared but not yet launched from this execution context:
    - fixed the mixed-precision `bce_dice` loss path by casting combined loss terms to `float32`
    - replaced callback-driven encoder unfreezing with an explicit two-phase fit / recompile flow
    - added phase-aware cosine scheduling support via `unfreeze_learning_rate` and `unfreeze_warmup_epochs`
    - added new configs:
      - `exp_stage1_resnet101_long_tfdata`
      - `exp_stage1_resnet101_bce_dice_tfdata`
      - `exp_stage1_resnet101_freeze_ft_tfdata`
      - `exp_stage1_resnet50_bce_dice_tfdata_v2`
    - local validation passed:
      - `py_compile` for the edited trainer/loss files
      - mixed-precision `bce_dice` returns a `float32` scalar without dtype errors
      - a toy two-phase freeze/unfreeze fit completed and wrote checkpoints
    - all four new configs resolve cleanly through `load_config(...)`
    - attempted `sbatch` submission for all four new experiments, but all failed with `Unable to contact slurm controller (connect failure)`
    - `scontrol ping` from this execution context still reports `Slurmctld(primary) at sc.jc.rl.ac.uk is DOWN`
- Updated on 2026-03-13 later:
  - the user submitted the four prepared accuracy-focused reruns from an interactive shell:
    - `exp_stage1_resnet101_long_tfdata`
    - `exp_stage1_resnet101_bce_dice_tfdata`
    - `exp_stage1_resnet101_freeze_ft_tfdata`
    - `exp_stage1_resnet50_bce_dice_tfdata_v2`
  - current Stage-2 strategy is now explicitly documented in `docs/training_unet_stage1.md`
  - evaluation priority is:
    - Dice / IoU first
    - then precision / recall balance
    - do not use pixel `binary_accuracy` as the main success criterion because it is inflated by background pixels
  - current decision rule after the four runs finish:
    - keep `exp_stage1_resnet101_tfdata` as the reference model unless one of the four reruns beats it materially
    - if `resnet101_long_tfdata` wins, continue tuning schedule and LR around ResNet101
    - if `resnet101_bce_dice_tfdata` wins, continue tuning the loss family and weights around ResNet101
    - if `resnet101_freeze_ft_tfdata` wins, keep two-phase fine-tuning for future pretrained runs
    - if none wins clearly, shift the next iteration away from more hyperparameter churn and toward data improvements and future hard-negative work
- Updated on 2026-03-13 later still:
  - refreshed `python scripts/compare_training_runs.py`; `experiments/run_summary.csv` now includes the newly completed reruns
  - `7253718` `exp_stage1_resnet101_long_tfdata` completed:
    - best validation Dice `0.7544`
    - test Dice `0.7239`
    - test IoU `0.6677`
    - result: worse than the clean `exp_stage1_resnet101_tfdata` baseline
  - `7253785` `exp_stage1_resnet101_bce_dice_tfdata` completed:
    - best validation Dice `0.7909`
    - test Dice `0.7640`
    - test IoU `0.6810`
    - test precision / recall `0.8822 / 0.8369`
    - important caveat: `loss` is `NaN` in `history.csv`, `metrics.json`, and `run_summary.csv`, so this is a promising but not yet fully trusted winner
  - `7253786` `exp_stage1_resnet101_freeze_ft_tfdata` is still running on `gpuhost016`
  - `7253787` `exp_stage1_resnet50_bce_dice_tfdata_v2` is still running on `gpuhost016`
  - current interpretation:
    - do not replace `exp_stage1_resnet101_tfdata` as the clean reference yet
    - wait for the remaining two runs and investigate why the repaired `bce_dice` path still produces `NaN` loss values despite finite segmentation metrics
- Updated on 2026-03-13 later again:
  - added `scripts/evaluate_training_run.py` for offline threshold sweeps and failure galleries on completed experiments
  - added a safer explicit `float32` BCE computation with clipped probabilities inside `solar_ml.losses.get_loss(..., loss_name='bce_dice')`
  - tightened `scripts/train_unet_stage1.py` summary writing so future `metrics.json` files record the best finite validation Dice more robustly
  - updated `scripts/compare_training_runs.py` to recover `best_val_dice` from `history.csv` when older `metrics.json` files contain `NaN`
  - ran threshold-aware test-split evaluation for the two completed ResNet101 runs:
    - `exp_stage1_resnet101_tfdata`
      - best threshold `0.55`
      - pixel Dice `0.8577`
      - pixel IoU `0.7508`
      - precision / recall `0.9032 / 0.8165`
      - missed-chip rate `0.0369`
      - artifacts:
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/threshold_sweep_test.csv`
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/per_chip_test_thr_0.55.csv`
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/failure_gallery_test_thr_0.55.png`
    - `exp_stage1_resnet101_bce_dice_tfdata`
      - best threshold `0.10`
      - pixel Dice `0.8593`
      - pixel IoU `0.7532`
      - precision / recall `0.8746 / 0.8445`
      - missed-chip rate `0.0260`
      - artifacts:
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/threshold_sweep_test.csv`
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/per_chip_test_thr_0.10.csv`
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/failure_gallery_test_thr_0.10.png`
  - threshold-aware interpretation:
    - the BCE/Dice run still looks slightly better on segmentation quality
    - the difference is small, and the BCE/Dice model is calibrated much lower than the focal+dice baseline
    - future comparisons should always include a threshold sweep instead of judging runs only at the default `0.5`
- Updated on 2026-03-13 late night:
  - checked the unfinished follow-up runs from this workspace; `squeue` is now empty and both remaining jobs had already failed
  - inspected `slurm_logs/exp_stage1_resnet101_freeze_ft_tfdata_7253786.runtime.log`
    and `slurm_logs/exp_stage1_resnet50_bce_dice_tfdata_v2_7253787.runtime.log`
  - both jobs died on the first traced training step with:
    - `TypeError: ... tf___safe_probabilities() takes from 0 to 1 positional arguments but 2 were given`
  - root cause was a newer local regression in `solar_ml.losses.py`:
    - the clipped-probability helper introduced for safer BCE handling was breaking AutoGraph under `model.fit(...)`
    - this affected both `bce_dice` and `focal_dice`, so it blocked the BCE rerun and the freeze/fine-tune rerun alike
  - fixed `solar_ml.losses.py` by:
    - inlining the cast+clip logic inside the loss / metric primitives
    - replacing nested closure loss functions with top-level `functools.partial(...)` wrappers
  - validated the repair locally in `tf-gpu` with a mixed-precision smoke fit that completed for:
    - `bce_dice`
    - a frozen `focal_dice` phase followed by recompile + unfrozen fine-tune phase
  - validated syntax with:
    - `/home/users/ruisong/mambaforge_new/envs/tf-gpu/bin/python -m py_compile solar_ml/losses.py scripts/train_unet_stage1.py scripts/compare_training_runs.py scripts/evaluate_training_run.py`
  - attempted to resubmit both failed experiments from this execution context, but `sbatch` calls timed out after 10 s with no response
  - practical consequence:
    - the two repaired experiments still need resubmission from a shell with working Slurm connectivity
- Updated on 2026-03-13 very late:
  - the repaired follow-up experiments were later resubmitted successfully from an interactive shell and completed cleanly:
    - `7345733` `exp_stage1_resnet101_freeze_ft_tfdata`
      - best validation Dice `0.7100`
      - test Dice `0.6816`
      - test IoU `0.6287`
      - test precision / recall `0.7466 / 0.7988`
      - threshold-aware `test` evaluation:
        - best threshold `0.65`
        - pixel Dice `0.7847`
        - pixel IoU `0.6456`
        - missed-chip rate `0.0205`
    - `7345734` `exp_stage1_resnet50_bce_dice_tfdata_v2`
      - best validation Dice `0.6627`
      - test Dice `0.6285`
      - test IoU `0.5962`
      - test precision / recall `0.6687 / 0.7291`
      - important difference versus the earlier broken BCE run: `loss` stayed finite
      - threshold-aware `test` evaluation:
        - best threshold `0.50`
        - pixel Dice `0.6975`
        - pixel IoU `0.5355`
        - missed-chip rate `0.0383`
  - refreshed `experiments/run_summary.csv`
  - wrote new evaluation artifacts:
    - `experiments/exp_stage1_resnet101_freeze_ft_tfdata/evaluation_test/`
    - `experiments/exp_stage1_resnet50_bce_dice_tfdata_v2/evaluation_test/`
  - current interpretation:
    - neither new rerun beat its reference baseline
    - the explicit freeze -> fine-tune hypothesis did not help ResNet101 on this dataset
    - the repaired BCE/Dice path is now numerically stable on ResNet50, but the loss-family gain did not generalize there
    - keep `exp_stage1_resnet101_tfdata` as the clean reference model
    - keep `exp_stage1_resnet101_bce_dice_tfdata` as the only promising quality lead, but do not promote it until the `NaN`-loss anomaly is explained

## Decisions
- Exports are standardized to Google Drive folder: `solar_openEO_temporal_mosaics`.
- Stage-1 extraction exports to Drive folder: `solar_openEO_stage1_v3` (active folder, all chips).
- `rclone` Drive access in this workspace uses remote `google`.
- Stage-2 training uses pure `tf.keras` rather than `segmentation_models` because the latter is broken against the current Keras install on this server.
- Stage-2 model design uses all 13 Sentinel-2 L1C bands, projected through a learnable `1x1` RGB adapter into pretrained 3-channel image backbones.
- Stage-2 experiments are config-driven and prepared for backbone comparison on JASMIN GPU via the `orchid` account.
- Stage-2 default training backend is now an in-memory `tf.data` pipeline; `Sequence` remains as an explicit fallback only.
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
- Stage-2: design negative patch sampling strategy.

## Next Steps
1. **Keep `exp_stage1_resnet101_tfdata` as the clean reference**:
   - the two repaired reruns finished and neither beat the reference baselines
2. **Preserve threshold-aware comparison as the default review path**:
   ```
   python scripts/compare_training_runs.py
   ```
3. **Use offline threshold sweeps and failure galleries for any future candidate**:
   ```
   /home/users/ruisong/mambaforge_new/envs/tf-gpu/bin/python scripts/evaluate_training_run.py \
     --experiment-dir experiments/<exp_id> \
     --split test
   ```
4. **Judge winners by segmentation quality, not pixel accuracy**:
   - prioritize Dice and IoU
   - then check precision / recall balance
5. **Resolve the `bce_dice` anomaly before promoting it to the new default direction**:
   - `exp_stage1_resnet101_bce_dice_tfdata` currently leads on test Dice
   - but its train / val / test `loss` is `NaN`, so confirm whether this is a logging bug or a real numerical instability
6. **Shift the next iteration toward data and failure analysis rather than more backbone churn**:
   - inspect the new failure galleries for the clean ResNet101 baseline, the ResNet101 BCE/Dice run, and the two repaired reruns
   - prioritize harder examples and future hard-negative construction
7. **When submitting manually, always set env vars explicitly**:
   ```
   env EXP_ID=... CONFIG_PATH=... EXTRA_ARGS=-v sbatch --job-name=... scripts/slurm/train_stage1_gpu.sh
   ```
8. **Only if you intentionally need duplicate reruns**, override the resume guard:
   ```
   RESUBMIT_EXISTING=1 scripts/slurm/submit_stage1_backbone_grid.sh
   ```
11. **If none of the remaining clean reruns wins clearly, shift the next effort toward data**:
   - inspect qualitative prediction failures
   - prioritize harder examples and future hard-negative extraction
   - avoid opening a larger backbone sweep before that
12. **Optional Stage-1 QA refresh**:
   ```
   conda run -n geospatial python scripts/06_visualize_chips.py
   ```

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
- 2026-03-11: Verified active Drive folder `google:solar_openEO_stage1_v3` exists and contains 4,922 `.tif` mosaics. Updated repo defaults from `solar_openEO_stage1_mosaics` to `solar_openEO_stage1_v3` and replaced stale hardcoded `rclone` remote `rui.song90:` with configurable remote `google`.
- 2026-03-11: Added Stage-2 training stack:
  - `solar_ml/` package for config loading, HDF5 data access, loss functions, and pretrained-backbone U-Net construction
  - `scripts/prepare_stage1_training_data.py` to build spatial splits and band stats from `stage1_positives.h5`
  - `scripts/train_unet_stage1.py` for config-driven training and evaluation
  - `scripts/slurm/train_stage1_gpu.sh` and `scripts/slurm/submit_stage1_backbone_grid.sh` for JASMIN GPU runs via `orchid`
  - `scripts/compare_training_runs.py` for experiment comparison
  - `scripts/sync_stage1_v3_mosaics.py` for bulk Drive download
  - `docs/training_unet_stage1.md` documenting the workflow
  - validated the training stack with a synthetic HDF5 dry run in `tf-gpu`
  - could not start real supervised training because no local Stage-1 HDF5 or source polygon GPKG was found
- 2026-03-11: Added local-recovery helpers for rebuilding Stage-1 artifacts:
  - `scripts/04_extract_chips.py` now falls back to reconstructing the unique chip list from downloaded `outputs/stage1/mosaics/*.tif` filenames when `chip_manifest.csv` is absent
  - `scripts/05_package_dataset.py` now supports packaging directly from a `chips/` directory by scanning `*_image.tif` / `*_mask.tif` pairs and can optionally write a recovered `chip_metadata.csv`
  - added `scripts/08_visualize_h5_samples.py` to render random RGB and RGB+mask overlays from `stage1_positives.h5`
  - confirmed `rclone` download from `google:solar_openEO_stage1_v3` works with unsandboxed execution; interrupted after 287 mosaics (~538 MiB) because the polygon GPKG is still missing locally
- 2026-03-11: User later restored the polygon label data locally:
  - found `data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg`
  - ran `python scripts/04_extract_chips.py` using the new manifest-from-mosaics fallback
  - extracted 287/287 currently downloaded mosaics into `outputs/stage1/chips/`
  - wrote `outputs/stage1/chip_metadata.csv`
  - packaged `outputs/stage1/stage1_positives.h5` with `images(287,13,256,256)` and `masks(287,256,256)`
  - generated `outputs/stage1/figures/h5_random_samples.png` for RGB and RGB+mask QA
- 2026-03-11: Continued toward full Stage-1 rebuild:
  - verified old `segmentation_models` path is still broken in `tf-gpu` with `AttributeError: module 'keras.utils' has no attribute 'generic_utils'`
  - verified maintained backbone path works: direct `tf.keras.applications.ResNet50(weights="imagenet")` loads successfully
  - verified end-to-end dry-run with `scripts/train_unet_stage1.py --config configs/training/experiments/exp_stage1_resnet50.yaml --dry-run -v` succeeds once split/stats inputs exist
  - added `scripts/run_full_stage1_rebuild.sh` to automate sync -> extract -> package -> HDF5 QA
  - launched unattended rebuild in background; current log path: `outputs/stage1/logs/full_stage1_rebuild.log`
- 2026-03-11: Resumed the interrupted Stage-1 rebuild from local state:
  - reran `python scripts/sync_stage1_v3_mosaics.py` and completed local sync to `outputs/stage1/mosaics/` with `4,922` GeoTIFFs
  - restarted `python scripts/04_extract_chips.py`; the refresh was still running at session end, so `chip_metadata.csv` and `stage1_positives.h5` had not yet been regenerated from the full local mosaic set
- 2026-03-11: Generated a fresh HDF5 QA figure from the current partial packaged dataset:
  - `python scripts/08_visualize_h5_samples.py --h5 outputs/stage1/stage1_positives.h5 --n 8 --seed 20260311 --out outputs/stage1/figures/h5_random_samples_check_20260311.png`
  - confirmed the current packaged HDF5 still has shape `images(287,13,256,256)` / `masks(287,256,256)` while full extraction is still running
- 2026-03-11: Completed the full local Stage-1 rebuild from the synced mosaic set:
  - `python scripts/04_extract_chips.py` finished with `4,922` extracted chips from `4,922` local mosaics
  - `python scripts/05_package_dataset.py` rebuilt `outputs/stage1/stage1_positives.h5` with shape `images(4922,13,256,256)` / `masks(4922,256,256)`
  - generated `outputs/stage1/figures/h5_random_samples_full_20260311.png` from the rebuilt full HDF5
- 2026-03-11: Generated individual random HDF5 sample PNGs from the rebuilt full dataset:
  - folder: `outputs/stage1/figures/h5_random_individuals_20260311/`
  - each sampled chip has paired `*_rgb.png` and `*_rgb_mask.png` files
- 2026-03-11: Prepared Stage-2 training inputs from the rebuilt full HDF5:
  - ran `python scripts/prepare_stage1_training_data.py --h5 outputs/stage1/stage1_positives.h5 --out-dir outputs/training_prep/stage1_v3`
  - wrote `outputs/training_prep/stage1_v3/split_manifest.csv`
  - wrote `outputs/training_prep/stage1_v3/split_summary.json`
  - wrote `outputs/training_prep/stage1_v3/band_stats.npz`
  - wrote `outputs/training_prep/stage1_v3/band_stats_summary.json`
  - split sizes: train `3357`, val `834`, test `731`
- 2026-03-11: Verified Stage-2 trainer still works against the rebuilt full dataset:
  - ran `python scripts/train_unet_stage1.py --config configs/training/experiments/exp_stage1_resnet50.yaml --dry-run -v`
  - dry-run completed successfully with `batch_x=(16, 256, 256, 13)`, `batch_y=(16, 256, 256, 1)`, `preds=(2, 256, 256, 1)`
- 2026-03-11: Fixed the Slurm GPU wrapper for Stage-2 training:
  - `scripts/slurm/train_stage1_gpu.sh` now writes an explicit runtime log and `ERR` trap message
  - made NVIDIA library-path discovery best-effort (`find ... || true`) so the wrapper no longer exits before startup on `gpuhost003`
- 2026-03-11: Submitted the first real baseline GPU run:
  - job `6591808` on `gpuhost003`
  - experiment id `exp_stage1_resnet50`
  - confirmed from runtime log that the job reaches real training on an `NVIDIA A100-SXM4-40GB`
  - observed live training progress in epoch 1 reaching at least `28/210` batches with improving early metrics
- 2026-03-12: Resumed the unfinished Stage-2 experiment queue:
  - confirmed `6591808` is still running and has already written `best.weights.h5` / `latest.weights.h5`
  - updated `scripts/slurm/submit_stage1_backbone_grid.sh` so reruns skip queued or checkpointed experiments by default
  - submitted pending jobs `6683362` (`exp_stage1_efficientnetb0`), `6683363` (`exp_stage1_mobilenetv2`), and `6683364` (`exp_stage1_resnet50_freeze`)
- 2026-03-12: Investigated slow epoch time for `exp_stage1_resnet50`:
  - verified TensorFlow created `GPU:0` on `NVIDIA A100-SXM4-40GB` and loaded CUDA/cuDNN
  - verified with `srun --jobid 6591808 --overlap nvidia-smi ...` that the training Python process holds about `38.9 GB` GPU memory
  - sampled GPU utilization repeatedly at `0%`, indicating the GPU is allocated but frequently idle
  - optimized Stage-2 input throughput in `solar_ml/data.py` and `scripts/train_unet_stage1.py`
  - local benchmark against the full HDF5 showed approximately `7.0x` faster batch assembly for the loader path
- 2026-03-12: Submitted a like-for-like throughput benchmark run:
  - added `configs/training/experiments/exp_stage1_resnet50_throughputopt.yaml`
  - kept the model identical to `exp_stage1_resnet50` but pinned `steps_per_execution: 16`, `loader_workers: 4`, `loader_use_multiprocessing: false`, and `loader_max_queue_size: 16`
  - submitted Slurm job `6705773` as experiment `exp_stage1_resnet50_throughputopt`
  - confirmed the runtime log on `gpuhost010` reached `Epoch 1/60` on `NVIDIA A100-SXM4-40GB`
- 2026-03-12: Reworked the Stage-2 input pipeline again to address the remaining GPU-underfed behavior:
  - `solar_ml/data.py` now exposes `load_h5_split_arrays(...)` and `build_in_memory_dataset(...)`
  - `scripts/train_unet_stage1.py` now defaults to `training.input_backend: tfdata` and bulk-loads each split once, then trains from an in-memory `tf.data` pipeline
  - new default training knobs in `configs/training/base.yaml`:
    - `input_backend: tfdata`
    - `in_memory_image_dtype: float16`
    - `drop_remainder: true`
    - `shuffle_buffer: 2048`
    - `tfdata_num_parallel_calls: 8`
    - `tfdata_prefetch_buffer: 4`
  - verified the new backend with a model forward-pass smoke test on real Stage-1 data
  - local subset benchmark on 512 training chips:
    - `H5SegmentationSequence`: about `6.39 s/batch`
    - in-memory `tf.data`: about `0.061 s/batch` after preload
    - preload cost: about `38.6 s` one time for 512 chips
- 2026-03-12: Prepared but could not launch a fresh GPU rerun for the new backend:
  - added `configs/training/experiments/exp_stage1_resnet50_tfdata.yaml`
  - attempted `sbatch` submission twice for `exp_stage1_resnet50_tfdata`
  - both submissions failed with `Unable to contact slurm controller (connect failure)`
  - confirmed `scontrol ping` reported `Slurmctld(primary) at sc.jc.rl.ac.uk is DOWN`
  - no new Slurm job id was assigned; `experiments/runs.csv` was intentionally left unchanged
- 2026-03-13: Follow-up training observations and fixes:
  - `6925712` (`exp_stage1_resnet50_tfdata`) confirmed the new backend is healthy:
    - one-time preload completed
    - steady-state epochs dropped to about `25-30 s/epoch`
    - validation Dice reached about `0.761`
  - fixed the `efficientnetb0` decoder skip wiring in `solar_ml/model.py`; the backbone now builds successfully instead of failing with a spatial shape mismatch
  - discovered that user-submitted jobs `6929324` and `6929325` were launched without the intended `EXP_ID` / `CONFIG_PATH` env vars:
    - both logs show `Experiment ID : manual_stage1`
    - both are running `configs/training/experiments/exp_stage1_resnet50.yaml`
    - they should be treated as duplicate baseline ResNet50 runs rather than valid ResNet101 / MobileNetV2 comparisons
- 2026-03-13: Prepared the overnight comparison set for next-day review:
  - `6925712` (`exp_stage1_resnet50_tfdata`) completed and established the current best reference run
    - best validation Dice: `0.7610`
    - test Dice: `0.7177`
    - test IoU: `0.6693`
  - valid active comparison jobs:
    - `6929039` `exp_stage1_resnet50_bce_dice_tfdata`
    - `6931005` `exp_stage1_resnet101_tfdata`
    - `6931039` `exp_stage1_resnet101_freeze_tfdata`
    - `6931072` `exp_stage1_resnet50_freeze_tfdata`
    - `6931085` `exp_stage1_efficientnetb0_tfdata`
  - invalid duplicate jobs to exclude from any comparison table:
    - `6929324`
    - `6929325`
    - `6930992`
- 2026-03-13: Final interpretation after all comparison jobs completed:
  - best completed model: `exp_stage1_resnet101_tfdata`
    - best validation Dice from `history.csv`: `0.7657` at epoch `40`
    - test Dice: `0.7399`
    - test IoU: `0.6902`
  - second-best completed model: `exp_stage1_resnet50_tfdata`
    - best validation Dice from `history.csv`: `0.7610` at epoch `38`
    - test Dice: `0.7177`
    - test IoU: `0.6693`
  - `exp_stage1_efficientnetb0_tfdata` completed but underperformed badly:
    - best validation Dice: `0.4496`
    - test Dice: `0.4459`
    - test precision/recall imbalance suggests strong over-segmentation with many false positives
  - `exp_stage1_resnet50_bce_dice_tfdata` is not a valid result:
    - crashed at epoch 1 with a mixed-precision dtype mismatch inside `solar_ml.losses.bce_dice`
  - `exp_stage1_resnet101_freeze_tfdata` and `exp_stage1_resnet50_freeze_tfdata` are not valid results:
    - both crashed immediately after encoder unfreezing at epoch 5 with `TypeError: 'NoneType' object is not callable`
    - freeze-path training logic likely needs a code fix before rerunning
- 2026-03-13: Accuracy-focused rerun preparation:
  - fixed `solar_ml.losses.get_loss(...)` so `bce_dice` and `focal_dice` combine loss terms in `float32` under mixed precision
  - refactored `scripts/train_unet_stage1.py` freeze training into explicit phase-1 frozen training plus phase-2 unfrozen fine-tuning
  - phase-2 scheduling now supports a lower `unfreeze_learning_rate` and optional `unfreeze_warmup_epochs`
  - added configs for the next comparison batch:
    - `configs/training/experiments/exp_stage1_resnet101_long_tfdata.yaml`
    - `configs/training/experiments/exp_stage1_resnet101_bce_dice_tfdata.yaml`
    - `configs/training/experiments/exp_stage1_resnet101_freeze_ft_tfdata.yaml`
    - `configs/training/experiments/exp_stage1_resnet50_bce_dice_tfdata_v2.yaml`
  - validated locally with:
    - `python -m py_compile solar_ml/losses.py scripts/train_unet_stage1.py`
    - a mixed-precision loss smoke test for `bce_dice`
    - a toy two-phase fit that completed both frozen and unfrozen phases successfully
  - attempted to submit all four experiments, but `sbatch` failed for each with controller-connect errors and no job ids were assigned
