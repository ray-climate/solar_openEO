# Stage-1 U-Net Training

## Purpose
This document describes the current machine-learning training path for solar panel segmentation from Stage-1 Sentinel-2 chips.

## Current Design
- Training input: `outputs/stage1/stage1_positives.h5`
- Image tensor: `(N, 13, 256, 256)` from Sentinel-2 L1C spectral bands only
- Mask tensor: `(N, 256, 256)` binary panel mask
- Model: U-Net decoder with pretrained `tf.keras.applications` backbone
- Backbone input adaptation: a learnable `1x1` RGB adapter projects 13 bands to 3 channels
- Initial RGB adapter weights map Sentinel-2 `B4/B3/B2` to ImageNet-style `R/G/B`
- Default split strategy: deterministic spatial-block split from chip IDs to reduce geographic leakage

## Why This Design
- Keeps all 13 Sentinel-2 bands in the training signal
- Reuses strong pretrained backbones without hacking the first convolution weights directly
- Avoids the broken `segmentation_models` dependency currently present on this server
- Makes backbone comparisons easy through config-only experiment files

## Training Preparation
1. Build or locate the Stage-1 HDF5 dataset:
   - `outputs/stage1/stage1_positives.h5`
2. Prepare split manifest and normalization stats:
   ```bash
   python scripts/prepare_stage1_training_data.py
   ```
3. Spot-check random packaged samples before training:
   ```bash
   python scripts/08_visualize_h5_samples.py \
     --h5 outputs/stage1/stage1_positives.h5 \
     --n 6
   ```
4. Dry-run model + first batch:
   ```bash
   python scripts/train_unet_stage1.py \
     --config configs/training/experiments/exp_stage1_resnet50.yaml \
     --dry-run -v
   ```

## GPU Training
Single run:
```bash
env EXP_ID=exp_stage1_resnet50 \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet50.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet50 scripts/slurm/train_stage1_gpu.sh
```

Like-for-like throughput comparison run:
```bash
env EXP_ID=exp_stage1_resnet50_throughputopt \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet50_throughputopt.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet50_throughputopt scripts/slurm/train_stage1_gpu.sh
```

Pinned rerun for the new in-memory `tf.data` backend:
```bash
env EXP_ID=exp_stage1_resnet50_tfdata \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet50_tfdata.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet50_tfdata scripts/slurm/train_stage1_gpu.sh
```

Prepared accuracy-focused reruns:
```bash
env EXP_ID=exp_stage1_resnet101_long_tfdata \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet101_long_tfdata.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet101_long_tfdata scripts/slurm/train_stage1_gpu.sh

env EXP_ID=exp_stage1_resnet101_bce_dice_tfdata \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet101_bce_dice_tfdata.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet101_bce_dice_tfdata scripts/slurm/train_stage1_gpu.sh

env EXP_ID=exp_stage1_resnet101_freeze_ft_tfdata \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet101_freeze_ft_tfdata.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet101_freeze_ft_tfdata scripts/slurm/train_stage1_gpu.sh

env EXP_ID=exp_stage1_resnet50_bce_dice_tfdata_v2 \
  CONFIG_PATH=configs/training/experiments/exp_stage1_resnet50_bce_dice_tfdata_v2.yaml \
  EXTRA_ARGS=-v \
  sbatch --job-name=exp_stage1_resnet50_bce_dice_tfdata_v2 scripts/slurm/train_stage1_gpu.sh
```

Important:
- `scripts/slurm/train_stage1_gpu.sh` falls back to `EXP_ID=manual_stage1` and `CONFIG_PATH=configs/training/experiments/exp_stage1_resnet50.yaml` if those env vars are omitted.
- For manual submissions, prefer the `env ... sbatch ...` pattern above.

Grid run:
```bash
# Safe to resume: skips queued or checkpointed experiments by default.
DRY_RUN=1 scripts/slurm/submit_stage1_backbone_grid.sh
scripts/slurm/submit_stage1_backbone_grid.sh
```

## Result Comparison
After runs finish:
```bash
python scripts/compare_training_runs.py
```

This writes:
- `experiments/run_summary.csv`

Threshold-aware offline review for a completed run:
```bash
/home/users/ruisong/mambaforge_new/envs/tf-gpu/bin/python scripts/evaluate_training_run.py \
  --experiment-dir experiments/exp_stage1_resnet101_tfdata \
  --split test
```

This writes under the experiment directory:
- `evaluation_test/threshold_sweep_test.csv`
- `evaluation_test/per_chip_test_thr_<best>.csv`
- `evaluation_test/failure_gallery_test_thr_<best>.png`
- `evaluation_test/summary_test.json`

## Current Optimization Strategy
- Treat Dice and IoU as the primary segmentation targets. Do not optimize for pixel `binary_accuracy`; that metric is already very high because most pixels are background and is not a useful proxy for panel delineation quality.
- Keep the input pipeline fixed at the current in-memory `tf.data` backend while comparing model quality. Throughput is no longer the main bottleneck, so avoid mixing loader changes into the current accuracy study.
- Keep `exp_stage1_resnet101_tfdata` as the clean reference model. `exp_stage1_resnet101_bce_dice_tfdata` now has the best completed test Dice, but its recorded `loss` is `NaN`, so it is not yet a clean replacement baseline.
- Prioritize a small number of high-signal levers before opening a wider sweep:
  - longer schedule / lower floor LR
  - alternate loss (`bce_dice`)
  - explicit frozen warmup followed by lower-LR fine-tuning
  - keep backbone search centered on the ResNet family unless a strong reason emerges otherwise
- Deprioritize batch-size tuning for now. Batch size can affect optimization, but it is less likely than loss or schedule changes to explain the current quality ceiling.

### Submitted Follow-Up Jobs
- `exp_stage1_resnet101_long_tfdata`
  - hypothesis: the current best model was undertrained and benefits from a longer, gentler cosine schedule
  - status on 2026-03-13 21:10 UTC: completed, but worse than the clean ResNet101 baseline
- `exp_stage1_resnet101_bce_dice_tfdata`
  - hypothesis: the main limitation is loss shape rather than architecture
  - status on 2026-03-13 21:10 UTC: completed with the best test Dice so far, but all recorded `loss` values are `NaN`
- `exp_stage1_resnet101_freeze_ft_tfdata`
  - hypothesis: explicit warmup + lower-LR fine-tuning improves adaptation of the pretrained encoder
  - status on 2026-03-13 very late: repaired rerun `7345733` completed cleanly, but it was not competitive with the clean ResNet101 baseline
  - result summary:
    - best validation Dice `0.7100`
    - test Dice `0.6816`
    - threshold-swept test Dice `0.7847` at threshold `0.65`
- `exp_stage1_resnet50_bce_dice_tfdata_v2`
  - hypothesis: the repaired loss path helps more generally and is not specific to ResNet101
  - status on 2026-03-13 very late: repaired rerun `7345734` completed cleanly with finite loss values, but it was much worse than the clean ResNet50 baseline
  - result summary:
    - best validation Dice `0.6627`
    - test Dice `0.6285`
    - threshold-swept test Dice `0.6975` at threshold `0.50`

### What To Do Next
1. Keep `exp_stage1_resnet101_tfdata` as the clean reference model.
2. Keep `exp_stage1_resnet101_bce_dice_tfdata` as the only promising quality lead, but do not promote it until the `NaN`-loss anomaly is understood.
3. Refresh the summary table when new results appear:
   ```bash
   python scripts/compare_training_runs.py
   ```
4. Run threshold-aware offline review for every serious candidate:
   ```bash
   /home/users/ruisong/mambaforge_new/envs/tf-gpu/bin/python scripts/evaluate_training_run.py \
     --experiment-dir experiments/<exp_id> \
     --split test
   ```
5. Treat a result as a meaningful win only if it beats the clean `exp_stage1_resnet101_tfdata` baseline on Dice / IoU and holds up under threshold sweeping.
6. Do not keep expanding the freeze/fine-tune or ResNet50 BCE/Dice branches right now:
   - `exp_stage1_resnet101_freeze_ft_tfdata` lost clearly to the reference
   - `exp_stage1_resnet50_bce_dice_tfdata_v2` proved the repaired loss is stable, but not good
7. Shift the next effort toward qualitative failure review, harder examples, and future hard-negative construction.

### Near-Term Decision Rule
- If a submitted run improves Dice by roughly `0.01-0.02` or more while keeping precision/recall reasonable, treat that axis as the next tuning direction.
- If gains are smaller than that or unstable, assume the model is reaching the current dataset ceiling and move the next iteration toward data rather than more optimizer/backbone churn.

## Current Status
- Full local Stage-1 packaged dataset is available:
  - `outputs/stage1/stage1_positives.h5`
  - current shape: `images(4922,13,256,256)` and `masks(4922,256,256)`
- Training prep artifacts are available under `outputs/training_prep/stage1_v3/`.
- Trainer defaults now include host-side throughput improvements:
  - default `training.input_backend: tfdata`
  - full split preload into host memory, with transpose/normalization done once up front
  - cached image dtype `float16`
  - parallel augmentation and prefetch from an in-memory `tf.data` pipeline
  - `steps_per_execution: 16`
  - `drop_remainder: true`
  - `shuffle_buffer: 2048`
  - `tfdata_num_parallel_calls: 8`
  - `tfdata_prefetch_buffer: 4`
  - fallback backend remains available with `training.input_backend: sequence`
- Finished baseline as of 2026-03-13:
  - `6925712` `exp_stage1_resnet50_tfdata`
  - healthy steady-state throughput after preload: about `25-30 s/epoch`
  - best validation Dice: `0.7610`
  - test Dice: `0.7177`
  - test IoU: `0.6693`
- Refreshed comparison snapshot on 2026-03-13 21:10 UTC:
  - threshold-aware evaluation was added on 2026-03-13 later:
    - baseline `exp_stage1_resnet101_tfdata` test split:
      - best threshold in sweep: `0.55`
      - pixel Dice `0.8577`
      - pixel IoU `0.7508`
      - precision / recall `0.9032 / 0.8165`
      - missed-chip rate `0.0369`
      - artifacts:
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/threshold_sweep_test.csv`
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/per_chip_test_thr_0.55.csv`
        - `experiments/exp_stage1_resnet101_tfdata/evaluation_test/failure_gallery_test_thr_0.55.png`
    - `exp_stage1_resnet101_bce_dice_tfdata` test split:
      - best threshold in sweep: `0.10`
      - pixel Dice `0.8593`
      - pixel IoU `0.7532`
      - precision / recall `0.8746 / 0.8445`
      - missed-chip rate `0.0260`
      - artifacts:
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/threshold_sweep_test.csv`
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/per_chip_test_thr_0.10.csv`
        - `experiments/exp_stage1_resnet101_bce_dice_tfdata/evaluation_test/failure_gallery_test_thr_0.10.png`
      - interpretation:
        - the BCE/Dice run still wins slightly on segmentation quality
        - but it is calibrated very differently from the focal+dice baseline, so threshold tuning must be part of future comparisons
  - highest completed test-Dice run: `exp_stage1_resnet101_bce_dice_tfdata`
    - best validation Dice: `0.7909`
    - test Dice: `0.7640`
    - test IoU: `0.6810`
    - test precision / recall: `0.8822 / 0.8369`
    - caveat: train / validation / test `loss` is `NaN`, so this run is promising but not yet a clean baseline replacement
  - best clean completed run: `exp_stage1_resnet101_tfdata`
    - best validation Dice from `history.csv`: `0.7657`
    - test Dice: `0.7399`
    - test IoU: `0.6902`
  - `exp_stage1_resnet101_long_tfdata` completed but underperformed:
    - best validation Dice: `0.7544`
    - test Dice: `0.7239`
    - test IoU: `0.6677`
  - next clean completed baseline: `exp_stage1_resnet50_tfdata`
    - best validation Dice: `0.7610`
    - test Dice: `0.7177`
    - test IoU: `0.6693`
  - `exp_stage1_efficientnetb0_tfdata` completed but is not competitive:
    - best validation Dice: `0.4496`
    - test Dice: `0.4459`
    - high recall / low precision indicates many false positives
  - `exp_stage1_resnet50_bce_dice_tfdata` failed at epoch 1 due to a mixed-precision dtype mismatch in the custom loss
  - `exp_stage1_resnet101_freeze_tfdata` and `exp_stage1_resnet50_freeze_tfdata` both failed when the encoder was unfrozen at epoch 5
- The failed `bce_dice` and freeze/unfreeze code paths are now fixed locally:
  - custom losses now combine terms in `float32` under mixed precision
  - freeze training now uses an explicit frozen phase plus an unfrozen fine-tune phase instead of recompiling inside a callback
  - fine-tune configs can now set `unfreeze_learning_rate` and `unfreeze_warmup_epochs`
- A later local regression in `solar_ml.losses.py` was also fixed:
  - the clipped-probability helper was breaking AutoGraph during real `model.fit(...)`
  - cast+clip is now inlined inside the loss / metric primitives
  - `get_loss(...)` now returns top-level `functools.partial(...)` loss callables instead of nested closures
  - a mixed-precision `tf-gpu` smoke fit now succeeds for both the BCE/Dice path and the frozen -> unfrozen focal+dice path
- Prepared next experiments to try for accuracy gains:
  - `exp_stage1_resnet101_long_tfdata`
  - `exp_stage1_resnet101_bce_dice_tfdata`
  - `exp_stage1_resnet101_freeze_ft_tfdata`
  - `exp_stage1_resnet50_bce_dice_tfdata_v2`
- User later submitted all four accuracy-focused reruns from an interactive shell with working Slurm access.
- The first job IDs for the last two reruns did fail during epoch-1 startup:
  - `7253786` `exp_stage1_resnet101_freeze_ft_tfdata`
  - `7253787` `exp_stage1_resnet50_bce_dice_tfdata_v2`
- The repaired reruns later completed successfully as:
  - `7345733` `exp_stage1_resnet101_freeze_ft_tfdata`
  - `7345734` `exp_stage1_resnet50_bce_dice_tfdata_v2`
- Both repaired reruns underperformed their baselines, so neither axis is the next default direction.
- Use `RESUBMIT_EXISTING=1` only when you explicitly want to rerun an experiment that is already queued or has checkpoints on disk.

## Throughput Note
- The original trainer was underfeeding the GPU:
  - live `nvidia-smi` confirmed the process occupied about `39 GB` on the A100 while sampled GPU utilization stayed near `0%`
  - the prior fit loop used `workers=1` and Python-side per-sample HDF5 reads
- A local like-for-like loader benchmark on 2026-03-12 showed about `7x` faster batch assembly after vectorized HDF5 reads:
  - old path: about `30.6 s/batch`
  - new path: about `4.4 s/batch`
- A later local subset benchmark on 2026-03-12 showed the in-memory `tf.data` path is much faster than the current `Sequence` backend in steady state:
  - benchmark subset: 512 train chips
  - `H5SegmentationSequence`: about `6.39 s/batch`
  - in-memory `tf.data`: about `0.061 s/batch` after preload
  - one-time preload for that 512-chip subset: about `38.6 s`
- Important: already-running jobs keep using the old in-memory code. The speedup only applies to jobs started after the code change.
- The in-memory preload adds startup time, but it is paid once per job instead of once per batch.
- For an apples-to-apples Slurm comparison, use `exp_stage1_resnet50_throughputopt`, which keeps the ResNet50 baseline model the same and can be rerun after the input-backend change for a fresh wall-clock comparison.
