#!/usr/bin/env python
"""
Continuous experiment pipeline manager.

Maintains up to MAX_JOBS Slurm jobs running at all times by working through
a priority-ordered experiment list. Safe to re-run repeatedly — skips
experiments that already have results or are already queued.

Usage:
    python scripts/auto_submit_pipeline.py [--dry-run] [--max-jobs N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_FILE = REPO / "slurm_logs" / "auto_submit_pipeline.log"
EXPERIMENTS_DIR = REPO / "experiments"
CONFIGS_DIR = REPO / "configs" / "training" / "experiments"
SLURM_SCRIPT = REPO / "scripts" / "slurm" / "train_stage1_gpu.sh"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority-ordered experiment list.
# Add new experiments here — they will be submitted in order as slots open.
# ---------------------------------------------------------------------------
EXPERIMENT_QUEUE = [
    # Tier 1: Tversky loss variants (highest expected impact after dice win)
    "exp_stage1_r101_tversky_recall",       # α=0.3 β=0.7 — recall-focused
    "exp_stage1_r101_focal_tversky",        # Focal Tversky γ=0.75
    "exp_stage1_r101_tversky_balanced",     # α=β=0.5 — generalised dice
    "exp_stage1_r101_tversky_precise",      # α=0.7 β=0.3 — precision-focused

    # Tier 2: Backbone with dice loss
    "exp_stage1_r152_dice",                 # ResNet152 — deeper encoder
    "exp_stage1_r50_dice",                  # ResNet50 — is 101 necessary?

    # Tier 3: Training hyperparameters with dice
    "exp_stage1_r101_dice_lr001",           # LR=0.001 — dice has different gradient dynamics
    "exp_stage1_r101_dice_dropout20",       # Decoder dropout 0.2 — more regularisation
    "exp_stage1_r101_dice_zscore",          # Z-score normalisation

    # Wave 2 (already submitted — will be skipped if done)
    "exp_stage1_r101_dice_spectral",
    "exp_stage1_r101_dice_6band",
    "exp_stage1_r101_dice_longer",
    "exp_stage1_r101_dice_spectral_6band",

    # Tier 4: best model (dice_longer) + remaining improvements
    "exp_stage1_r101_dice_longer_spectral",       # dice+longer + spectral aug
    "exp_stage1_r101_dice_longer_6band",          # dice+longer + SWIR 6-band
    "exp_stage1_r152_dice_longer",                # ResNet152 + dice + longer
    "exp_stage1_r101_dice_longer_spectral_6band", # dice+longer + spectral + 6-band

    # Tier 5: zscore is a massive win (+5pp) — explore combinations
    "exp_stage1_r101_dice_zscore_longer",         # zscore + 100 epochs (highest priority)
    "exp_stage1_r101_dice_zscore_6band",          # zscore + SWIR 6-band
    "exp_stage1_r101_dice_zscore_spectral",       # zscore + spectral aug
    "exp_stage1_r101_dice_zscore_longer_6band",   # zscore + longer + 6-band
    "exp_stage1_r101_dice_zscore_longer_spectral",# zscore + longer + spectral

    # Tier 6: best combos from Tier 5 + new backbones
    "exp_stage1_r152_dice_zscore",                # ResNet152 + zscore
    "exp_stage1_efficientnetb0_dice_zscore",      # EfficientNetB0 + zscore (untested backbone)
    "exp_stage1_r101_dice_zscore_6band_spectral", # zscore + 6band + spectral (triple combo)
    "exp_stage1_r50_dice_zscore_longer",          # R50 + zscore + longer — lighter model check

    # Tier 7: zscore_longer follow-ups + new backbones
    "exp_stage1_r152_dice_zscore_longer",         # R152 + zscore + longer
    "exp_stage1_efficientnetb0_dice_zscore_longer", # EfficientNetB0 + zscore + longer
    "exp_stage1_r101_dice_zscore_longer_bs16",    # batch_size 16 — does it help?
    "exp_stage1_r101_dice_zscore_longer_spectral_fixed",       # spectral aug with bug fixed (no clip)
    "exp_stage1_mobilenetv2_dice_zscore_longer",               # MobileNetV2 — lightweight backbone test
    "exp_stage1_r101_dice_zscore_longer_6band_spectral_fixed", # triple combo: zscore+longer+6band+spectral (fixed)
    "exp_stage1_r101_dice_zscore_longer_wd0",                  # no weight decay — is L2 reg helping?

    # Tier 8: band ratios (NDVI, NDBI, NDWI) as extra input channels
    "exp_stage1_r101_dice_zscore_longer_bandratios",           # best model + band ratios
    "exp_stage1_r101_dice_zscore_bandratios",                  # zscore + band ratios (50-epoch baseline)
    "exp_stage1_r101_dice_zscore_longer_6band_bandratios",     # zscore+longer+6band+ratios
    "exp_stage1_r50_dice_zscore_longer_bandratios",            # R50 + zscore+longer+ratios (lightweight)
    "exp_stage1_mobilenetv2_dice_zscore_longer_bandratios",    # MobileNetV2 + band ratios
    "exp_stage1_r152_dice_zscore_longer_bandratios",           # R152 + band ratios
    "exp_stage1_r101_dice_zscore_longer_bandratios_spectral",  # band ratios + spectral aug
    "exp_stage1_r101_dice_zscore_longer_bandratios_6band_spectral", # band ratios + 6band + spectral

    # Tier 9: Attention U-Net (Oktay et al. 2018)
    "exp_stage1_r101_dice_zscore_longer_attention",           # attention gates on best model
    "exp_stage1_r50_dice_zscore_longer_attention",            # attention + R50 (lightweight)
    "exp_stage1_r101_dice_zscore_longer_attention_bandratios",# attention + band ratios combined
    "exp_stage1_r101_dice_zscore_longer_attention_6band",     # attention + 6-band
    "exp_stage1_r152_dice_zscore_longer_attention",           # R152 + attention

    # Tier 10: SE (Squeeze-and-Excite) channel attention + extended training
    "exp_stage1_r101_dice_zscore_longer150",                  # 150 epochs — squeeze last gains
    "exp_stage1_r101_dice_zscore_longer_se",                  # SE channel attention on best model
    "exp_stage1_r50_dice_zscore_longer_se",                   # SE + R50 (lightweight)
    "exp_stage1_r101_dice_zscore_longer_se_attention",        # SE + spatial attention combined

    # Tier 11: ensemble seeds — robustness + model averaging
    "exp_stage1_r101_dice_zscore_longer_seed1",
    "exp_stage1_r101_dice_zscore_longer_seed2",
    "exp_stage1_r101_dice_zscore_longer_seed3",
    "exp_stage1_r50_dice_zscore_longer_seed1",
    "exp_stage1_r50_dice_zscore_longer_seed2",
    "exp_stage1_r50_dice_zscore_longer_seed3",
    "exp_stage1_mobilenetv2_dice_zscore_longer_seed1",
    "exp_stage1_r101_dice_zscore_longer_seed4",
    "exp_stage1_r101_dice_zscore_longer_seed5",
    "exp_stage1_mobilenetv2_dice_zscore_longer_seed2",
    "exp_stage1_mobilenetv2_dice_zscore_longer_seed3",
    "exp_stage1_r50_dice_zscore_longer_seed4",
    "exp_stage1_r50_dice_zscore_longer_seed5",
    "exp_stage1_r101_dice_zscore_longer_seed6",
    "exp_stage1_r50_dice_zscore_longer_seed6",
    "exp_stage1_r50_dice_zscore_longer_seed7",
    "exp_stage1_mobilenetv2_dice_zscore_longer_seed4",
]


def get_queued_job_names() -> set[str]:
    result = subprocess.run(
        ["squeue", "--user", os.environ.get("USER", "ruisong"),
         "-h", "-t", "PENDING,RUNNING", "--format=%.200j"],
        capture_output=True, text=True,
    )
    return {l.strip() for l in result.stdout.splitlines() if l.strip()}


def get_queue_depth() -> int:
    result = subprocess.run(
        ["squeue", "--user", os.environ.get("USER", "ruisong"),
         "-h", "-t", "PENDING,RUNNING", "-o", "%i"],
        capture_output=True, text=True,
    )
    return len([l for l in result.stdout.splitlines() if l.strip()])


def exp_done(exp_id: str) -> bool:
    return (EXPERIMENTS_DIR / exp_id / "metrics.json").exists()


def exp_has_checkpoint(exp_id: str) -> bool:
    return (EXPERIMENTS_DIR / exp_id / "best.weights.h5").exists()


def config_exists(exp_id: str) -> bool:
    return (CONFIGS_DIR / f"{exp_id}.yaml").exists()


def read_result(exp_id: str) -> dict | None:
    path = EXPERIMENTS_DIR / exp_id / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def submit_job(exp_id: str, dry_run: bool = False) -> bool:
    cfg = f"configs/training/experiments/{exp_id}.yaml"
    if not (REPO / cfg).exists():
        LOGGER.warning("SKIP %s — config not found: %s", exp_id, cfg)
        return False
    if exp_done(exp_id):
        LOGGER.info("SKIP %s — already completed.", exp_id)
        return False
    if exp_has_checkpoint(exp_id):
        LOGGER.info("SKIP %s — checkpoint exists (may be running).", exp_id)
        return False
    queued = get_queued_job_names()
    if exp_id in queued:
        LOGGER.info("SKIP %s — already in queue.", exp_id)
        return False

    env = {**os.environ, "EXP_ID": exp_id, "CONFIG_PATH": cfg, "EXTRA_ARGS": "-v"}
    cmd = ["sbatch", f"--job-name={exp_id}", str(SLURM_SCRIPT)]

    if dry_run:
        LOGGER.info("DRY-RUN: would submit %s", exp_id)
        return True

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        LOGGER.info("SUBMITTED %s => job %s", exp_id, job_id)
        return True
    LOGGER.error("FAILED %s: %s", exp_id, result.stderr.strip())
    return False


def print_results_table() -> None:
    baseline = 0.740
    rows = []
    for exp_id in EXPERIMENT_QUEUE:
        m = read_result(exp_id)
        if m:
            t = m.get("test_metrics", {})
            dice = t.get("dice_coefficient", float("nan"))
            iou = t.get("iou_coefficient", float("nan"))
            val = m.get("best_val_dice", float("nan"))
            ep = m.get("best_epoch", "?")
            delta = dice - baseline
            marker = " ✓" if delta > 0.010 else (" ✗" if delta < -0.020 else "")
            rows.append((exp_id, val, ep, dice, iou, delta, marker))

    if not rows:
        LOGGER.info("No completed experiments yet.")
        return

    LOGGER.info("=" * 90)
    LOGGER.info("%-48s  %6s %4s  %6s  %6s  %7s", "experiment", "valDce", "ep", "tstDce", "tstIoU", "Δbaseln")
    LOGGER.info("-" * 90)
    for exp_id, val, ep, dice, iou, delta, marker in sorted(rows, key=lambda x: -x[3]):
        LOGGER.info("%-48s  %.4f %4s  %.4f  %.4f  %+.4f%s",
                    exp_id, val, ep, dice, iou, delta, marker)
    LOGGER.info("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=8)
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline manager starting. max_jobs=%d dry_run=%s", args.max_jobs, args.dry_run)

    print_results_table()

    depth = get_queue_depth()
    slots = args.max_jobs - depth
    LOGGER.info("Queue depth: %d  |  Available slots: %d", depth, slots)

    if slots <= 0:
        LOGGER.info("Queue full — nothing to submit.")
        return

    submitted = 0
    for exp_id in EXPERIMENT_QUEUE:
        if submitted >= slots:
            break
        if not config_exists(exp_id):
            LOGGER.debug("SKIP %s — no config yet.", exp_id)
            continue
        if exp_done(exp_id):
            continue
        if submit_job(exp_id, dry_run=args.dry_run):
            submitted += 1

    LOGGER.info("Pipeline done. Submitted %d new jobs.", submitted)


if __name__ == "__main__":
    main()
