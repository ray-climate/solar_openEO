#!/usr/bin/env python
"""
Wave-2 adaptive monitor: reads Wave-1 results and submits follow-up experiments.

Run on a cron from JASMIN (every 90 min). Safe to re-run — skips already-submitted
experiments and won't submit if >=6 jobs are already running.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_FILE = REPO / "slurm_logs" / "auto_wave2_monitor.log"
EXPERIMENTS_DIR = REPO / "experiments"
CONFIGS_DIR = REPO / "configs" / "training" / "experiments"
SLURM_SCRIPT = REPO / "scripts" / "slurm" / "train_stage1_gpu.sh"

# Baseline to beat (exp_stage1_resnet101_tfdata best threshold-tuned run)
BASELINE_TEST_DICE = 0.740
MIN_IMPROVEMENT = 0.010  # must beat baseline by at least this to be "a winner"
MAX_RUNNING_JOBS = 6     # don't submit if queue is already this full

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
LOGGER = logging.getLogger(__name__)


def get_running_job_count() -> int:
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "ruisong"), "-h", "-t", "PENDING,RUNNING", "-o", "%i"],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return len(lines)


def read_metrics(exp_id: str) -> dict | None:
    path = EXPERIMENTS_DIR / exp_id / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def exp_exists(exp_id: str) -> bool:
    """True if the experiment already has a checkpoint or metrics file."""
    exp_dir = EXPERIMENTS_DIR / exp_id
    return (exp_dir / "metrics.json").exists() or (exp_dir / "best.weights.h5").exists()


def submit_job(exp_id: str, config_path: str) -> str | None:
    if exp_exists(exp_id):
        LOGGER.info("SKIP %s — already exists.", exp_id)
        return None
    env = {**os.environ, "EXP_ID": exp_id, "CONFIG_PATH": config_path, "EXTRA_ARGS": "-v"}
    result = subprocess.run(
        ["sbatch", f"--job-name={exp_id}", str(SLURM_SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        LOGGER.info("SUBMITTED %s => job %s", exp_id, job_id)
        return job_id
    LOGGER.error("FAILED to submit %s: %s", exp_id, result.stderr.strip())
    return None


def create_config(exp_id: str, inherits: str, overrides: dict) -> str:
    """Write a new YAML config and return its path."""
    import yaml
    cfg_path = CONFIGS_DIR / f"{exp_id}.yaml"
    if cfg_path.exists():
        return str(cfg_path.relative_to(REPO))
    content = {"inherits": f"{inherits}.yaml", "experiment": {"name": exp_id}}
    content.update(overrides)
    cfg_path.write_text(yaml.safe_dump(content, sort_keys=False))
    LOGGER.info("Created config: %s", cfg_path.name)
    return str(cfg_path.relative_to(REPO))


def main() -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Wave-2 monitor starting.")

    # --- Read Wave-1 results ---
    wave1_ids = [
        "exp_stage1_r101_bce_dice_fp32",
        "exp_stage1_r101_spectral_aug",
        "exp_stage1_r101_6band",
        "exp_stage1_r101_10band",
        "exp_stage1_r101_wide_decoder",
        "exp_stage1_r101_longer",
        "exp_stage1_r101_dice_only",
        "exp_stage1_r101_spectral_6band",
    ]

    completed: dict[str, dict] = {}
    pending: list[str] = []
    for exp_id in wave1_ids:
        m = read_metrics(exp_id)
        if m is not None:
            completed[exp_id] = m
        else:
            pending.append(exp_id)

    LOGGER.info("Wave-1 completed: %d / %d", len(completed), len(wave1_ids))
    if pending:
        LOGGER.info("Still pending: %s", pending)

    if not completed:
        LOGGER.info("Nothing completed yet — nothing to do.")
        return

    # --- Score each completed experiment ---
    scores: dict[str, float] = {}
    for exp_id, m in completed.items():
        dice = m.get("test_metrics", {}).get("dice_coefficient")
        if dice is None:
            # fall back to best_val_dice
            dice = m.get("best_val_dice")
        if dice is not None:
            scores[exp_id] = float(dice)
            delta = scores[exp_id] - BASELINE_TEST_DICE
            LOGGER.info("  %-45s  test_dice=%.4f  Δ=%.4f", exp_id, scores[exp_id], delta)

    winners = {k: v for k, v in scores.items() if v > BASELINE_TEST_DICE + MIN_IMPROVEMENT}
    LOGGER.info("Winners (beat baseline by >%.3f): %s", MIN_IMPROVEMENT, list(winners.keys()))

    if not winners:
        LOGGER.info("No winner beats the baseline yet — waiting for more results.")
        return

    # --- Check queue capacity ---
    running = get_running_job_count()
    LOGGER.info("Current queue depth: %d jobs", running)
    if running >= MAX_RUNNING_JOBS:
        LOGGER.info("Queue full (%d >= %d) — not submitting yet.", running, MAX_RUNNING_JOBS)
        return

    # --- Wave-2 decision logic ---
    # Rank winners by score descending
    ranked = sorted(winners.items(), key=lambda x: x[1], reverse=True)
    LOGGER.info("Ranked winners: %s", [(k, f"{v:.4f}") for k, v in ranked])

    best_id, best_dice = ranked[0]
    to_submit: list[tuple[str, str]] = []  # (exp_id, config_path)

    spectral_won = any("spectral_aug" in k and "6band" not in k for k in winners)
    sixband_won = any("6band" in k and "spectral" not in k for k in winners)
    spectral_6band_won = "exp_stage1_r101_spectral_6band" in winners
    bce_fp32_won = "exp_stage1_r101_bce_dice_fp32" in winners
    wide_decoder_won = "exp_stage1_r101_wide_decoder" in winners
    longer_won = "exp_stage1_r101_longer" in winners
    dice_only_won = "exp_stage1_r101_dice_only" in winners
    tenband_won = "exp_stage1_r101_10band" in winners

    # Combination follow-ups
    if spectral_won and bce_fp32_won:
        eid = "exp_stage1_r101_spectral_bce_fp32"
        cfg = create_config(eid, "exp_stage1_r101_bce_dice_fp32",
                            {"data": {"augment_spectral": True}})
        to_submit.append((eid, cfg))

    if sixband_won and spectral_won and not spectral_6band_won:
        # Already have spectral_6band in Wave1 but it might not have completed yet
        eid = "exp_stage1_r101_spectral_6band"
        cfg = str((CONFIGS_DIR / f"{eid}.yaml").relative_to(REPO))
        to_submit.append((eid, cfg))

    if sixband_won and bce_fp32_won:
        eid = "exp_stage1_r101_6band_bce_fp32"
        cfg = create_config(eid, "exp_stage1_r101_bce_dice_fp32",
                            {"data": {"band_indices": [1, 2, 3, 7, 11, 12]}})
        to_submit.append((eid, cfg))

    if spectral_6band_won and bce_fp32_won:
        eid = "exp_stage1_r101_spectral_6band_bce_fp32"
        cfg = create_config(eid, "exp_stage1_r101_bce_dice_fp32",
                            {"data": {"band_indices": [1, 2, 3, 7, 11, 12],
                                      "augment_spectral": True}})
        to_submit.append((eid, cfg))

    if wide_decoder_won and spectral_won:
        eid = "exp_stage1_r101_wide_decoder_spectral"
        cfg = create_config(eid, "exp_stage1_r101_wide_decoder",
                            {"data": {"augment_spectral": True}})
        to_submit.append((eid, cfg))

    if tenband_won and spectral_won:
        eid = "exp_stage1_r101_10band_spectral"
        cfg = create_config(eid, "exp_stage1_r101_10band",
                            {"data": {"augment_spectral": True}})
        to_submit.append((eid, cfg))

    # If spectral_6band is already a winner, try it with longer training
    if spectral_6band_won and not longer_won:
        eid = "exp_stage1_r101_spectral_6band_longer"
        cfg = create_config(eid, "exp_stage1_r101_spectral_6band",
                            {"training": {"epochs": 100,
                                          "early_stopping_patience": 20}})
        to_submit.append((eid, cfg))

    # Best single winner + longer if "longer" itself didn't win
    if best_id not in ("exp_stage1_r101_longer",) and not longer_won:
        eid = f"{best_id}_longer"
        cfg = create_config(eid, best_id,
                            {"training": {"epochs": 100,
                                          "early_stopping_patience": 20}})
        to_submit.append((eid, cfg))

    if not to_submit:
        LOGGER.info("No new combinations to submit yet.")
        return

    # --- Submit (respecting queue limit) ---
    slots = MAX_RUNNING_JOBS - running
    LOGGER.info("Submitting up to %d new jobs.", slots)
    submitted = 0
    for exp_id, cfg_path in to_submit:
        if submitted >= slots:
            LOGGER.info("Queue limit reached — remaining deferred to next run.")
            break
        job_id = submit_job(exp_id, cfg_path)
        if job_id:
            submitted += 1

    LOGGER.info("Wave-2 monitor done. Submitted %d jobs.", submitted)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
