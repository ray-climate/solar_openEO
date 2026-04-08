#!/usr/bin/env python
"""Prepare training splits and band stats from the Stage-1 HDF5 dataset.

Usage:
  python scripts/prepare_stage1_training_data.py
  python scripts/prepare_stage1_training_data.py --h5 outputs/stage1/stage1_positives.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_ml.data import compute_h5_band_stats, create_split_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        default="outputs/stage1/stage1_positives.h5",
        help="Path to Stage-1 HDF5 dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/training_prep/stage1_v3",
        help="Directory for split manifest and band stats.",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--block-size-chips", type=int, default=32)
    parser.add_argument("--max-stat-chips", type=int, default=512)
    parser.add_argument("--samples-per-band-per-chip", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5)
    if not h5_path.exists():
        LOGGER.error("Stage-1 dataset not found: %s", h5_path)
        LOGGER.error(
            "Build it first via scripts/04_extract_chips.py and scripts/05_package_dataset.py."
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "split_manifest.csv"
    split_summary_json = out_dir / "split_summary.json"
    stats_npz = out_dir / "band_stats.npz"
    stats_summary_json = out_dir / "band_stats_summary.json"

    LOGGER.info("Creating split manifest: %s", manifest_csv)
    create_split_manifest(
        h5_path=h5_path,
        out_csv=manifest_csv,
        out_summary_json=split_summary_json,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_strategy="spatial_block",
        block_size_chips=args.block_size_chips,
    )

    LOGGER.info("Computing band stats: %s", stats_npz)
    compute_h5_band_stats(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        out_npz=stats_npz,
        out_summary_json=stats_summary_json,
        max_stat_chips=args.max_stat_chips,
        samples_per_band_per_chip=args.samples_per_band_per_chip,
    )

    print(f"Split manifest written  -> {manifest_csv}")
    print(f"Band stats written      -> {stats_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
