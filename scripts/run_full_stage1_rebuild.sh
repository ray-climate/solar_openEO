#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/stage1/logs outputs/stage1/figures

echo "[stage1] sync mosaics"
python scripts/sync_stage1_v3_mosaics.py

echo "[stage1] extract chips"
python scripts/04_extract_chips.py

echo "[stage1] package hdf5"
python scripts/05_package_dataset.py

echo "[stage1] render hdf5 qa"
python scripts/08_visualize_h5_samples.py \
  --h5 outputs/stage1/stage1_positives.h5 \
  --n 6 \
  --out outputs/stage1/figures/h5_random_samples.png

echo "[stage1] complete"
