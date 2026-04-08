#!/usr/bin/env python
"""Visualize random samples from the packaged Stage-1 HDF5 dataset.

Creates a figure with one column per random sample:
  top row    -> RGB quicklook from B4/B3/B2
  bottom row -> RGB with mask overlay

Usage:
  python scripts/08_visualize_h5_samples.py
  python scripts/08_visualize_h5_samples.py --h5 outputs/stage1/stage1_positives.h5 --n 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="outputs/stage1/stage1_positives.h5")
    parser.add_argument("--n", type=int, default=6, help="Number of random samples to plot.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="outputs/stage1/figures/h5_random_samples.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((band.astype(np.float32) - p_lo) / (p_hi - p_lo), 0, 1)


def make_rgb(image: np.ndarray) -> np.ndarray:
    r = percentile_stretch(image[3])  # B4
    g = percentile_stretch(image[2])  # B3
    b = percentile_stretch(image[1])  # B2
    return np.stack([r, g, b], axis=-1)


def decode_chip_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        return value.decode()
    return str(value)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5)
    out_path = Path(args.out)

    if not h5_path.exists():
        print(f"HDF5 not found: {h5_path}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        images = f["images"]
        masks = f["masks"]
        chip_ids = f["chip_ids"]
        n_total = len(chip_ids)
        if n_total == 0:
            print(f"No samples in HDF5: {h5_path}")
            return 1

        rng = np.random.default_rng(args.seed)
        n_plot = min(args.n, n_total)
        sample_idx = np.sort(rng.choice(n_total, size=n_plot, replace=False))

        fig, axes = plt.subplots(2, n_plot, figsize=(n_plot * 3.0, 6.0), squeeze=False)
        mask_color = np.array([1.0, 0.15, 0.15], dtype=np.float32)

        for col, idx in enumerate(sample_idx):
            image = images[idx].astype(np.float32)
            mask = masks[idx].astype(np.uint8)
            chip_id = decode_chip_id(chip_ids[idx])
            rgb = make_rgb(image)
            overlay = rgb.copy()

            if mask.any():
                panel_px = mask == 1
                overlay[panel_px] = 0.55 * overlay[panel_px] + 0.45 * mask_color

            axes[0][col].imshow(rgb)
            axes[0][col].set_title(chip_id, fontsize=8)
            axes[0][col].axis("off")

            axes[1][col].imshow(overlay)
            axes[1][col].set_title(f"mask px: {int(mask.sum())}", fontsize=8)
            axes[1][col].axis("off")

        fig.suptitle(
            f"Random HDF5 samples from {h5_path.name} ({n_plot} of {n_total})",
            fontsize=12,
            y=0.98,
        )
        plt.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figure -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
