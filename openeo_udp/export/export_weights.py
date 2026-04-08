#!/usr/bin/env python
"""Package model weights and band statistics for release.

Usage:
    python openeo_udp/export/export_weights.py --version v1.0

This script copies the best model checkpoint and band statistics into a
release-ready directory (openeo_udp/export/releases/<version>/) and prints
the commands needed to create and upload a GitHub Release.

After running this script, follow the printed instructions to upload the
artifacts to GitHub Releases, then update model_registry.yaml with the
new URLs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Default source paths (relative to repo root)
DEFAULT_WEIGHTS = "experiments/exp_stage1_r101_dice_zscore_longer/best.weights.h5"
DEFAULT_BAND_STATS = "outputs/training_prep/stage1_v3/band_stats.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True,
                        help="Release version tag, e.g. v1.0")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to model weights (relative to repo root)")
    parser.add_argument("--band-stats", default=DEFAULT_BAND_STATS,
                        help="Path to band_stats.npz (relative to repo root)")
    parser.add_argument("--config",
                        default=None,
                        help="Path to experiment config.yaml to include (relative to repo root)")
    args = parser.parse_args()

    weights_src = REPO / args.weights
    stats_src = REPO / args.band_stats

    if not weights_src.exists():
        raise FileNotFoundError(f"Weights not found: {weights_src}")
    if not stats_src.exists():
        raise FileNotFoundError(f"Band stats not found: {stats_src}")

    release_dir = Path(__file__).resolve().parent / "releases" / args.version
    release_dir.mkdir(parents=True, exist_ok=True)

    weights_dst = release_dir / "best.weights.h5"
    stats_dst = release_dir / "band_stats.npz"

    print(f"Copying weights:    {weights_src}")
    print(f"             ->     {weights_dst}")
    shutil.copy2(weights_src, weights_dst)

    print(f"Copying band stats: {stats_src}")
    print(f"             ->     {stats_dst}")
    shutil.copy2(stats_src, stats_dst)

    if args.config:
        config_src = REPO / args.config
        if config_src.exists():
            config_dst = release_dir / "config.yaml"
            shutil.copy2(config_src, config_dst)
            print(f"Copying config:     {config_src}")
            print(f"             ->     {config_dst}")

    print(f"\nRelease artifacts ready in: {release_dir}")
    print(f"  - {weights_dst.name}  ({weights_dst.stat().st_size / 1e6:.0f} MB)")
    print(f"  - {stats_dst.name}  ({stats_dst.stat().st_size / 1e3:.0f} KB)")

    # Detect GitHub remote for instructions
    repo_url = "<your-github-repo-url>"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(REPO), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        )
        repo_url = result.stdout.strip().replace(".git", "")
    except Exception:
        pass

    print(f"""
--- Next steps ---

1. Create a GitHub Release (requires `gh` CLI):

   cd {REPO}
   gh release create {args.version} \\
       {weights_dst} \\
       {stats_dst} \\
       --title "Solar PV Detection Model {args.version}" \\
       --notes "Model: ResNet101 U-Net, Dice loss, z-score normalization.
   Trained on 4,922 Sentinel-2 L1C chips (256x256, 13 bands).
   Best test Dice: 0.8853 at threshold 0.80."

2. Update openeo_udp/model_registry.yaml with the download URLs:

   weights_url: {repo_url}/releases/download/{args.version}/best.weights.h5
   band_stats_url: {repo_url}/releases/download/{args.version}/band_stats.npz
   version: "{args.version}"
""")


if __name__ == "__main__":
    main()
