#!/usr/bin/env python
"""Launcher for Phase 1 of the FP-classifier pipeline.

Builds the training polygon set (positives + negatives) and writes
``outputs/fp_classifier/polygons.gpkg`` plus a distribution PNG.

Usage:
    python scripts/run_fp_phase1_build_polygons.py
    python scripts/run_fp_phase1_build_polygons.py --n-positives 1000  # quick test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positives",  type=int, default=10_000)
    ap.add_argument("--n-land",       type=int, default=5_000)
    ap.add_argument("--n-water",      type=int, default=2_000)
    ap.add_argument("--n-industrial", type=int, default=3_000)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--skip-plot",    action="store_true")
    args = ap.parse_args()

    polygons.build_polygon_set(
        n_positives=args.n_positives,
        n_land=args.n_land,
        n_water=args.n_water,
        n_industrial=args.n_industrial,
        seed=args.seed,
    )
    if not args.skip_plot:
        polygons.plot_distribution()


if __name__ == "__main__":
    main()
