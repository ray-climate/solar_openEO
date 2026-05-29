#!/usr/bin/env python
"""Launcher for Phase 2 extraction via Google Earth Engine.

    python scripts/run_fp_phase2_extract_gee.py --limit 500   # pilot
    python scripts/run_fp_phase2_extract_gee.py               # full run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import timeseries_gee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N samples (pilot mode)")
    ap.add_argument("--chunk-size", type=int, default=150,
                    help="Bounded by GEE 10MB request payload (inline geometries)")
    args = ap.parse_args()
    timeseries_gee.extract_for_polygons_gee(
        chunk_size=args.chunk_size, limit=args.limit)


if __name__ == "__main__":
    main()
