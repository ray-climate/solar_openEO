#!/usr/bin/env python
"""Launcher for Phase 2 — extract aggregate_spatial time series.

By default runs the smoke test (60 samples). Pass --full for the
26K-sample extraction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import timeseries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Run on all 26K samples (default: smoke test)")
    ap.add_argument("--smoke-n-per-class", type=int, default=15)
    ap.add_argument("--max-concurrent-jobs", type=int, default=4)
    args = ap.parse_args()
    timeseries.extract_for_polygons(
        smoke_test=not args.full,
        smoke_n_per_class=args.smoke_n_per_class,
        max_concurrent_jobs=args.max_concurrent_jobs,
    )


if __name__ == "__main__":
    main()
