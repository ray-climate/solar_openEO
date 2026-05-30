#!/usr/bin/env python
"""Extract per-pixel NDVI/NDBI percentile features (idea #1)."""
from __future__ import annotations

import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from solar_fp_filter import timeseries_gee as tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=40)
    a = ap.parse_args()
    tg.extract_percentiles_gee(chunk_size=a.chunk_size, limit=a.limit)


if __name__ == "__main__":
    main()
