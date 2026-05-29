#!/usr/bin/env python
"""Launcher for Phase 5 — EU-unseen FP-filter evaluation (4-panel PNGs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import eval_eu_unseen as E
from solar_fp_filter import inference as I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=I.KEEP_THRESHOLD)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    E.evaluate(threshold=args.threshold, push=not args.no_push)


if __name__ == "__main__":
    main()
