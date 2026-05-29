#!/usr/bin/env python
"""Launcher for Phase 3 — feature engineering + LightGBM training."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import train


if __name__ == "__main__":
    train.train()
