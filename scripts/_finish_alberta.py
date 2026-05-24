#!/usr/bin/env python
"""Process the Alberta site stack that's already on disk (one-off recovery)."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
spec = importlib.util.spec_from_file_location(
    "rw", REPO / "scripts/19_test_real_world_sites.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cid = "c-004902_r+002536"
mm = {}
with open(REPO / "outputs/final/master_manifest.csv") as f:
    for r in csv.DictReader(f):
        mm[r["chip_id_str"]] = r

r = mm[cid]
site = {
    "chip_id": cid, "tier": "large", "half_size_km": 5.0,
    "continent": r["continent"],
    "lat": float(r["chip_center_lat"]), "lon": float(r["chip_center_lon"]),
    "panel_frac": float(r["panel_frac"]),
    "col": int(r["chip_col"]), "row": int(r["chip_row"]),
}
out_dir = REPO / "docs/realworld_test_2026Q1"
nc = out_dir / f"{cid}_stack.nc"
mod.process_site(site, nc, out_dir)
