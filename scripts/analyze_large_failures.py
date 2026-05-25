#!/usr/bin/env python
"""Analyze the audit_30k_h5.csv results focused on large positives only.

Large = panel_frac >= 0.40 (the deployment-relevant utility-scale PV sites
that Sentinel-2 is well-suited to detect).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
CSV  = REPO / "docs/audit_30k_h5.csv"


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    large = [r for r in rows if r["label"] == "1" and float(r["panel_frac"]) >= 0.40]
    print(f"Total large positives (panel_frac >= 0.40): {len(large):,}\n")

    dices = [float(r["dice"]) for r in large]
    recs  = [float(r["recall"])     for r in large if r["recall"]     != ""]
    precs = [float(r["precision"])  for r in large if r["precision"]  != ""]

    healthy  = [r for r in large if float(r["dice"]) >= 0.85]
    middling = [r for r in large if 0.50 <= float(r["dice"]) < 0.85]
    failing  = [r for r in large if float(r["dice"]) < 0.50]

    print(f"=== HEALTH BUCKETS (large positives, n={len(large)}) ===")
    print(f"  healthy  (>=0.85): {len(healthy):>5d}  ({100*len(healthy)/len(large):.1f}%)")
    print(f"  middling (0.5-0.85): {len(middling):>5d}  ({100*len(middling)/len(large):.1f}%)")
    print(f"  failing  (< 0.5)  : {len(failing):>5d}  ({100*len(failing)/len(large):.1f}%)")
    print(f"  median dice/recall/precision: {np.median(dices):.4f} / {np.median(recs):.4f} / {np.median(precs):.4f}")
    print(f"  Dice q25 / q10 / q05: {np.quantile(dices,.25):.4f} / {np.quantile(dices,.10):.4f} / {np.quantile(dices,.05):.4f}")
    print()

    print("=== LARGE BY CONTINENT ===")
    print(f"  {'continent':<12s} {'n':>4s}  {'median':>7s}  {'q25':>7s}  {'%<0.85':>6s}  {'%<0.50':>6s}")
    by_cont: dict[str, list[float]] = {}
    for r in large:
        by_cont.setdefault(r["continent"], []).append(float(r["dice"]))
    for c, ds in sorted(by_cont.items(), key=lambda kv: -len(kv[1])):
        pct_85 = 100*sum(1 for d in ds if d < 0.85)/len(ds)
        pct_50 = 100*sum(1 for d in ds if d < 0.50)/len(ds)
        print(f"  {c:<12s} {len(ds):>4d}  {np.median(ds):>7.4f}  {np.quantile(ds,.25):>7.4f}  {pct_85:>5.1f}%  {pct_50:>5.1f}%")
    print()

    print("=== LARGE BY panel_frac SUB-BUCKET ===")
    for lo, hi in [(0.40, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.001)]:
        sub = [r for r in large if lo <= float(r["panel_frac"]) < hi]
        if not sub: continue
        ds = [float(r["dice"]) for r in sub]
        pct_85 = 100*sum(1 for d in ds if d<0.85)/len(ds)
        pct_50 = 100*sum(1 for d in ds if d<0.50)/len(ds)
        print(f"  pf [{lo:.2f},{hi:.2f})  n={len(sub):>4d}  median={np.median(ds):.4f}  q25={np.quantile(ds,.25):.4f}  %<0.85={pct_85:.1f}%  %<0.50={pct_50:.1f}%")
    print()

    print("=== TOP 25 WORST LARGE CHIPS ===")
    print(f"  {'chip_id':<22s} {'cont':<10s} {'lat':>7s} {'lon':>8s} {'pf':>5s} {'dice':>6s} {'P':>5s} {'R':>5s}")
    for r in sorted(large, key=lambda x: float(x["dice"]))[:25]:
        lat = float(r["lat"]) if r["lat"] else 0.0
        lon = float(r["lon"]) if r["lon"] else 0.0
        print(f"  {r['chip_id']:<22s} {r['continent']:<10s} {lat:>+7.2f} {lon:>+8.2f} "
              f"{float(r['panel_frac']):>5.2f} {r['dice']:>6s} {r['precision']:>5s} {r['recall']:>5s}")
    print()

    print("=== LARGE FAILING (<0.85) BY LATITUDE BAND ===")
    for lo, hi in [(-90,-30),(-30,-10),(-10,0),(0,10),(10,23),(23,35),(35,45),(45,55),(55,90)]:
        ds = [float(r["dice"]) for r in large if r["lat"] != "" and lo <= float(r["lat"]) < hi]
        nfail = sum(1 for d in ds if d < 0.50)
        nmid  = sum(1 for d in ds if 0.50 <= d < 0.85)
        if not ds: continue
        print(f"  lat [{lo:>+3d},{hi:>+3d}) total={len(ds):>4d}  failing={nfail:>3d}  middling={nmid:>3d}  "
              f"({100*(nfail+nmid)/len(ds):.0f}% problematic)")

    # Save the failing+middling chips to its own CSV for downstream analysis
    bad = sorted([r for r in large if float(r["dice"]) < 0.85], key=lambda x: float(x["dice"]))
    bad_path = REPO / "docs/audit_30k_large_problem.csv"
    if bad:
        with bad_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bad[0].keys())); w.writeheader()
            for r in bad: w.writerow(r)
        print(f"\nWrote {len(bad)} problem-large chips -> {bad_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
