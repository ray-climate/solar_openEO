#!/usr/bin/env python
"""Compare the pre-fine-tune (R3 R101 dropout02) audit against the post-fine-tune
(R3 R101 reviewed) audit. Focus on:
  - Bucket counts: healthy / middling / failing
  - Median Dice overall, by split, by continent, by tier
  - The 119 'keep' hard-mine chips' Dice before vs after
  - The 1128 'reject' chips' Dice before vs after (sanity: should not degrade)
  - The 1156 healthy-large chips (Dice >= 0.85 pre): retention rate
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter

import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
PRE  = REPO / "docs/audit_30k_h5.csv"
POST = REPO / "docs/audit_30k_h5_reviewed.csv"
DECISIONS_CSV = REPO / "docs/review_large/decisions_2026-05-25.csv"


def load_audit(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            rows[r["chip_id"]] = r
    return rows


def diffs(label: str, ids: set[str], pre: dict, post: dict, only_pos: bool = True) -> None:
    pre_ds, post_ds = [], []
    for cid in ids:
        if cid not in pre or cid not in post:
            continue
        if only_pos and (pre[cid]["label"] != "1"):
            continue
        pre_ds.append(float(pre[cid]["dice"]))
        post_ds.append(float(post[cid]["dice"]))
    if not pre_ds:
        print(f"  {label}: no data"); return
    pre_arr  = np.array(pre_ds)
    post_arr = np.array(post_ds)
    delta = post_arr - pre_arr
    n = len(pre_arr)
    n_improved = int((delta > 0.01).sum())
    n_worse    = int((delta < -0.01).sum())
    n_same     = n - n_improved - n_worse
    print(f"  {label} (n={n}):")
    print(f"    pre  median {np.median(pre_arr):.4f}  q25 {np.quantile(pre_arr,.25):.4f}  q75 {np.quantile(pre_arr,.75):.4f}")
    print(f"    post median {np.median(post_arr):.4f}  q25 {np.quantile(post_arr,.25):.4f}  q75 {np.quantile(post_arr,.75):.4f}")
    print(f"    mean delta {delta.mean():+.4f}  median delta {np.median(delta):+.4f}")
    print(f"    improved (Δ>+0.01): {n_improved}  unchanged: {n_same}  worse (Δ<-0.01): {n_worse}")


def main() -> None:
    pre  = load_audit(PRE)
    post = load_audit(POST)
    print(f"PRE  audit: {len(pre):,} chips  ({PRE.name})")
    print(f"POST audit: {len(post):,} chips  ({POST.name})")
    assert set(pre) == set(post), "chip_id sets differ between audits — can't diff"

    # Buckets — positives only
    def buckets(audit):
        pos = [r for r in audit.values() if r["label"] == "1"]
        h = sum(1 for r in pos if float(r["dice"]) >= 0.85)
        m = sum(1 for r in pos if 0.50 <= float(r["dice"]) < 0.85)
        f = sum(1 for r in pos if float(r["dice"]) < 0.50)
        return len(pos), h, m, f, np.median([float(r["dice"]) for r in pos])

    n_pre, h_pre, m_pre, f_pre, med_pre = buckets(pre)
    n_post, h_post, m_post, f_post, med_post = buckets(post)
    print(f"\n=== POSITIVES — BUCKETS (Dice on training-time imagery) ===")
    print(f"  {'bucket':<22s} {'pre':>8s}  {'post':>8s}  {'Δ':>7s}")
    print(f"  {'healthy (>=0.85)':<22s} {h_pre:>8d}  {h_post:>8d}  {h_post-h_pre:>+7d}")
    print(f"  {'middling (0.5-0.85)':<22s} {m_pre:>8d}  {m_post:>8d}  {m_post-m_pre:>+7d}")
    print(f"  {'failing (<0.5)':<22s} {f_pre:>8d}  {f_post:>8d}  {f_post-f_pre:>+7d}")
    print(f"  {'median Dice':<22s} {med_pre:>8.4f}  {med_post:>8.4f}  {med_post-med_pre:>+7.4f}")

    # By tier (positives)
    print(f"\n=== POSITIVES BY TIER ===")
    for tier in ("small", "medium", "large"):
        ids = {cid for cid, r in pre.items() if r["label"] == "1" and r["tier"] == tier}
        diffs(f"tier={tier}", ids, pre, post)

    # By split
    print(f"\n=== POSITIVES BY V2 SPLIT ===")
    for sp in ("train", "val", "test"):
        ids = {cid for cid, r in pre.items() if r["label"] == "1" and r["split"] == sp}
        diffs(f"split={sp}", ids, pre, post)

    # Large-only, by continent
    print(f"\n=== LARGE POSITIVES BY CONTINENT ===")
    by_cont: dict[str, set] = {}
    for cid, r in pre.items():
        if r["label"] == "1" and r["tier"] == "large":
            by_cont.setdefault(r["continent"], set()).add(cid)
    for c, ids in sorted(by_cont.items(), key=lambda kv: -len(kv[1])):
        diffs(f"continent={c}", ids, pre, post)

    # Review-decision subgroups
    print(f"\n=== BY USER REVIEW DECISION (large chips only) ===")
    dec_groups: dict[str, set[str]] = {"keep": set(), "reject": set(), "negative": set()}
    with DECISIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            d = r["decision"]; cid = r["chip_id"]
            if d in dec_groups:
                dec_groups[d].add(cid)
    for d, ids in dec_groups.items():
        diffs(f"decision={d}", ids, pre, post)

    # Negative chips (false-positive rate)
    print(f"\n=== NEGATIVE CHIPS (false-positive rate) ===")
    neg_ids = [cid for cid, r in pre.items() if r["label"] == "0"]
    pre_fp = sum(1 for cid in neg_ids if int(pre[cid]["pred_px"]) > 0)
    post_fp = sum(1 for cid in neg_ids if int(post[cid]["pred_px"]) > 0)
    print(f"  negatives with any false positive:  pre={pre_fp}  post={post_fp}  Δ={post_fp-pre_fp:+d}")


if __name__ == "__main__":
    main()
