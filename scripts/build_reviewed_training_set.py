#!/usr/bin/env python
"""Turn the user's review CSV into config artifacts for fine-tuning.

Inputs:
  - docs/review_large/decisions_2026-05-25.csv  (chip_id, decision, timestamp)
  - outputs/training_prep/final_v2/split_manifest.csv  (existing train/val/test split)
  - docs/audit_30k_h5.csv  (per-chip Dice for sanity reporting)

Interpretation of decisions (user-corrected semantics):
  keep     -> POSITIVE, oversample (hard-mining) during fine-tuning
  negative -> mask zeroed at training time, treat as HARD NEGATIVE example
  reject   -> NO ACTION — chip stays in training unchanged with current mask
  skip     -> NO ACTION (same as reject)

Outputs (under outputs/training_prep/final_v2_reviewed/):
  decisions_summary.json   - counts + per-bucket median Dice
  keep_oversample.csv      - chip_ids to oversample during training
  negative_override.csv    - chip_ids whose mask should be zeroed during training
  rejected_log.csv         - chip_ids the user marked 'reject' (for record only;
                             these are NOT removed from training)

The fine-tune script (Phase 6) reads these config files and applies them at
batch-construction time. The base split_manifest.csv is NOT modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
DEFAULT_DECISIONS = REPO / "docs/review_large/decisions_2026-05-25.csv"
SPLIT_MANIFEST    = REPO / "outputs/training_prep/final_v2/split_manifest.csv"
AUDIT_CSV         = REPO / "docs/audit_30k_h5.csv"
OUT_DIR           = REPO / "outputs/training_prep/final_v2_reviewed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=str(DEFAULT_DECISIONS),
                    help="Path to the exported decisions CSV")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    decisions_path = Path(args.decisions)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading decisions: {decisions_path}")
    print(f"Writing to:        {out_dir.relative_to(REPO)}/\n")

    # Load decisions
    by_decision: dict[str, list[str]] = {"keep": [], "reject": [], "negative": [], "skip": []}
    with decisions_path.open() as f:
        for r in csv.DictReader(f):
            d = r.get("decision", "").strip()
            cid = r.get("chip_id", "").strip()
            if not cid or not d:
                continue
            if d not in by_decision:
                print(f"  WARN unknown decision '{d}' for {cid}, treating as skip")
                d = "skip"
            by_decision[d].append(cid)

    print("Decision counts:")
    for k in ("keep", "reject", "negative", "skip"):
        print(f"  {k:<8s} {len(by_decision[k]):>5d}")
    total = sum(len(v) for v in by_decision.values())
    print(f"  total    {total:>5d}\n")

    # Sanity-cross-reference with existing split manifest
    split_chip_to_split: dict[str, str] = {}
    with SPLIT_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            if r["chip_id"] not in split_chip_to_split:
                split_chip_to_split[r["chip_id"]] = r["split"]
    print(f"Split manifest has {len(split_chip_to_split):,} unique chip_ids "
          f"({sum(1 for s in split_chip_to_split.values() if s=='train')} train, "
          f"{sum(1 for s in split_chip_to_split.values() if s=='val')} val, "
          f"{sum(1 for s in split_chip_to_split.values() if s=='test')} test)\n")

    # Audit-based Dice cross-ref
    audit_dice: dict[str, float] = {}
    with AUDIT_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["label"] == "1":
                audit_dice[r["chip_id"]] = float(r["dice"])

    # Build outputs
    # 1) keep_oversample.csv — hard-mining set; restrict to TRAIN chips
    #    (we don't oversample val/test even if user marked them keep, since those
    #    chips guide model selection / final eval).
    keep_train: list[dict] = []
    keep_nontrain: list[dict] = []
    for cid in by_decision["keep"]:
        sp = split_chip_to_split.get(cid, "?")
        d  = audit_dice.get(cid, float("nan"))
        rec = {"chip_id": cid, "split": sp, "audit_dice": f"{d:.4f}" if not np.isnan(d) else ""}
        (keep_train if sp == "train" else keep_nontrain).append(rec)

    keep_path = out_dir / "keep_oversample.csv"
    with keep_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_id", "split", "audit_dice"])
        w.writeheader()
        for r in keep_train: w.writerow(r)
    print(f"keep_oversample.csv: {len(keep_train)} train chips (val/test 'keep' chips not oversampled: {len(keep_nontrain)})")
    if keep_train:
        ds = [float(r["audit_dice"]) for r in keep_train if r["audit_dice"]]
        if ds:
            print(f"  audit-dice on hard-mine set: median={np.median(ds):.3f}  "
                  f"q25={np.quantile(ds,.25):.3f}  count<0.85={sum(1 for d in ds if d<0.85)}/{len(ds)}")

    # 2) negative_override.csv — chips to treat as hard negative
    neg_path = out_dir / "negative_override.csv"
    with neg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_id", "split", "audit_dice"])
        w.writeheader()
        for cid in by_decision["negative"]:
            sp = split_chip_to_split.get(cid, "?")
            d  = audit_dice.get(cid, float("nan"))
            w.writerow({"chip_id": cid, "split": sp,
                        "audit_dice": f"{d:.4f}" if not np.isnan(d) else ""})
    print(f"negative_override.csv: {len(by_decision['negative'])} chips")

    # 3) rejected_log.csv (record only — these chips remain in training)
    rej_path = out_dir / "rejected_log.csv"
    with rej_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_id", "split", "audit_dice"])
        w.writeheader()
        for cid in by_decision["reject"]:
            sp = split_chip_to_split.get(cid, "?")
            d  = audit_dice.get(cid, float("nan"))
            w.writerow({"chip_id": cid, "split": sp,
                        "audit_dice": f"{d:.4f}" if not np.isnan(d) else ""})
    print(f"rejected_log.csv: {len(by_decision['reject'])} chips (logged only — they STAY in training)")

    # 4) Summary JSON
    summary = {
        "decisions_csv": str(decisions_path.relative_to(REPO)),
        "decision_counts": {k: len(v) for k, v in by_decision.items()},
        "hard_mining_train_chips": len(keep_train),
        "keep_chips_not_in_train": len(keep_nontrain),
        "negative_override_chips": len(by_decision["negative"]),
        "rejected_unchanged_chips": len(by_decision["reject"]),
        "interpretation_note": (
            "User used 'reject' to mean 'no action needed' (not 'drop from training'). "
            "All 'reject' chips stay in training unchanged. Only 'keep' and 'negative' "
            "decisions modify training behaviour."
        ),
    }
    (out_dir / "decisions_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {(out_dir / 'decisions_summary.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
