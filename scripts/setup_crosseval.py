#!/usr/bin/env python
"""Build 4 cross-eval experiment dirs: {v3_r152, round3_r101} × {easy, hard} slices.

Slices come from v2's manifest (so indices align with final_dataset_repacked.h5):
  - easy = chip_ids in (v1_test ∩ v2_test)
  - hard = chip_ids in (v2_test \\ v1_all)        # entirely new harvest
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import yaml

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
V1_MANIFEST = REPO / "outputs/training_prep/final_v1/split_manifest.csv"
V2_MANIFEST = REPO / "outputs/training_prep/final_v2/split_manifest.csv"
REPACKED_H5 = REPO / "outputs/final/final_dataset_repacked.h5"

CROSSEVAL_DIR = REPO / "experiments/crosseval"
MANIFEST_DIR = CROSSEVAL_DIR / "manifests"


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["index", "chip_id", "split", "col", "row", "block_col", "block_row"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def build_slices() -> dict[str, list[dict]]:
    v1 = load_manifest(V1_MANIFEST)
    v2 = load_manifest(V2_MANIFEST)

    v1_test = {r["chip_id"] for r in v1 if r["split"] == "test"}
    v1_all = {r["chip_id"] for r in v1}

    easy_ids = set()
    hard_ids = set()
    for r in v2:
        if r["split"] != "test":
            continue
        if r["chip_id"] in v1_test:
            easy_ids.add(r["chip_id"])
        elif r["chip_id"] not in v1_all:
            hard_ids.add(r["chip_id"])

    # Build rows from v2 (which has the right indices for the repacked H5),
    # force split="test" so the eval script's split filter picks them up.
    # Dedupe by chip_id — v2 has duplicate rows for oversampled high-panel-frac
    # positives and we want each chip scored exactly once.
    easy_rows: list[dict] = []
    hard_rows: list[dict] = []
    seen: set[str] = set()
    for r in v2:
        cid = r["chip_id"]
        if cid in seen:
            continue
        if cid in easy_ids:
            new = dict(r); new["split"] = "test"; easy_rows.append(new); seen.add(cid)
        elif cid in hard_ids:
            new = dict(r); new["split"] = "test"; hard_rows.append(new); seen.add(cid)

    print(f"easy_intersection: {len(easy_rows)} rows")
    print(f"hard_newharvest  : {len(hard_rows)} rows")
    return {"easy_intersection": easy_rows, "hard_newharvest": hard_rows}


def make_eval_dir(
    name: str,
    src_exp: Path,
    manifest_path: Path,
    stats_path: Path,
) -> Path:
    out = CROSSEVAL_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    src_cfg = yaml.safe_load((src_exp / "config.yaml").read_text())
    src_cfg["experiment"]["name"] = name
    src_cfg["data"]["h5_path"] = str(REPACKED_H5.relative_to(REPO))
    src_cfg["data"]["manifest_path"] = str(manifest_path.relative_to(REPO))
    src_cfg["data"]["stats_path"] = str(stats_path.relative_to(REPO))
    src_cfg["data"]["augment"] = False
    (out / "config.yaml").write_text(yaml.safe_dump(src_cfg, sort_keys=False))

    # symlink weights so we don't copy 200 MB
    link = out / "best.weights.h5"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((src_exp / "best.weights.h5").resolve())
    return out


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    slices = build_slices()
    easy_csv = MANIFEST_DIR / "easy_intersection.csv"
    hard_csv = MANIFEST_DIR / "hard_newharvest.csv"
    write_manifest(easy_csv, slices["easy_intersection"])
    write_manifest(hard_csv, slices["hard_newharvest"])

    v3_src = REPO / "experiments/exp_v3_r152_dropout02"
    r3_src = REPO / "experiments/exp_round3_r101_dropout02"
    v1_stats = REPO / "outputs/training_prep/final_v1/band_stats.npz"
    v2_stats = REPO / "outputs/training_prep/final_v2/band_stats.npz"

    dirs = [
        make_eval_dir("xeval_v3_r152_on_easy",   v3_src, easy_csv, v1_stats),
        make_eval_dir("xeval_v3_r152_on_hard",   v3_src, hard_csv, v1_stats),
        make_eval_dir("xeval_round3_r101_on_easy", r3_src, easy_csv, v2_stats),
        make_eval_dir("xeval_round3_r101_on_hard", r3_src, hard_csv, v2_stats),
    ]
    print("\nCreated:")
    for d in dirs:
        print(f"  {d.relative_to(REPO)}")


if __name__ == "__main__":
    main()
