#!/usr/bin/env python
"""Summarize completed training runs for backbone/setup comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_best_val_dice(metrics: dict, metrics_path: Path) -> float | None:
    value = metrics.get("best_val_dice")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)

    history_path = metrics_path.parent / "history.csv"
    if not history_path.exists():
        return None

    best = None
    with history_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("val_dice_coefficient")
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            if best is None or value > best:
                best = value
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--out-csv", default="experiments/run_summary.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in sorted(experiments_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        config = metrics.get("config", {})
        row = {
            "experiment": metrics_path.parent.name,
            "backbone": config.get("model", {}).get("backbone"),
            "loss_name": config.get("training", {}).get("loss_name"),
            "freeze_backbone_epochs": config.get("training", {}).get("freeze_backbone_epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_dice": read_best_val_dice(metrics, metrics_path),
            "test_loss": metrics.get("test_metrics", {}).get("loss"),
            "test_dice": metrics.get("test_metrics", {}).get("dice_coefficient"),
            "test_iou": metrics.get("test_metrics", {}).get("iou_coefficient"),
            "test_precision": metrics.get("test_metrics", {}).get("precision"),
            "test_recall": metrics.get("test_metrics", {}).get("recall"),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["test_dice"] is None, -(row["test_dice"] or -1.0)))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "backbone",
                "loss_name",
                "freeze_backbone_epochs",
                "best_epoch",
                "best_val_dice",
                "test_loss",
                "test_dice",
                "test_iou",
                "test_precision",
                "test_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Run summary written -> {out_csv}")
    for row in rows[:10]:
        print(
            f"{row['experiment']:28s} backbone={row['backbone']:14s} "
            f"test_dice={row['test_dice']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
