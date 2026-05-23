#!/usr/bin/env python
"""Read the 6 TTA cross-eval results, pick the champion, and update model_registry.yaml.

Run after the 6 xeval_tta_* jobs land. Champion is the model with highest hard-slice
pixel-Dice. Threshold for production is the model's best threshold on the hard slice
(deployment-relevant distribution).
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import yaml

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
CROSSEVAL = REPO / "experiments/crosseval"
REGISTRY = REPO / "openeo_udp/model_registry.yaml"

# label -> (easy_dir, hard_dir, source_exp, backbone, decoder_filters, decoder_dropout)
CANDIDATES = {
    "round3_r101": (
        "xeval_tta_round3_r101_on_easy", "xeval_tta_round3_r101_on_hard",
        "exp_round3_r101_dropout02", "resnet101", [256, 128, 96, 64, 32], 0.2,
    ),
    "r3_r152_widedecoder": (
        "xeval_tta_r3_r152_widedecoder_on_easy", "xeval_tta_r3_r152_widedecoder_on_hard",
        "exp_round3_r152_widedecoder", "resnet152", [384, 192, 128, 96, 48], 0.2,
    ),
    "r3_r152_dropout03": (
        "xeval_tta_r3_r152_dropout03_on_easy", "xeval_tta_r3_r152_dropout03_on_hard",
        "exp_round3_r152_dropout03", "resnet152", [256, 128, 96, 64, 32], 0.3,
    ),
}


def load_summary(dirname: str) -> dict | None:
    p = CROSSEVAL / dirname / "evaluation_test/summary_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    rows = []
    for label, (e_d, h_d, exp, backbone, filters, dropout) in CANDIDATES.items():
        e, h = load_summary(e_d), load_summary(h_d)
        if e is None or h is None:
            print(f"  [WAIT] {label}: missing eval ({e_d if e is None else h_d})")
            continue
        rows.append({
            "label": label, "exp": exp, "backbone": backbone,
            "filters": filters, "dropout": dropout,
            "easy_dice": e["best_pixel_dice"], "easy_thr": e["best_pixel_dice_threshold"],
            "easy_p": e["best_pixel_precision"], "easy_r": e["best_pixel_recall"],
            "hard_dice": h["best_pixel_dice"], "hard_thr": h["best_pixel_dice_threshold"],
            "hard_p": h["best_pixel_precision"], "hard_r": h["best_pixel_recall"],
            "hard_iou": h["best_pixel_iou"],
        })

    if len(rows) < len(CANDIDATES):
        print(f"\n{len(rows)}/{len(CANDIDATES)} TTA evals done. Re-run when all complete.")
        return

    print("\nTTA cross-eval results:")
    print(f"  {'Model':24s} {'Easy Dice':>11s} (thr) {'Hard Dice':>11s} (thr)  {'Hard P':>7s} {'Hard R':>7s}")
    for r in sorted(rows, key=lambda x: x["hard_dice"], reverse=True):
        print(f"  {r['label']:24s} {r['easy_dice']:.4f}  ({r['easy_thr']})  {r['hard_dice']:.4f}  ({r['hard_thr']})  "
              f"{r['hard_p']:.3f}  {r['hard_r']:.3f}")

    champ = max(rows, key=lambda x: x["hard_dice"])
    print(f"\nCHAMPION: {champ['label']}  (hard Dice {champ['hard_dice']:.4f})")
    print(f"  Production threshold: {champ['hard_thr']}  (best-pixel-Dice on hard slice)")

    # Backup current registry
    backup = REGISTRY.with_suffix(".yaml.bak_v1")
    if not backup.exists():
        shutil.copy2(REGISTRY, backup)
        print(f"  Backed up old registry -> {backup.name}")

    # Update registry
    text = REGISTRY.read_text()
    reg = yaml.safe_load(text)
    am = reg["active_model"]
    am.update({
        "name": f"{champ['label']}_focal_dice_zscore_round3_v2",
        "experiment": champ["exp"],
        "weights_url": f"https://github.com/ray-climate/solar_openEO/releases/download/v2.0.0/best.weights.h5",
        "band_stats_url": f"https://github.com/ray-climate/solar_openEO/releases/download/v2.0.0/band_stats.npz",
        "weights_local": f"experiments/{champ['exp']}/best.weights.h5",
        "band_stats_local": "outputs/training_prep/final_v2/band_stats.npz",
        "backbone": champ["backbone"],
        "decoder_filters": champ["filters"],
        "decoder_dropout": float(champ["dropout"]),
        "threshold": float(champ["hard_thr"]),
        "version": "2.0.0",
        "released": str(date.today()),
        "test_dice": round(float(champ["hard_dice"]), 4),
        "test_iou": round(float(champ["hard_iou"]), 4),
        "training_chips": 26940,
        "training_loss": "focal_dice",
    })
    REGISTRY.write_text(yaml.safe_dump(reg, sort_keys=False))
    print(f"\nUpdated {REGISTRY.relative_to(REPO)}")
    print("URLs are placeholders — user said skip delivery (step 6); leave as TBD for now.")
    print("\nNext: run smoke test")
    print(f"  conda run -n tf-gpu python openeo_udp/tests/test_udf_local.py --n-chips 5 -v")


if __name__ == "__main__":
    main()
