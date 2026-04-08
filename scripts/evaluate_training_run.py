#!/usr/bin/env python
"""Evaluate a trained segmentation run across thresholds and render failure galleries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_ml.data import BandStats, load_h5_split_arrays, load_manifest
from solar_ml.model import build_unet_model


RGB_BAND_INDICES = (3, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="Path to an experiment directory.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated probability thresholds to score.",
    )
    parser.add_argument(
        "--gallery-threshold",
        default="best_dice",
        help="Threshold for gallery masks, or 'best_dice' to use the best threshold from the sweep.",
    )
    parser.add_argument(
        "--gallery-count",
        type=int,
        default=18,
        help="Number of lowest-Dice chips to include in the failure gallery.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override prediction batch size. Defaults to config training.batch_size.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Apply 8-fold test-time augmentation (4 rotations × 2 flips) and average predictions.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/evaluation_<split>/.",
    )
    return parser.parse_args()


def load_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    return [row for row in load_manifest(manifest_path) if row["split"] == split]


def load_weights_compat(model: tf.keras.Model, weights_path: Path) -> None:
    """Load weights from a Keras ModelCheckpoint H5 file.

    Keras ModelCheckpoint saves with class-based H5 keys (e.g. 'conv2d') not
    instance names (e.g. 'rgb_adapter').  This reads the file directly and
    assigns weights by layer traversal order + class counter.
    """
    import re
    import h5py

    def _h5_class_key(layer: tf.keras.layers.Layer) -> str:
        name = type(layer).__name__
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        return name.lower()

    def _assign_from_group(layers, h5_group, class_ctr=None):
        if class_ctr is None:
            class_ctr = {}
        for layer in layers:
            cls_key = _h5_class_key(layer)
            cnt = class_ctr.get(cls_key, 0)
            h5_key = cls_key if cnt == 0 else f"{cls_key}_{cnt}"
            class_ctr[cls_key] = cnt + 1
            if h5_key not in h5_group:
                continue
            grp = h5_group[h5_key]
            if "layers" in grp:
                _assign_from_group(layer.layers, grp["layers"])
            if layer.weights and "vars" in grp and len(grp["vars"]) > 0:
                n = len(grp["vars"])
                layer.set_weights([grp["vars"][str(i)][:] for i in range(n)])

    with h5py.File(str(weights_path), "r") as f:
        _assign_from_group(model.layers, f["layers"])


def predict_with_tta(model: tf.keras.Model, images_hwc: np.ndarray, batch_size: int) -> np.ndarray:
    """8-fold TTA: average predictions across 4 rotations × 2 horizontal flips.

    Each augmented image is predicted, then the prediction mask is inverse-
    transformed back to the original orientation before averaging.
    """
    all_preds: list[np.ndarray] = []
    for k in range(4):
        for do_flip in (False, True):
            aug = np.rot90(images_hwc, k=k, axes=(1, 2))
            if do_flip:
                aug = np.flip(aug, axis=2).copy()
            preds = []
            for start in range(0, len(aug), batch_size):
                batch = aug[start : start + batch_size]
                preds.append(model.predict(batch, verbose=0))
            pred = np.concatenate(preds, axis=0)
            if do_flip:
                pred = np.flip(pred, axis=2)
            pred = np.rot90(pred, k=(4 - k) % 4, axes=(1, 2))
            all_preds.append(pred)
    return np.mean(all_preds, axis=0).astype(np.float32)


def build_model_for_experiment(config: dict[str, Any], input_shape: tuple[int, int, int]) -> tf.keras.Model:
    model, _ = build_unet_model(
        input_shape=input_shape,
        backbone_name=config["model"]["backbone"],
        pretrained=bool(config["model"].get("pretrained", True)),
        decoder_filters=tuple(config["model"].get("decoder_filters", [256, 128, 96, 64, 32])),
        decoder_dropout=float(config["model"].get("decoder_dropout", 0.1)),
    )
    return model


def parse_thresholds(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def safe_div(numer: np.ndarray | float, denom: np.ndarray | float) -> np.ndarray | float:
    denom_arr = np.asarray(denom, dtype=np.float64)
    numer_arr = np.asarray(numer, dtype=np.float64)
    out = np.zeros_like(numer_arr, dtype=np.float64)
    np.divide(numer_arr, denom_arr, out=out, where=denom_arr != 0)
    if np.isscalar(numer) and np.isscalar(denom):
        return float(out)
    return out


def compute_confusion_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.sum(y_true & y_pred, axis=(1, 2, 3), dtype=np.int64)
    fp = np.sum((~y_true) & y_pred, axis=(1, 2, 3), dtype=np.int64)
    fn = np.sum(y_true & (~y_pred), axis=(1, 2, 3), dtype=np.int64)
    return tp, fp, fn


def compute_threshold_metrics(
    y_true_bool: np.ndarray,
    probs: np.ndarray,
    thresholds: list[float],
) -> tuple[list[dict[str, float]], dict[float, dict[str, np.ndarray]]]:
    rows: list[dict[str, float]] = []
    per_threshold_arrays: dict[float, dict[str, np.ndarray]] = {}
    true_pixels_per_chip = np.sum(y_true_bool, axis=(1, 2, 3), dtype=np.int64)

    for threshold in thresholds:
        pred_bool = probs >= threshold
        tp_chip, fp_chip, fn_chip = compute_confusion_components(y_true_bool, pred_bool)

        tp = int(np.sum(tp_chip))
        fp = int(np.sum(fp_chip))
        fn = int(np.sum(fn_chip))

        pixel_precision = safe_div(tp, tp + fp)
        pixel_recall = safe_div(tp, tp + fn)
        pixel_dice = safe_div(2 * tp, 2 * tp + fp + fn)
        pixel_iou = safe_div(tp, tp + fp + fn)

        chip_dice = safe_div(2.0 * tp_chip, 2.0 * tp_chip + fp_chip + fn_chip)
        chip_iou = safe_div(tp_chip, tp_chip + fp_chip + fn_chip)
        pred_pixels_per_chip = np.sum(pred_bool, axis=(1, 2, 3), dtype=np.int64)
        missed_chip_rate = safe_div(np.sum(pred_pixels_per_chip == 0), len(pred_pixels_per_chip))

        row = {
            "threshold": float(threshold),
            "pixel_precision": float(pixel_precision),
            "pixel_recall": float(pixel_recall),
            "pixel_dice": float(pixel_dice),
            "pixel_iou": float(pixel_iou),
            "mean_chip_dice": float(np.mean(chip_dice)),
            "median_chip_dice": float(np.median(chip_dice)),
            "mean_chip_iou": float(np.mean(chip_iou)),
            "median_chip_iou": float(np.median(chip_iou)),
            "mean_pred_pixels_per_chip": float(np.mean(pred_pixels_per_chip)),
            "median_pred_pixels_per_chip": float(np.median(pred_pixels_per_chip)),
            "mean_true_pixels_per_chip": float(np.mean(true_pixels_per_chip)),
            "missed_chip_rate": float(missed_chip_rate),
        }
        rows.append(row)
        per_threshold_arrays[float(threshold)] = {
            "chip_dice": np.asarray(chip_dice, dtype=np.float32),
            "chip_iou": np.asarray(chip_iou, dtype=np.float32),
            "pred_pixels_per_chip": pred_pixels_per_chip.astype(np.int64, copy=False),
            "true_pixels_per_chip": true_pixels_per_chip.astype(np.int64, copy=False),
        }

    return rows, per_threshold_arrays


def choose_gallery_threshold(threshold_rows: list[dict[str, float]], raw_value: str) -> float:
    if raw_value == "best_dice":
        return float(max(threshold_rows, key=lambda row: row["pixel_dice"])["threshold"])
    return float(raw_value)


def rgb_preview(image_hwc: np.ndarray) -> np.ndarray:
    rgb = image_hwc[..., RGB_BAND_INDICES]
    return np.clip(rgb, 0.0, 1.0)


def render_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.45) -> np.ndarray:
    overlay = rgb.copy()
    mask2d = mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask2d] = (1.0 - alpha) * overlay[mask2d] + alpha * color_arr
    return overlay


def write_threshold_csv(rows: list[dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_per_chip_csv(
    records: list[dict[str, Any]],
    probs: np.ndarray,
    y_true_bool: np.ndarray,
    threshold: float,
    out_csv: Path,
) -> list[dict[str, Any]]:
    pred_bool = probs >= threshold
    tp_chip, fp_chip, fn_chip = compute_confusion_components(y_true_bool, pred_bool)
    chip_dice = safe_div(2.0 * tp_chip, 2.0 * tp_chip + fp_chip + fn_chip)
    chip_iou = safe_div(tp_chip, tp_chip + fp_chip + fn_chip)
    pred_pixels = np.sum(pred_bool, axis=(1, 2, 3), dtype=np.int64)
    true_pixels = np.sum(y_true_bool, axis=(1, 2, 3), dtype=np.int64)
    mean_prob = np.mean(probs, axis=(1, 2, 3), dtype=np.float64)
    max_prob = np.max(probs, axis=(1, 2, 3))

    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "index": int(record["index"]),
                "chip_id": record["chip_id"],
                "split": record["split"],
                "true_pixels": int(true_pixels[idx]),
                "pred_pixels": int(pred_pixels[idx]),
                "chip_dice": float(chip_dice[idx]),
                "chip_iou": float(chip_iou[idx]),
                "mean_probability": float(mean_prob[idx]),
                "max_probability": float(max_prob[idx]),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_failure_gallery(
    images_hwc: np.ndarray,
    masks_hwc: np.ndarray,
    probs: np.ndarray,
    per_chip_rows: list[dict[str, Any]],
    threshold: float,
    out_png: Path,
    max_items: int,
) -> None:
    rows_sorted = sorted(per_chip_rows, key=lambda row: (row["chip_dice"], row["pred_pixels"]))
    selected = rows_sorted[: max_items]
    if not selected:
        return

    n_items = len(selected)
    fig, axes = plt.subplots(n_items, 4, figsize=(14, max(3, 2.8 * n_items)))
    if n_items == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, item in enumerate(selected):
        sample_idx = next(i for i, row in enumerate(per_chip_rows) if row["chip_id"] == item["chip_id"])
        rgb = rgb_preview(images_hwc[sample_idx])
        true_mask = masks_hwc[sample_idx, ..., 0] > 0.5
        prob = probs[sample_idx, ..., 0]
        pred_mask = prob >= threshold

        panels = [
            (rgb, "RGB"),
            (render_overlay(rgb, true_mask, color=(0.0, 1.0, 0.1)), "GT Mask"),
            (prob, "Pred Prob"),
            (render_overlay(rgb, pred_mask, color=(1.0, 0.15, 0.15)), "Pred Mask"),
        ]
        for col_idx, (panel, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if title == "Pred Prob":
                ax.imshow(panel, cmap="magma", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title)

        axes[row_idx, 0].set_ylabel(
            f"{item['chip_id']}\nDice={item['chip_dice']:.3f}\ntrue={item['true_pixels']}\npred={item['pred_pixels']}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=28,
        )

    fig.suptitle(f"Worst-chip gallery at threshold {threshold:.2f}", y=0.995)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    config = load_config(experiment_dir)
    output_dir = Path(args.out_dir) if args.out_dir else experiment_dir / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_gpus = tf.config.list_physical_devices("GPU")
    if visible_gpus and config.get("training", {}).get("use_mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    h5_path = Path(config["data"]["h5_path"])
    manifest_path = Path(config["data"]["manifest_path"])
    stats_path = Path(config["data"]["stats_path"])
    weights_path = experiment_dir / "best.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    records = load_split_records(manifest_path, args.split)
    if not records:
        raise ValueError(f"No records found for split={args.split}")

    band_stats = BandStats.from_npz(stats_path)
    images_hwc, masks_hwc = load_h5_split_arrays(
        h5_path=h5_path,
        records=records,
        band_stats=band_stats,
        normalization=config["data"].get("normalization", "percentile"),
        image_dtype="float32",
    )
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    model = build_model_for_experiment(config, input_shape=images_hwc.shape[1:])
    load_weights_compat(model, weights_path)

    if args.tta:
        print("Running 8-fold TTA (4 rotations × 2 flips) ...")
        probs = predict_with_tta(model, images_hwc, batch_size)
    else:
        probs = model.predict(images_hwc, batch_size=batch_size, verbose=1).astype(np.float32, copy=False)
    y_true_bool = masks_hwc > 0.5
    thresholds = parse_thresholds(args.thresholds)
    threshold_rows, _ = compute_threshold_metrics(y_true_bool=y_true_bool, probs=probs, thresholds=thresholds)

    threshold_csv = output_dir / f"threshold_sweep_{args.split}.csv"
    write_threshold_csv(threshold_rows, threshold_csv)

    best_row = max(threshold_rows, key=lambda row: row["pixel_dice"])
    gallery_threshold = choose_gallery_threshold(threshold_rows, args.gallery_threshold)
    per_chip_csv = output_dir / f"per_chip_{args.split}_thr_{gallery_threshold:.2f}.csv"
    per_chip_rows = write_per_chip_csv(records, probs, y_true_bool, threshold=gallery_threshold, out_csv=per_chip_csv)
    gallery_png = output_dir / f"failure_gallery_{args.split}_thr_{gallery_threshold:.2f}.png"
    render_failure_gallery(
        images_hwc=images_hwc,
        masks_hwc=masks_hwc,
        probs=probs,
        per_chip_rows=per_chip_rows,
        threshold=gallery_threshold,
        out_png=gallery_png,
        max_items=args.gallery_count,
    )

    summary = {
        "experiment_dir": str(experiment_dir),
        "split": args.split,
        "n_samples": len(records),
        "weights_path": str(weights_path),
        "best_pixel_dice_threshold": float(best_row["threshold"]),
        "best_pixel_dice": float(best_row["pixel_dice"]),
        "best_pixel_iou": float(best_row["pixel_iou"]),
        "best_pixel_precision": float(best_row["pixel_precision"]),
        "best_pixel_recall": float(best_row["pixel_recall"]),
        "gallery_threshold": float(gallery_threshold),
        "threshold_csv": str(threshold_csv),
        "per_chip_csv": str(per_chip_csv),
        "failure_gallery_png": str(gallery_png),
    }
    (output_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
