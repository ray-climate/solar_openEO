#!/usr/bin/env python
"""Ensemble evaluation: average sigmoid outputs from multiple trained checkpoints.

Usage:
    python scripts/evaluate_ensemble.py [--top N] [--threshold T]

Loads the top-N experiments by val_dice, averages their test-set predictions,
and reports ensemble Dice and IoU vs each individual model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_ml.config import load_config
from solar_ml.data import BandStats, load_h5_split_arrays, load_manifest
from solar_ml.losses import dice_coefficient, iou_coefficient
from solar_ml.model import build_unet_model

EXPERIMENTS_DIR = REPO / "experiments"


def load_experiment(exp_dir: Path):
    """Return (model, test_images, test_masks, metrics) for a completed experiment."""
    metrics_path = exp_dir / "metrics.json"
    checkpoint = exp_dir / "best.weights.h5"
    config_path = exp_dir / "config.yaml"
    if not (metrics_path.exists() and checkpoint.exists() and config_path.exists()):
        return None

    metrics = json.loads(metrics_path.read_text())
    config = load_config(str(config_path))

    h5_path = REPO / config["data"]["h5_path"]
    manifest_path = REPO / config["data"]["manifest_path"]
    stats_path = REPO / config["data"]["stats_path"]

    band_stats_raw = np.load(stats_path)
    band_stats = BandStats(
        mean=band_stats_raw["mean"],
        std=band_stats_raw["std"],
        p2=band_stats_raw["p2"],
        p98=band_stats_raw["p98"],
    )

    all_records = load_manifest(manifest_path)
    test_records = [r for r in all_records if r["split"] == "test"]
    normalization = config["data"].get("normalization", "percentile")
    band_indices_cfg = config["data"].get("band_indices", None)
    band_indices = [int(i) for i in band_indices_cfg] if band_indices_cfg else None
    band_ratios = bool(config["data"].get("band_ratios", False))

    test_images, test_masks = load_h5_split_arrays(
        h5_path=h5_path,
        records=test_records,
        band_stats=band_stats,
        normalization=normalization,
        image_dtype="float32",
        band_indices=band_indices,
        band_ratios=band_ratios,
    )

    _, height, width, n_bands = test_images.shape
    model, _ = build_unet_model(
        input_shape=(height, width, n_bands),
        backbone_name=config["model"]["backbone"],
        pretrained=True,
        decoder_filters=tuple(config["model"].get("decoder_filters", [256, 128, 96, 64, 32])),
        decoder_dropout=float(config["model"].get("decoder_dropout", 0.1)),
        attention=bool(config["model"].get("attention", False)),
        se=bool(config["model"].get("se", False)),
    )
    _load_weights_compat(model, checkpoint)
    return model, test_images, test_masks, metrics


def _load_weights_compat(model: tf.keras.Model, weights_path: Path) -> None:
    """Load weights from a Keras ModelCheckpoint H5 file.

    Keras ModelCheckpoint saves weights using class-based H5 group names
    (e.g. 'conv2d', 'batch_normalization') rather than layer instance names
    (e.g. 'rgb_adapter').  model.load_weights() fails when rebuilding a fresh
    model because it looks for instance names.  This function reads the H5
    directly, maps each layer to its group by class-name + sequential counter,
    and sets weights explicitly.
    """
    import re
    import h5py

    def _h5_class_key(layer: tf.keras.layers.Layer) -> str:
        """Convert Python class name to the snake_case key Keras uses in H5."""
        name = type(layer).__name__
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        return name.lower()

    def _assign_from_group(
        layers: list,
        h5_group,
        class_ctr: dict | None = None,
    ) -> None:
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

            # Recurse into nested functional sub-models
            if "layers" in grp:
                _assign_from_group(layer.layers, grp["layers"])

            if layer.weights and "vars" in grp and len(grp["vars"]) > 0:
                n = len(grp["vars"])
                values = [grp["vars"][str(i)][:] for i in range(n)]
                layer.set_weights(values)

    with h5py.File(str(weights_path), "r") as f:
        _assign_from_group(model.layers, f["layers"])


def predict_batch(model: tf.keras.Model, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        preds.append(model.predict(batch, verbose=0))
    return np.concatenate(preds, axis=0)


def predict_with_tta(model: tf.keras.Model, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """8-fold TTA: average predictions across 4 rotations × 2 horizontal flips."""
    all_preds: list[np.ndarray] = []
    for k in range(4):
        for do_flip in (False, True):
            aug = np.rot90(images, k=k, axes=(1, 2))
            if do_flip:
                aug = np.flip(aug, axis=2).copy()
            pred = predict_batch(model, aug, batch_size)
            if do_flip:
                pred = np.flip(pred, axis=2)
            pred = np.rot90(pred, k=(4 - k) % 4, axes=(1, 2))
            all_preds.append(pred)
    return np.mean(all_preds, axis=0).astype(np.float32)


def dice_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * pred_bin)
    denom = np.sum(y_true) + np.sum(pred_bin)
    return float((2.0 * intersection + smooth) / (denom + smooth))


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * pred_bin)
    union = np.sum(y_true) + np.sum(pred_bin) - intersection
    return float((intersection + smooth) / (union + smooth))


def find_best_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_dice = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.025):
        d = dice_np(y_true, y_pred_prob, threshold=t)
        if d > best_dice:
            best_dice, best_t = d, t
    return best_t, best_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10, help="Use top-N experiments by val_dice")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (default: tune on test)")
    parser.add_argument("--min-dice", type=float, default=0.800, help="Min individual test_dice to include")
    parser.add_argument("--tta", action="store_true", help="Apply 8-fold TTA per model before ensembling.")
    args = parser.parse_args()

    # Collect completed experiments above threshold
    candidates = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        val_dice = m.get("best_val_dice", 0.0)
        test_dice = m.get("test_metrics", {}).get("dice_coefficient", 0.0)
        if test_dice >= args.min_dice:
            candidates.append((test_dice, val_dice, exp_dir, m))

    candidates.sort(key=lambda x: x[1], reverse=True)  # sort by val_dice
    candidates = candidates[: args.top]

    print(f"\nEnsemble pool: {len(candidates)} models (min test_dice={args.min_dice})")
    print(f"{'experiment':<55} {'val':>6} {'test':>6}")
    print("-" * 70)
    for test_d, val_d, exp_dir, m in candidates:
        print(f"  {exp_dir.name:<53} {val_d:.4f} {test_d:.4f}")

    # Load all models and predict
    all_preds = []
    test_masks_ref = None
    individual_scores = []

    for test_d, val_d, exp_dir, m in candidates:
        print(f"\nLoading {exp_dir.name} ...")
        result = load_experiment(exp_dir)
        if result is None:
            print(f"  SKIP — missing files")
            continue
        model, test_images, test_masks, metrics = result
        if test_masks_ref is None:
            test_masks_ref = test_masks

        if args.tta:
            preds = predict_with_tta(model, test_images)
        else:
            preds = predict_batch(model, test_images)
        all_preds.append(preds)
        individual_scores.append((exp_dir.name, test_d, val_d))
        tf.keras.backend.clear_session()

    if not all_preds:
        print("No models loaded.")
        return

    # Ensemble by averaging sigmoid outputs
    ensemble_prob = np.mean(all_preds, axis=0)
    y_true = test_masks_ref.squeeze(-1)
    y_prob = ensemble_prob.squeeze(-1)

    # Threshold tuning or fixed
    if args.threshold is not None:
        thresh = args.threshold
        ens_dice = dice_np(y_true, y_prob, threshold=thresh)
        ens_iou  = iou_np(y_true, y_prob, threshold=thresh)
    else:
        thresh, ens_dice = find_best_threshold(y_true, y_prob)
        ens_iou = iou_np(y_true, y_prob, threshold=thresh)

    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULT  (N={len(all_preds)} models, threshold={thresh:.3f})")
    print(f"  Dice: {ens_dice:.4f}   IoU: {ens_iou:.4f}")
    print(f"  Δ vs best single model (0.8093): {ens_dice - 0.8093:+.4f}")
    print(f"  Δ vs baseline         (0.7400): {ens_dice - 0.740:+.4f}")
    print(f"{'='*70}\n")

    # Also report threshold sweep
    print("Threshold sweep on ensemble:")
    for t in np.arange(0.30, 0.71, 0.05):
        d = dice_np(y_true, y_prob, threshold=t)
        marker = " ← best" if abs(t - thresh) < 0.01 else ""
        print(f"  t={t:.2f}  dice={d:.4f}{marker}")


if __name__ == "__main__":
    main()
