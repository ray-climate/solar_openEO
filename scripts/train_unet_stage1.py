#!/usr/bin/env python
"""Train a 13-band Sentinel-2 U-Net on the Stage-1 HDF5 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_ml.config import load_config
from solar_ml.data import (
    BandStats,
    H5SegmentationSequence,
    build_in_memory_dataset,
    load_h5_split_arrays,
    load_manifest,
)
from solar_ml.losses import dice_coefficient, get_loss, iou_coefficient
from solar_ml.model import build_unet_model


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to YAML experiment config.")
    parser.add_argument(
        "--experiment",
        "-e",
        default=None,
        help="Override experiment name. Defaults to config experiment.name.",
    )
    parser.add_argument("--resume", default=None, help="Optional weights checkpoint to resume from.")
    parser.add_argument("--dry-run", action="store_true", help="Build model and one batch, then exit.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def setup_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class CosineDecayCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        total_epochs: int,
        base_lr: float,
        min_lr: float,
        warmup_epochs: int,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epoch_offset = epoch_offset

    def on_epoch_begin(self, epoch: int, logs=None):
        effective_epoch = max(epoch - self.epoch_offset, 0)
        if effective_epoch < self.warmup_epochs:
            lr = self.base_lr * (effective_epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (effective_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


def load_split_records(manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    records = load_manifest(manifest_path)
    grouped = {"train": [], "val": [], "test": []}
    for row in records:
        grouped[row["split"]].append(row)
    return grouped


def write_history_csv(history: dict[str, list[float]], out_csv: Path) -> None:
    keys = list(history)
    epochs = len(history[keys[0]]) if keys else 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for epoch in range(epochs):
            writer.writerow([epoch + 1, *[history[key][epoch] for key in keys]])


def write_summary(
    output_dir: Path,
    config: dict[str, Any],
    history: dict[str, list[float]],
    test_metrics: dict[str, float],
) -> None:
    best_epoch = None
    best_val_dice = None
    if "val_dice_coefficient" in history and history["val_dice_coefficient"]:
        raw_values = np.asarray(history["val_dice_coefficient"], dtype=np.float32)
        finite_mask = np.isfinite(raw_values)
        if np.any(finite_mask):
            finite_indices = np.flatnonzero(finite_mask)
            best_index = int(finite_indices[np.argmax(raw_values[finite_mask])])
            best_epoch = best_index + 1
            best_val_dice = float(raw_values[best_index])

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def close_loader(loader: Any) -> None:
    close = getattr(loader, "close", None)
    if callable(close):
        close()


def build_metrics() -> list[Any]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        dice_coefficient,
        iou_coefficient,
    ]


def compile_model(
    model: tf.keras.Model,
    loss_fn,
    learning_rate: float,
    weight_decay: float,
    steps_per_execution: int,
) -> None:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=build_metrics(),
        steps_per_execution=steps_per_execution,
    )


def merge_histories(*histories: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def build_callbacks(
    output_dir: Path,
    config: dict[str, Any],
    total_epochs: int,
    base_learning_rate: float,
    epoch_offset: int = 0,
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "latest.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=int(config["training"].get("early_stopping_patience", 15)),
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    schedule = config["training"].get("lr_schedule", "cosine").lower()
    if epoch_offset > 0:
        warmup_epochs = int(config["training"].get("unfreeze_warmup_epochs", 0))
    else:
        warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))

    if schedule == "cosine":
        callbacks.append(
            CosineDecayCallback(
                total_epochs=total_epochs,
                base_lr=base_learning_rate,
                min_lr=float(config["training"].get("min_learning_rate", 1e-6)),
                warmup_epochs=warmup_epochs,
                epoch_offset=epoch_offset,
            )
        )
    elif schedule == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice_coefficient",
                mode="max",
                factor=0.5,
                patience=max(2, int(config["training"].get("lr_patience", 5))),
                min_lr=float(config["training"].get("min_learning_rate", 1e-6)),
                verbose=1,
            )
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    return callbacks


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    exp_name = args.experiment or config["experiment"]["name"]
    output_root = Path(config["training"]["output_root"])
    output_dir = output_root / exp_name
    setup_logging(output_dir, args.verbose)

    LOGGER.info("Experiment: %s", exp_name)
    LOGGER.info("Config: %s", Path(args.config).resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    if config["training"].get("use_mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("Enabled mixed precision policy.")

    h5_path = Path(config["data"]["h5_path"])
    manifest_path = Path(config["data"]["manifest_path"])
    stats_path = Path(config["data"]["stats_path"])
    for path in (h5_path, manifest_path, stats_path):
        if not path.exists():
            LOGGER.error("Required input not found: %s", path)
            return 1

    split_records = load_split_records(manifest_path)
    band_stats = BandStats.from_npz(stats_path)

    batch_size = int(config["training"]["batch_size"])
    normalization = config["data"].get("normalization", "percentile")
    input_backend = str(config["training"].get("input_backend", "tfdata")).lower()
    image_dtype = str(config["training"].get("in_memory_image_dtype", "float32")).lower()

    band_indices_cfg = config["data"].get("band_indices", None)
    band_indices: list[int] | None = [int(i) for i in band_indices_cfg] if band_indices_cfg else None
    augment_spectral = bool(config["data"].get("augment_spectral", False))
    augment_noise = bool(config["data"].get("augment_noise", False))
    augment_jitter = bool(config["data"].get("augment_jitter", False))
    augment_scale = bool(config["data"].get("augment_scale", False))
    band_ratios = bool(config["data"].get("band_ratios", False))

    if band_indices is not None:
        LOGGER.info("Band selection enabled: %s (%d bands)", band_indices, len(band_indices))
    if augment_spectral:
        LOGGER.info("Spectral brightness augmentation enabled.")
    if augment_noise:
        LOGGER.info("Noise augmentation enabled (per-pixel Gaussian, sigma=0.05).")
    if augment_jitter:
        LOGGER.info("Brightness jitter augmentation enabled (per-band additive, sigma=0.10).")
    if augment_scale:
        LOGGER.info("Scale jitter augmentation enabled (random crop 75-100%%, resize back).")
    if band_ratios:
        LOGGER.info("Band ratios enabled: appending NDVI, NDBI, NDWI channels.")

    if input_backend == "tfdata":
        LOGGER.info("Using tf.data input backend with in-memory split caching.")
        train_images, train_masks = load_h5_split_arrays(
            h5_path=h5_path,
            records=split_records["train"],
            band_stats=band_stats,
            normalization=normalization,
            image_dtype=image_dtype,
            band_indices=band_indices,
            band_ratios=band_ratios,
        )
        val_images, val_masks = load_h5_split_arrays(
            h5_path=h5_path,
            records=split_records["val"],
            band_stats=band_stats,
            normalization=normalization,
            image_dtype=image_dtype,
            band_indices=band_indices,
            band_ratios=band_ratios,
        )
        test_images, test_masks = load_h5_split_arrays(
            h5_path=h5_path,
            records=split_records["test"],
            band_stats=band_stats,
            normalization=normalization,
            image_dtype=image_dtype,
            band_indices=band_indices,
            band_ratios=band_ratios,
        )

        train_data = build_in_memory_dataset(
            train_images,
            train_masks,
            batch_size=batch_size,
            shuffle=True,
            augment=bool(config["data"].get("augment", True)),
            augment_spectral=augment_spectral,
            augment_noise=augment_noise,
            augment_jitter=augment_jitter,
            augment_scale=augment_scale,
            seed=seed,
            shuffle_buffer=int(config["training"].get("shuffle_buffer", len(train_images))),
            num_parallel_calls=int(config["training"].get("tfdata_num_parallel_calls", 0)) or None,
            prefetch_buffer=int(config["training"].get("tfdata_prefetch_buffer", 0)) or None,
            drop_remainder=bool(config["training"].get("drop_remainder", True)),
        )
        val_data = build_in_memory_dataset(
            val_images,
            val_masks,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            seed=seed,
            num_parallel_calls=None,
            prefetch_buffer=int(config["training"].get("tfdata_prefetch_buffer", 0)) or None,
        )
        test_data = build_in_memory_dataset(
            test_images,
            test_masks,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            seed=seed,
            num_parallel_calls=None,
            prefetch_buffer=int(config["training"].get("tfdata_prefetch_buffer", 0)) or None,
        )
    elif input_backend == "sequence":
        LOGGER.info("Using Sequence input backend.")
        train_data = H5SegmentationSequence(
            h5_path=h5_path,
            records=split_records["train"],
            band_stats=band_stats,
            batch_size=batch_size,
            shuffle=True,
            augment=bool(config["data"].get("augment", True)),
            normalization=normalization,
            seed=seed,
        )
        val_data = H5SegmentationSequence(
            h5_path=h5_path,
            records=split_records["val"],
            band_stats=band_stats,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            normalization=normalization,
            seed=seed,
        )
        test_data = H5SegmentationSequence(
            h5_path=h5_path,
            records=split_records["test"],
            band_stats=band_stats,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            normalization=normalization,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported input_backend: {input_backend}")

    if input_backend == "tfdata":
        _, height, width, n_bands = train_images.shape
    else:
        with h5py.File(h5_path, "r") as h5:
            _, h5_n_bands, height, width = h5["images"].shape
        n_bands = len(band_indices) if band_indices is not None else h5_n_bands
    input_shape = (height, width, n_bands)

    model, encoder = build_unet_model(
        input_shape=input_shape,
        backbone_name=config["model"]["backbone"],
        pretrained=bool(config["model"].get("pretrained", True)),
        decoder_filters=tuple(config["model"].get("decoder_filters", [256, 128, 96, 64, 32])),
        decoder_dropout=float(config["model"].get("decoder_dropout", 0.1)),
        attention=bool(config["model"].get("attention", False)),
        se=bool(config["model"].get("se", False)),
    )

    if args.resume:
        model.load_weights(args.resume)
        LOGGER.info("Loaded checkpoint weights: %s", args.resume)

    if config["training"].get("freeze_backbone_epochs", 0) > 0 and not args.resume:
        encoder.trainable = False
        LOGGER.info("Encoder frozen for first %d epochs.", config["training"]["freeze_backbone_epochs"])

    base_learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    steps_per_execution = int(config["training"].get("steps_per_execution", 1))
    loss_fn = get_loss(
        config["training"]["loss_name"],
        bce_weight=float(config["training"].get("bce_weight", 1.0)),
        dice_weight=float(config["training"].get("dice_weight", 1.0)),
        focal_weight=float(config["training"].get("focal_weight", 1.0)),
        tversky_alpha=float(config["training"].get("tversky_alpha", 0.3)),
        tversky_beta=float(config["training"].get("tversky_beta", 0.7)),
        focal_tversky_gamma=float(config["training"].get("focal_tversky_gamma", 0.75)),
    )
    compile_model(
        model=model,
        loss_fn=loss_fn,
        learning_rate=base_learning_rate,
        weight_decay=weight_decay,
        steps_per_execution=steps_per_execution,
    )

    model_summary_path = output_dir / "model_summary.txt"
    with model_summary_path.open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    LOGGER.info("Train samples: %d", len(split_records["train"]))
    LOGGER.info("Val samples:   %d", len(split_records["val"]))
    LOGGER.info("Test samples:  %d", len(split_records["test"]))

    if args.dry_run:
        if input_backend == "tfdata":
            batch_x, batch_y = next(iter(train_data.take(1)))
        else:
            batch_x, batch_y = train_data[0]
        preds = model(batch_x[:2], training=False)
        LOGGER.info(
            "Dry run OK. batch_x=%s batch_y=%s preds=%s",
            batch_x.shape,
            batch_y.shape,
            preds.shape,
        )
        close_loader(train_data)
        close_loader(val_data)
        close_loader(test_data)
        return 0

    fit_kwargs: dict[str, Any] = {
        "x": train_data,
        "validation_data": val_data,
        "epochs": int(config["training"]["epochs"]),
        "verbose": 1,
    }
    if input_backend == "sequence":
        fit_kwargs.update(
            workers=int(config["training"].get("loader_workers", 1)),
            use_multiprocessing=bool(config["training"].get("loader_use_multiprocessing", False)),
            max_queue_size=int(config["training"].get("loader_max_queue_size", 10)),
        )

    total_epochs = int(config["training"]["epochs"])
    freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
    unfreeze_learning_rate = float(config["training"].get("unfreeze_learning_rate", base_learning_rate))

    if freeze_epochs > 0 and not args.resume:
        phase1_kwargs = dict(fit_kwargs)
        phase1_epochs = min(freeze_epochs, total_epochs)
        phase1_kwargs["epochs"] = phase1_epochs
        phase1_kwargs["callbacks"] = build_callbacks(
            output_dir=output_dir,
            config=config,
            total_epochs=phase1_epochs,
            base_learning_rate=base_learning_rate,
        )
        phase1_history = model.fit(**phase1_kwargs)

        if freeze_epochs < total_epochs and not model.stop_training:
            encoder.trainable = True
            compile_model(
                model=model,
                loss_fn=loss_fn,
                learning_rate=unfreeze_learning_rate,
                weight_decay=weight_decay,
                steps_per_execution=steps_per_execution,
            )
            model.stop_training = False
            LOGGER.info("Encoder unfrozen at epoch %d.", freeze_epochs)

            phase2_kwargs = dict(fit_kwargs)
            phase2_kwargs["initial_epoch"] = freeze_epochs
            phase2_kwargs["callbacks"] = build_callbacks(
                output_dir=output_dir,
                config=config,
                total_epochs=total_epochs - freeze_epochs,
                base_learning_rate=unfreeze_learning_rate,
                epoch_offset=freeze_epochs,
            )
            phase2_history = model.fit(**phase2_kwargs)
            merged_history = merge_histories(phase1_history.history, phase2_history.history)
        else:
            merged_history = phase1_history.history
    else:
        fit_kwargs["callbacks"] = build_callbacks(
            output_dir=output_dir,
            config=config,
            total_epochs=total_epochs,
            base_learning_rate=base_learning_rate,
        )
        history = model.fit(**fit_kwargs)
        merged_history = history.history

    write_history_csv(merged_history, output_dir / "history.csv")

    best_weights = output_dir / "best.weights.h5"
    if best_weights.exists():
        model.load_weights(best_weights)

    test_metrics_raw = model.evaluate(test_data, return_dict=True, verbose=1)
    test_metrics = {key: float(value) for key, value in test_metrics_raw.items()}
    write_summary(output_dir, config, merged_history, test_metrics)

    LOGGER.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
    close_loader(train_data)
    close_loader(val_data)
    close_loader(test_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
