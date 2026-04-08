"""Data preparation and loading for Stage-1 solar training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf


_STD_CHIP_RE = re.compile(r"^c(?P<col>[+-]\d+)_r(?P<row>[+-]\d+)$")
_SANITIZED_CHIP_RE = re.compile(
    r"^(?:stage1_)?c(?P<col_sign>[mp])(?P<col>\d+)_r(?P<row_sign>[mp])(?P<row>\d+)$"
)


def decode_chip_id(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_chip_id(chip_id: str) -> tuple[int | None, int | None]:
    match = _STD_CHIP_RE.match(chip_id)
    if match:
        return int(match.group("col")), int(match.group("row"))

    match = _SANITIZED_CHIP_RE.match(chip_id)
    if match:
        col = int(match.group("col"))
        row = int(match.group("row"))
        if match.group("col_sign") == "m":
            col *= -1
        if match.group("row_sign") == "m":
            row *= -1
        return col, row

    return None, None


def _bucket_value(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(16**8)


def _assign_split(bucket_value: float, train_frac: float, val_frac: float) -> str:
    if bucket_value < train_frac:
        return "train"
    if bucket_value < train_frac + val_frac:
        return "val"
    return "test"


def create_split_manifest(
    h5_path: str | Path,
    out_csv: str | Path,
    out_summary_json: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_strategy: str = "spatial_block",
    block_size_chips: int = 32,
) -> list[dict[str, Any]]:
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.")

    h5_path = Path(h5_path)
    out_csv = Path(out_csv)
    out_summary_json = Path(out_summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        chip_ids = [decode_chip_id(v) for v in h5["chip_ids"][:]]
        for index, chip_id in enumerate(chip_ids):
            col, row = parse_chip_id(chip_id)
            block_col = col // block_size_chips if col is not None else None
            block_row = row // block_size_chips if row is not None else None
            if split_strategy == "spatial_block" and block_col is not None and block_row is not None:
                split_key = f"{block_col}:{block_row}"
            else:
                split_key = chip_id

            split = _assign_split(_bucket_value(split_key), train_frac, val_frac)
            records.append(
                {
                    "index": index,
                    "chip_id": chip_id,
                    "split": split,
                    "col": col,
                    "row": row,
                    "block_col": block_col,
                    "block_row": block_row,
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "chip_id", "split", "col", "row", "block_col", "block_row"],
        )
        writer.writeheader()
        writer.writerows(records)

    counts = {
        split: sum(1 for r in records if r["split"] == split) for split in ("train", "val", "test")
    }
    summary = {
        "h5_path": str(h5_path),
        "split_strategy": split_strategy,
        "block_size_chips": block_size_chips,
        "counts": counts,
        "fractions": {
            "train": train_frac,
            "val": val_frac,
            "test": test_frac,
        },
    }
    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return records


def _sample_band_pixels(
    image_chw: np.ndarray,
    rng: np.random.Generator,
    samples_per_band: int,
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for band_idx in range(image_chw.shape[0]):
        values = image_chw[band_idx].reshape(-1)
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            samples.append(np.zeros((0,), dtype=np.float32))
            continue
        take = min(samples_per_band, valid.size)
        choice = rng.choice(valid.size, size=take, replace=False)
        samples.append(valid[choice].astype(np.float32))
    return samples


def compute_h5_band_stats(
    h5_path: str | Path,
    manifest_csv: str | Path,
    out_npz: str | Path,
    out_summary_json: str | Path,
    max_stat_chips: int = 512,
    samples_per_band_per_chip: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    h5_path = Path(h5_path)
    manifest_csv = Path(manifest_csv)
    out_npz = Path(out_npz)
    out_summary_json = Path(out_summary_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    train_indices: list[int] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_indices.append(int(row["index"]))

    if not train_indices:
        raise ValueError("No training indices found in split manifest.")

    rng = np.random.default_rng(seed)
    if len(train_indices) > max_stat_chips:
        selected = sorted(rng.choice(train_indices, size=max_stat_chips, replace=False).tolist())
    else:
        selected = train_indices

    band_samples: list[list[np.ndarray]] | None = None
    with h5py.File(h5_path, "r") as h5:
        for idx in selected:
            image = h5["images"][idx].astype(np.float32)
            sampled = _sample_band_pixels(image, rng, samples_per_band_per_chip)
            if band_samples is None:
                band_samples = [[] for _ in range(len(sampled))]
            for band_idx, values in enumerate(sampled):
                band_samples[band_idx].append(values)

    if band_samples is None:
        raise ValueError("No band samples were collected.")

    means: list[float] = []
    stds: list[float] = []
    p2s: list[float] = []
    p98s: list[float] = []
    counts: list[int] = []
    for samples in band_samples:
        merged = np.concatenate(samples, axis=0) if samples else np.zeros((0,), dtype=np.float32)
        if merged.size == 0:
            means.append(0.0)
            stds.append(1.0)
            p2s.append(0.0)
            p98s.append(1.0)
            counts.append(0)
            continue
        means.append(float(np.mean(merged)))
        stds.append(float(np.std(merged) + 1e-6))
        p2 = float(np.percentile(merged, 2))
        p98 = float(np.percentile(merged, 98))
        if p98 <= p2:
            p98 = p2 + 1.0
        p2s.append(p2)
        p98s.append(p98)
        counts.append(int(merged.size))

    np.savez(
        out_npz,
        mean=np.asarray(means, dtype=np.float32),
        std=np.asarray(stds, dtype=np.float32),
        p2=np.asarray(p2s, dtype=np.float32),
        p98=np.asarray(p98s, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int64),
    )
    summary = {
        "h5_path": str(h5_path),
        "manifest_csv": str(manifest_csv),
        "selected_train_chips": len(selected),
        "max_stat_chips": max_stat_chips,
        "samples_per_band_per_chip": samples_per_band_per_chip,
        "counts_per_band": counts,
    }
    out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest(manifest_csv: str | Path) -> list[dict[str, Any]]:
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class BandStats:
    mean: np.ndarray
    std: np.ndarray
    p2: np.ndarray
    p98: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "BandStats":
        data = np.load(Path(path))
        return cls(
            mean=data["mean"].astype(np.float32),
            std=data["std"].astype(np.float32),
            p2=data["p2"].astype(np.float32),
            p98=data["p98"].astype(np.float32),
        )


def compute_band_ratios(images_hwc: np.ndarray) -> np.ndarray:
    """Compute NDVI, NDBI and NDWI ratio channels from raw S2 reflectance.

    Band order (0-indexed): B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
                             0  1  2  3  4  5  6  7  8   9  10  11  12

    NDVI = (B8 − B4) / (B8 + B4)   low for panels (no vegetation)
    NDBI = (B11 − B8) / (B11 + B8) elevated for built-up / panels
    NDWI = (B3 − B8) / (B3 + B8)   negative for panels (not water)

    All three indices are naturally in [−1, 1].  Input must be the *raw*
    (pre-normalisation) float32 array of shape (N, H, W, 13).
    """
    eps = 1e-6
    b3  = images_hwc[..., 2].astype(np.float32)
    b4  = images_hwc[..., 3].astype(np.float32)
    b8  = images_hwc[..., 7].astype(np.float32)
    b11 = images_hwc[..., 11].astype(np.float32)

    ndvi = (b8  - b4)  / (b8  + b4  + eps)
    ndbi = (b11 - b8)  / (b11 + b8  + eps)
    ndwi = (b3  - b8)  / (b3  + b8  + eps)

    ratios = np.stack([ndvi, ndbi, ndwi], axis=-1)   # (N, H, W, 3)
    return np.clip(ratios, -1.0, 1.0).astype(np.float32)


def normalize_batch(
    images_hwc: np.ndarray,
    band_stats: BandStats,
    normalization: str,
) -> np.ndarray:
    if normalization == "zscore":
        return (images_hwc - band_stats.mean[None, None, None, :]) / band_stats.std[None, None, None, :]
    denom = np.maximum(band_stats.p98 - band_stats.p2, 1.0)
    scaled = (images_hwc - band_stats.p2[None, None, None, :]) / denom[None, None, None, :]
    return np.clip(scaled, 0.0, 1.0)


def augment_pair_numpy(
    image_hwc: np.ndarray,
    mask_hw: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image_hwc = np.flip(image_hwc, axis=1)
        mask_hw = np.flip(mask_hw, axis=1)
    if rng.random() < 0.5:
        image_hwc = np.flip(image_hwc, axis=0)
        mask_hw = np.flip(mask_hw, axis=0)
    k = int(rng.integers(0, 4))
    if k:
        image_hwc = np.rot90(image_hwc, k=k, axes=(0, 1))
        mask_hw = np.rot90(mask_hw, k=k, axes=(0, 1))
    return image_hwc.copy(), mask_hw.copy()


def load_h5_split_arrays(
    h5_path: str | Path,
    records: list[dict[str, Any]],
    band_stats: BandStats,
    normalization: str = "percentile",
    image_dtype: str = "float32",
    band_indices: list[int] | None = None,
    band_ratios: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if image_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported image_dtype: {image_dtype}")

    if not records:
        raise ValueError("Cannot build a dataset from an empty split.")

    batch_indices = np.asarray([int(record["index"]) for record in records], dtype=np.int64)
    order = np.argsort(batch_indices)
    restore_order = np.argsort(order)
    sorted_indices = batch_indices[order]

    with h5py.File(Path(h5_path), "r") as h5:
        images_chw = h5["images"][sorted_indices].astype(np.float32)[restore_order]
        masks_hw = h5["masks"][sorted_indices].astype(np.float32)[restore_order]

    images_hwc = np.transpose(images_chw, (0, 2, 3, 1))

    # Compute ratio channels from raw reflectance before any normalisation.
    if band_ratios:
        ratios = compute_band_ratios(images_hwc)  # (N, H, W, 3), in [-1, 1]

    if band_indices is not None:
        idx = np.asarray(band_indices, dtype=np.int32)
        images_hwc = images_hwc[:, :, :, idx]
        active_stats = BandStats(
            mean=band_stats.mean[idx],
            std=band_stats.std[idx],
            p2=band_stats.p2[idx],
            p98=band_stats.p98[idx],
        )
    else:
        active_stats = band_stats
    images_hwc = normalize_batch(images_hwc, band_stats=active_stats, normalization=normalization)

    if band_ratios:
        images_hwc = np.concatenate([images_hwc, ratios], axis=-1)

    images_hwc = images_hwc.astype(np.float16 if image_dtype == "float16" else np.float32, copy=False)
    masks_hwc = masks_hw[..., np.newaxis].astype(np.float32, copy=False)
    return images_hwc, masks_hwc


def _augment_example_tf(
    image: tf.Tensor,
    mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    image = tf.image.random_flip_up_down(image)
    mask = tf.image.random_flip_up_down(mask)
    k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    return image, mask


def _augment_example_spectral_tf(
    image: tf.Tensor,
    mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Spatial augmentation (flips + rotations) plus per-band brightness jitter ±20%.

    No hard clip is applied so this works correctly with both minmax-normalised
    data (values in [0, 1]) and z-score-normalised data (values roughly in
    [-3, 3]).  The scale factor is multiplicative which preserves zero-mean
    structure for z-scored inputs.
    """
    image, mask = _augment_example_tf(image, mask)
    n_bands = tf.shape(image)[-1]
    scale = tf.random.uniform([1, 1, n_bands], minval=0.8, maxval=1.2, dtype=tf.float32)
    image = tf.cast(tf.cast(image, tf.float32) * scale, image.dtype)
    return image, mask


def _augment_example_enhanced_tf(
    image: tf.Tensor,
    mask: tf.Tensor,
    augment_noise: bool = False,
    augment_jitter: bool = False,
    augment_scale: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Spatial augmentation plus optional radiometric and scale perturbations.

    augment_noise:  per-pixel Gaussian noise (σ=0.05, additive).  Simulates
                    sensor noise.  Safe with z-scored inputs.
    augment_jitter: per-band additive brightness offset (σ=0.10, same value
                    across all pixels of a band).  Simulates per-band
                    calibration drift / atmospheric correction uncertainty.
    augment_scale:  random crop + bilinear resize back to original size
                    (crop fraction uniform in [0.75, 1.0]).  Simulates
                    slight scale/resolution variation.
    """
    image, mask = _augment_example_tf(image, mask)
    img_f32 = tf.cast(image, tf.float32)

    if augment_noise:
        noise = tf.random.normal(tf.shape(img_f32), mean=0.0, stddev=0.05)
        img_f32 = img_f32 + noise

    if augment_jitter:
        n_bands = tf.shape(img_f32)[-1]
        jitter = tf.random.normal([1, 1, n_bands], mean=0.0, stddev=0.10)
        img_f32 = img_f32 + jitter

    image = tf.cast(img_f32, image.dtype)

    if augment_scale:
        shape = tf.shape(image)
        h, w = shape[0], shape[1]
        scale = tf.random.uniform([], 0.75, 1.0)
        crop_h = tf.maximum(tf.cast(tf.cast(h, tf.float32) * scale, tf.int32), 1)
        crop_w = tf.maximum(tf.cast(tf.cast(w, tf.float32) * scale, tf.int32), 1)
        offset_h = tf.random.uniform([], 0, tf.maximum(h - crop_h + 1, 1), dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, tf.maximum(w - crop_w + 1, 1), dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)
        mask = tf.image.crop_to_bounding_box(mask, offset_h, offset_w, crop_h, crop_w)
        image = tf.cast(tf.image.resize(tf.cast(image, tf.float32), [h, w], method="bilinear"), image.dtype)
        mask = tf.image.resize(mask, [h, w], method="nearest")

    return image, mask


def build_in_memory_dataset(
    images_hwc: np.ndarray,
    masks_hwc: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int = 42,
    shuffle_buffer: int | None = None,
    num_parallel_calls: int | None = None,
    prefetch_buffer: int | None = None,
    drop_remainder: bool = False,
    augment_spectral: bool = False,
    augment_noise: bool = False,
    augment_jitter: bool = False,
    augment_scale: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images_hwc, masks_hwc))

    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

    if shuffle:
        buffer_size = shuffle_buffer or len(images_hwc)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    if augment:
        use_enhanced = augment_noise or augment_jitter or augment_scale
        if use_enhanced:
            import functools
            aug_fn = functools.partial(
                _augment_example_enhanced_tf,
                augment_noise=augment_noise,
                augment_jitter=augment_jitter,
                augment_scale=augment_scale,
            )
        elif augment_spectral:
            aug_fn = _augment_example_spectral_tf
        else:
            aug_fn = _augment_example_tf
        dataset = dataset.map(
            aug_fn,
            num_parallel_calls=tf.data.AUTOTUNE if num_parallel_calls is None else num_parallel_calls,
            deterministic=False,
        )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    prefetch = tf.data.AUTOTUNE if prefetch_buffer in (None, 0) else prefetch_buffer
    return dataset.prefetch(prefetch)


class H5SegmentationSequence(tf.keras.utils.Sequence):
    """Keras Sequence for Stage-1 HDF5 segmentation data."""

    def __init__(
        self,
        h5_path: str | Path,
        records: list[dict[str, Any]],
        band_stats: BandStats,
        batch_size: int = 16,
        shuffle: bool = True,
        augment: bool = False,
        normalization: str = "percentile",
        seed: int = 42,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.records = list(records)
        self.band_stats = band_stats
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.normalization = normalization
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.records))
        self._h5: h5py.File | None = None
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(math.ceil(len(self.records) / self.batch_size))

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _normalize(self, image_hwc: np.ndarray) -> np.ndarray:
        return normalize_batch(
            image_hwc[np.newaxis, ...],
            band_stats=self.band_stats,
            normalization=self.normalization,
        )[0]

    def _normalize_batch(self, images_hwc: np.ndarray) -> np.ndarray:
        return normalize_batch(images_hwc, band_stats=self.band_stats, normalization=self.normalization)

    def _augment_pair(self, image_hwc: np.ndarray, mask_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return augment_pair_numpy(image_hwc, mask_hw, rng=self.rng)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.records))
        batch_positions = self.indices[start:stop]
        batch_records = [self.records[pos] for pos in batch_positions]
        h5 = self._get_h5()
        batch_indices = np.asarray([int(record["index"]) for record in batch_records], dtype=np.int64)

        # h5py fancy indexing is most efficient and reliable with monotonically increasing indices.
        order = np.argsort(batch_indices)
        restore_order = np.argsort(order)
        sorted_indices = batch_indices[order]

        images_chw = h5["images"][sorted_indices].astype(np.float32)[restore_order]
        masks_hw = h5["masks"][sorted_indices].astype(np.float32)[restore_order]

        images_hwc = np.transpose(images_chw, (0, 2, 3, 1))
        images_hwc = self._normalize_batch(images_hwc).astype(np.float32, copy=False)
        masks_hwc = masks_hw[..., np.newaxis]

        if self.augment:
            aug_images: list[np.ndarray] = []
            aug_masks: list[np.ndarray] = []
            for image, mask in zip(images_hwc, masks_hw):
                image_aug, mask_aug = self._augment_pair(image, mask)
                aug_images.append(image_aug)
                aug_masks.append(mask_aug[..., np.newaxis])
            images_hwc = np.stack(aug_images, axis=0).astype(np.float32, copy=False)
            masks_hwc = np.stack(aug_masks, axis=0).astype(np.float32, copy=False)
        else:
            masks_hwc = masks_hwc.astype(np.float32, copy=False)

        return images_hwc, masks_hwc

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
