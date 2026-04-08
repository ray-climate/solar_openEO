"""Training utilities for solar panel segmentation from Sentinel-2 chips."""

from .config import load_config
from .data import (
    H5SegmentationSequence,
    build_in_memory_dataset,
    compute_h5_band_stats,
    create_split_manifest,
    load_h5_split_arrays,
)
from .losses import get_loss, dice_coefficient, iou_coefficient
from .model import build_unet_model

__all__ = [
    "H5SegmentationSequence",
    "build_in_memory_dataset",
    "build_unet_model",
    "compute_h5_band_stats",
    "create_split_manifest",
    "dice_coefficient",
    "get_loss",
    "iou_coefficient",
    "load_h5_split_arrays",
    "load_config",
]
