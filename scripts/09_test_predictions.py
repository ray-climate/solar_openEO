"""Visualise model predictions on 10 unseen test chips.

For each chip produces a 3-panel figure:
  col 0 – true-colour RGB  (B4/B3/B2, percentile stretch)
  col 1 – ground-truth mask  (red = panel)
  col 2 – predicted mask     (red = panel, threshold 0.5)

Usage
-----
  python scripts/09_test_predictions.py \
      [--exp  exp_stage1_r101_dice_zscore_longer] \
      [--n    10] \
      [--seed 42] \
      [--threshold 0.5] \
      [--out  outputs/test_predictions.png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from solar_ml.data import BandStats, load_manifest, normalize_batch  # noqa: E402
from solar_ml.model import build_unet_model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_stretch(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Clip to [p_lo, p_hi] percentiles and scale to [0, 1]."""
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi <= lo:
        hi = lo + 1.0
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def make_rgb(image_hwc_raw: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB image from raw S2 reflectance.

    Band order: B1..B12 at indices 0..12.
    RGB = B4 (index 3), B3 (index 2), B2 (index 1).
    """
    r = percentile_stretch(image_hwc_raw[:, :, 3])
    g = percentile_stretch(image_hwc_raw[:, :, 2])
    b = percentile_stretch(image_hwc_raw[:, :, 1])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def mask_overlay(rgb_u8: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Blend a red tint onto pixels where mask == 1."""
    out = rgb_u8.astype(np.float32) / 255.0
    panel = mask_hw.astype(bool)
    blend = 0.45
    out[panel, 0] = out[panel, 0] * (1 - blend) + 1.0 * blend
    out[panel, 1] = out[panel, 1] * (1 - blend) + 0.15 * blend
    out[panel, 2] = out[panel, 2] * (1 - blend) + 0.15 * blend
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Compatibility weight loader
# ---------------------------------------------------------------------------

def _file_layer_key(base: str, counter: int) -> str:
    """Return the H5 layer key for the N-th layer of a given base type.

    Keras names the first layer 'conv2d' (no suffix), subsequent ones
    'conv2d_1', 'conv2d_2', etc.
    """
    return base if counter == 0 else f"{base}_{counter}"


def _get_file_vars(grp: "h5py.Group", key: str) -> list:
    """Return a list of numpy arrays for a layer in a .weights.h5 group."""
    if key not in grp:
        return []
    layer_grp = grp[key]
    if "vars" not in layer_grp:
        return []
    return [layer_grp["vars"][vi][:] for vi in sorted(layer_grp["vars"].keys(), key=int)]


def _collect_vars_by_build_order(
    model: "tf.keras.Model",
    h5_layers: "h5py.Group",
    *,
    sub_model_key: str = "functional",
) -> tuple[list, list]:
    """Return (model_vars, file_tensors) matched in build order.

    Handles two kinds of layers:
    - Direct top-level layers  (rgb_adapter → 'conv2d', decoder → 'conv2d_N', etc.)
    - One nested sub-model     (the encoder backbone → 'functional')

    For the nested sub-model we recurse using the same type-counter approach:
    iterate the sub-model's layers in build order and map each Conv2D/BN to
    its sequentially-numbered entry inside 'functional/layers/'.

    Parameters
    ----------
    model:          the tf.keras.Model (top-level)
    h5_layers:      the root 'layers' h5py.Group from the .weights.h5 file
    sub_model_key:  the key used for the nested encoder in the file ('functional')
    """
    import tensorflow as _tf

    # Layer-type counters for the top level and the nested sub-model level
    top_counters: dict[str, int] = {}
    sub_counters: dict[str, int] = {}

    model_vars: list = []
    file_tensors: list = []

    # Map tf.keras layer class → H5 base name
    _TYPE_MAP = {
        _tf.keras.layers.Conv2D: "conv2d",
        _tf.keras.layers.DepthwiseConv2D: "depthwise_conv2d",
        _tf.keras.layers.BatchNormalization: "batch_normalization",
        _tf.keras.layers.Dense: "dense",
    }

    def _add_layer_vars(layer, grp: "h5py.Group", counters: dict[str, int]) -> None:
        """Look up a single layer in 'grp' by type-counter, collect vars."""
        base = None
        for cls, name in _TYPE_MAP.items():
            if isinstance(layer, cls):
                base = name
                break
        if base is None:
            return  # Unknown type – skip (should have no trainable vars)

        idx = counters.get(base, 0)
        counters[base] = idx + 1
        key = _file_layer_key(base, idx)

        tensors = _get_file_vars(grp, key)
        if len(tensors) != len(layer.weights):
            raise ValueError(
                f"Var count mismatch for layer '{layer.name}' "
                f"(file key '{key}'): "
                f"model {len(layer.weights)} vs file {len(tensors)}"
            )
        for var, tensor in zip(layer.weights, tensors):
            if tuple(var.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch: model '{var.name}' {var.shape} "
                    f"vs file key '{key}' {tensor.shape}"
                )
            model_vars.append(var)
            file_tensors.append(tensor)

    for layer in model.layers:
        # Sub-model (encoder backbone) – recurse one level
        if hasattr(layer, "layers") and layer.name != model.name:
            if sub_model_key not in h5_layers:
                raise ValueError(
                    f"Expected sub-model key '{sub_model_key}' in checkpoint "
                    f"but it was not found."
                )
            sub_grp = h5_layers[sub_model_key]["layers"]
            for sub_layer in layer.layers:
                if sub_layer.weights:
                    _add_layer_vars(sub_layer, sub_grp, sub_counters)
        else:
            if layer.weights:
                _add_layer_vars(layer, h5_layers, top_counters)

    return model_vars, file_tensors


def _load_weights_compat(model: "tf.keras.Model", weights_path: str) -> None:
    """Load a Keras 3 .weights.h5 checkpoint into *model*.

    Handles the case where layers were renamed after the checkpoint was saved.
    Falls back to a type-counter–based positional load if the standard Keras
    by-name load fails.
    """
    try:
        model.load_weights(weights_path)
        return
    except (ValueError, Exception):
        pass  # fall through

    import h5py as _h5py
    import tensorflow as _tf

    with _h5py.File(weights_path, "r") as f:
        model_vars, file_tensors = _collect_vars_by_build_order(model, f["layers"])

    for var, tensor in zip(model_vars, file_tensors):
        var.assign(_tf.constant(tensor, dtype=var.dtype))

    n = len(model_vars)
    print(f"[info] Loaded {n} weight tensors via type-counter positional matching.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp",       default="exp_stage1_r101_dice_zscore_longer",
                   help="Experiment folder name under experiments/")
    p.add_argument("--n",         type=int, default=10, help="Number of test chips to visualise")
    p.add_argument("--seed",      type=int, default=42, help="Random seed for chip selection")
    p.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    p.add_argument("--out",       default="outputs/test_predictions.png",
                   help="Output figure path (PNG)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    exp_dir   = REPO_ROOT / "experiments" / args.exp
    weights   = exp_dir / "best.weights.h5"
    cfg_path  = exp_dir / "config.yaml"

    if not weights.exists():
        sys.exit(f"[ERROR] Weights not found: {weights}")

    # ------------------------------------------------------------------ #
    # Load config (minimal YAML parse – avoid full config loader)         #
    # ------------------------------------------------------------------ #
    import yaml
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    h5_path       = REPO_ROOT / data_cfg["h5_path"]
    manifest_path = REPO_ROOT / data_cfg["manifest_path"]
    stats_path    = REPO_ROOT / data_cfg["stats_path"]
    normalization = data_cfg.get("normalization", "zscore")
    backbone      = model_cfg.get("backbone", "resnet101")
    dec_filters   = tuple(model_cfg.get("decoder_filters", [256, 128, 96, 64, 32]))
    dec_dropout   = float(model_cfg.get("decoder_dropout", 0.1))
    attention     = bool(model_cfg.get("attention", False))
    se            = bool(model_cfg.get("se", False))
    band_indices  = data_cfg.get("band_indices", None)
    band_ratios   = bool(data_cfg.get("band_ratios", False))

    print(f"[info] Experiment : {args.exp}")
    print(f"[info] Backbone   : {backbone}")
    print(f"[info] Weights    : {weights}")

    # ------------------------------------------------------------------ #
    # Pick test-split indices                                              #
    # ------------------------------------------------------------------ #
    records   = load_manifest(manifest_path)
    test_recs = [r for r in records if r["split"] == "test"]
    print(f"[info] Test-split size: {len(test_recs)}")

    rng     = np.random.default_rng(args.seed)
    chosen  = rng.choice(len(test_recs), size=min(args.n, len(test_recs)), replace=False)
    samples = [test_recs[i] for i in chosen]

    # ------------------------------------------------------------------ #
    # Load raw images + masks                                              #
    # ------------------------------------------------------------------ #
    h5_indices = np.asarray([int(r["index"]) for r in samples], dtype=np.int64)
    with h5py.File(h5_path, "r") as h5:
        images_chw = h5["images"][np.sort(h5_indices)].astype(np.float32)
        masks_hw   = h5["masks"][np.sort(h5_indices)].astype(np.float32)
        # restore original order
        sort_order    = np.argsort(h5_indices)
        restore_order = np.argsort(sort_order)
        images_chw = images_chw[restore_order]
        masks_hw   = masks_hw[restore_order]

    images_hwc_raw = np.transpose(images_chw, (0, 2, 3, 1))   # (N, H, W, 13)

    # ------------------------------------------------------------------ #
    # Normalise for model input                                            #
    # ------------------------------------------------------------------ #
    band_stats  = BandStats.from_npz(stats_path)

    if band_indices is not None:
        idx          = np.asarray(band_indices, dtype=np.int32)
        images_model = images_hwc_raw[:, :, :, idx]
        active_stats = BandStats(
            mean=band_stats.mean[idx],
            std=band_stats.std[idx],
            p2=band_stats.p2[idx],
            p98=band_stats.p98[idx],
        )
    else:
        images_model = images_hwc_raw.copy()
        active_stats = band_stats

    images_norm = normalize_batch(images_model, active_stats, normalization)

    if band_ratios:
        from solar_ml.data import compute_band_ratios
        ratios      = compute_band_ratios(images_hwc_raw)
        images_norm = np.concatenate([images_norm, ratios], axis=-1)

    n_bands = images_norm.shape[-1]

    # ------------------------------------------------------------------ #
    # Build model and load weights                                         #
    # ------------------------------------------------------------------ #
    print(f"[info] Building model (input bands={n_bands}) …")
    model, _ = build_unet_model(
        input_shape   = (256, 256, n_bands),
        backbone_name = backbone,
        pretrained    = False,
        decoder_filters = dec_filters,
        decoder_dropout = dec_dropout,
        attention     = attention,
        se            = se,
    )
    # The weights file was saved when the rgb_adapter layer had its default
    # name ("conv2d").  Keras 3 .weights.h5 matches layers by name, so we
    # load the file manually and assign weights by matching (layer-name,
    # variable-index) pairs with a fallback to shape-based positional matching.
    _load_weights_compat(model, str(weights))
    print("[info] Weights loaded.")

    # ------------------------------------------------------------------ #
    # Run inference                                                        #
    # ------------------------------------------------------------------ #
    preds = model.predict(images_norm, batch_size=4, verbose=0)   # (N, H, W, 1)
    preds_binary = (preds[..., 0] >= args.threshold).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    n    = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB (true colour)", "Ground-truth mask", f"Predicted mask (≥{args.threshold})"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row_idx, rec in enumerate(samples):
        chip_id   = rec["chip_id"]
        raw_hwc   = images_hwc_raw[row_idx]
        gt_mask   = masks_hw[row_idx]
        pred_mask = preds_binary[row_idx]

        rgb_u8   = make_rgb(raw_hwc)
        gt_vis   = mask_overlay(rgb_u8, gt_mask)
        pred_vis = mask_overlay(rgb_u8, pred_mask)

        # panel coverage stats
        gt_frac   = gt_mask.mean() * 100
        pred_frac = pred_mask.mean() * 100

        ax0 = axes[row_idx, 0]
        ax1 = axes[row_idx, 1]
        ax2 = axes[row_idx, 2]

        ax0.imshow(rgb_u8)
        ax1.imshow(gt_vis)
        ax2.imshow(pred_vis)

        ax0.set_ylabel(chip_id, fontsize=7, rotation=0, labelpad=80, va="center")
        ax1.set_xlabel(f"panel: {gt_frac:.1f}%",   fontsize=7)
        ax2.set_xlabel(f"panel: {pred_frac:.1f}%", fontsize=7)

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Solar PV – test predictions  |  model: {args.exp}  |  n={n}",
        fontsize=10,
        y=1.002,
    )
    plt.tight_layout()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Saved → {out_path}")


if __name__ == "__main__":
    main()
