"""4-panel comparison: RGB | GT mask | Best single model | Ensemble+TTA.

For each of N randomly selected test chips generates a 4-column figure row:
  col 0 – RGB true colour          (B4/B3/B2, percentile stretch)
  col 1 – ground-truth mask        (red = solar panel)
  col 2 – best single model mask   (exp_stage1_r101_dice_zscore_longer, t=0.5)
  col 3 – ensemble + TTA mask      (top-15 by val_dice, 8-fold TTA, t=0.5)

Usage
-----
  python scripts/10_compare_predictions.py \
      [--n       10]           # number of test chips
      [--seed    42]           # random seed for chip selection
      [--top     15]           # ensemble size (top-N by val_dice)
      [--threshold 0.5]        # binary threshold
      [--batch   8]            # inference batch size
      [--out     outputs/comparison_ensemble_tta.png]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from solar_ml.data import BandStats, compute_band_ratios, load_manifest, normalize_batch  # noqa: E402
from solar_ml.model import build_unet_model  # noqa: E402


# ---------------------------------------------------------------------------
# Weight loading (compatible with checkpoints saved under generic layer names)
# ---------------------------------------------------------------------------

def _h5_class_key(layer) -> str:
    """Convert the Python class name to the snake_case key used in H5."""
    name = type(layer).__name__
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return name.lower()


def _assign_from_group(layers, h5_group, class_ctr: dict | None = None) -> None:
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

        # Recurse into nested sub-models (encoder backbone)
        if "layers" in grp:
            _assign_from_group(layer.layers, grp["layers"])

        if layer.weights and "vars" in grp and len(grp["vars"]) > 0:
            n = len(grp["vars"])
            layer.set_weights([grp["vars"][str(i)][:] for i in range(n)])


def load_weights_compat(model, weights_path: Path) -> None:
    with h5py.File(str(weights_path), "r") as f:
        _assign_from_group(model.layers, f["layers"])


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def percentile_stretch(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(arr, p_lo), np.percentile(arr, p_hi)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def make_rgb(image_hwc_raw: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB from raw S2 reflectance (B4=idx3, B3=idx2, B2=idx1)."""
    rgb = np.stack([
        percentile_stretch(image_hwc_raw[:, :, 3]),
        percentile_stretch(image_hwc_raw[:, :, 2]),
        percentile_stretch(image_hwc_raw[:, :, 1]),
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)


def mask_overlay(rgb_u8: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    out = rgb_u8.astype(np.float32) / 255.0
    panel = mask_hw.astype(bool)
    blend = 0.45
    out[panel, 0] = out[panel, 0] * (1 - blend) + 1.0 * blend
    out[panel, 1] = out[panel, 1] * (1 - blend) + 0.15 * blend
    out[panel, 2] = out[panel, 2] * (1 - blend) + 0.15 * blend
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def prepare_inputs(
    images_hwc_raw: np.ndarray,
    band_stats: BandStats,
    normalization: str,
    band_indices: list[int] | None,
    band_ratios: bool,
) -> np.ndarray:
    """Normalise raw (N,H,W,13) chips for a specific model config."""
    imgs = images_hwc_raw.copy()

    if band_ratios:
        ratios = compute_band_ratios(imgs)

    if band_indices is not None:
        idx = np.asarray(band_indices, dtype=np.int32)
        imgs = imgs[:, :, :, idx]
        active = BandStats(
            mean=band_stats.mean[idx],
            std=band_stats.std[idx],
            p2=band_stats.p2[idx],
            p98=band_stats.p98[idx],
        )
    else:
        active = band_stats

    imgs = normalize_batch(imgs, active, normalization)

    if band_ratios:
        imgs = np.concatenate([imgs, ratios], axis=-1)

    return imgs.astype(np.float32)


def predict_batch(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        out.append(model.predict(images[i: i + batch_size], verbose=0))
    return np.concatenate(out, axis=0)   # (N, H, W, 1)


def predict_with_tta(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    """8-fold TTA: 4 rotations × 2 horizontal flips, outputs averaged."""
    accum: list[np.ndarray] = []
    for k in range(4):
        for flip in (False, True):
            aug = np.rot90(images, k=k, axes=(1, 2))
            if flip:
                aug = np.flip(aug, axis=2).copy()
            pred = predict_batch(model, aug, batch_size)
            if flip:
                pred = np.flip(pred, axis=2)
            pred = np.rot90(pred, k=(4 - k) % 4, axes=(1, 2))
            accum.append(pred)
    return np.mean(accum, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Experiment ranking
# ---------------------------------------------------------------------------

def rank_experiments(min_test_dice: float = 0.800) -> list[dict]:
    """Return experiments sorted by val_dice (descending), above threshold."""
    results = []
    for exp_dir in sorted((REPO / "experiments").iterdir()):
        mf = exp_dir / "metrics.json"
        if not mf.exists():
            continue
        try:
            d = json.loads(mf.read_text())
            val_dice  = d.get("best_val_dice", 0.0)
            test_dice = d.get("test_metrics", {}).get("dice_coefficient", 0.0)
            if test_dice >= min_test_dice:
                results.append({"exp_dir": exp_dir, "val_dice": val_dice, "test_dice": test_dice})
        except Exception:
            continue
    results.sort(key=lambda x: x["val_dice"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n",          type=int,   default=10,    help="Number of test chips")
    p.add_argument("--seed",       type=int,   default=42,    help="Random seed for chip selection")
    p.add_argument("--top",        type=int,   default=15,    help="Ensemble pool size (top-N by val_dice)")
    p.add_argument("--threshold",  type=float, default=0.5,   help="Binary threshold")
    p.add_argument("--batch",      type=int,   default=8,     help="Inference batch size")
    p.add_argument("--best-exp",   default="exp_stage1_r101_dice_zscore_longer",
                   help="Experiment name for the 'best single model' column")
    p.add_argument("--out", default="outputs/comparison_ensemble_tta.png", help="Output PNG path")
    return p.parse_args()


def load_config_for_exp(exp_dir: Path) -> dict:
    import yaml
    with open(exp_dir / "config.yaml") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    # ------------------------------------------------------------------ #
    # Select test chips                                                    #
    # ------------------------------------------------------------------ #
    # Use the best-model config for data paths (all experiments share them)
    best_exp_dir  = REPO / "experiments" / args.best_exp
    best_cfg      = load_config_for_exp(best_exp_dir)

    h5_path       = REPO / best_cfg["data"]["h5_path"]
    manifest_path = REPO / best_cfg["data"]["manifest_path"]
    stats_path    = REPO / best_cfg["data"]["stats_path"]

    all_records  = load_manifest(manifest_path)
    test_records = [r for r in all_records if r["split"] == "test"]
    print(f"[info] Test-split size: {len(test_records)}")

    rng     = np.random.default_rng(args.seed)
    chosen  = rng.choice(len(test_records), size=min(args.n, len(test_records)), replace=False)
    samples = [test_records[i] for i in chosen]
    n       = len(samples)

    # ------------------------------------------------------------------ #
    # Load raw images + masks for the selected chips                       #
    # ------------------------------------------------------------------ #
    h5_indices = np.asarray([int(r["index"]) for r in samples], dtype=np.int64)
    sort_order    = np.argsort(h5_indices)
    restore_order = np.argsort(sort_order)
    sorted_indices = h5_indices[sort_order]

    with h5py.File(h5_path, "r") as h5:
        images_chw = h5["images"][sorted_indices].astype(np.float32)[restore_order]
        masks_hw   = h5["masks"][sorted_indices].astype(np.float32)[restore_order]

    images_hwc_raw = np.transpose(images_chw, (0, 2, 3, 1))   # (N,256,256,13)
    band_stats_base = BandStats.from_npz(stats_path)

    # ------------------------------------------------------------------ #
    # Best single model predictions                                        #
    # ------------------------------------------------------------------ #
    print(f"\n[step 1/2] Best single model: {args.best_exp}")
    bcfg = best_cfg
    best_imgs = prepare_inputs(
        images_hwc_raw,
        band_stats_base,
        normalization  = bcfg["data"].get("normalization", "zscore"),
        band_indices   = [int(i) for i in bcfg["data"]["band_indices"]] if bcfg["data"].get("band_indices") else None,
        band_ratios    = bool(bcfg["data"].get("band_ratios", False)),
    )
    n_bands_best = best_imgs.shape[-1]
    best_model, _ = build_unet_model(
        input_shape     = (256, 256, n_bands_best),
        backbone_name   = bcfg["model"]["backbone"],
        pretrained      = False,
        decoder_filters = tuple(bcfg["model"].get("decoder_filters", [256, 128, 96, 64, 32])),
        decoder_dropout = float(bcfg["model"].get("decoder_dropout", 0.1)),
        attention       = bool(bcfg["model"].get("attention", False)),
        se              = bool(bcfg["model"].get("se", False)),
    )
    load_weights_compat(best_model, best_exp_dir / "best.weights.h5")
    best_preds = predict_batch(best_model, best_imgs, args.batch)   # (N,H,W,1)
    tf.keras.backend.clear_session()
    print(f"  [done] best model predictions shape: {best_preds.shape}")

    # ------------------------------------------------------------------ #
    # Ensemble + TTA predictions                                           #
    # ------------------------------------------------------------------ #
    print(f"\n[step 2/2] Ensemble top-{args.top} + 8-fold TTA …")
    pool = rank_experiments(min_test_dice=0.800)[: args.top]
    print(f"  Pool: {len(pool)} models")
    for i, e in enumerate(pool):
        print(f"    {i+1:2d}. val={e['val_dice']:.4f}  test={e['test_dice']:.4f}  {e['exp_dir'].name}")

    ens_accum = np.zeros((n, 256, 256, 1), dtype=np.float64)

    for idx, entry in enumerate(pool):
        exp_dir = entry["exp_dir"]
        print(f"\n  [{idx+1}/{len(pool)}] {exp_dir.name}")
        try:
            cfg = load_config_for_exp(exp_dir)
        except Exception as exc:
            print(f"    SKIP – config error: {exc}")
            continue

        # Load stats (may differ per experiment if split manifest differs)
        try:
            sp   = REPO / cfg["data"]["stats_path"]
            bs   = BandStats.from_npz(sp)
        except Exception:
            bs = band_stats_base

        bi   = [int(i) for i in cfg["data"]["band_indices"]] if cfg["data"].get("band_indices") else None
        br   = bool(cfg["data"].get("band_ratios", False))
        norm = cfg["data"].get("normalization", "zscore")

        imgs = prepare_inputs(images_hwc_raw, bs, normalization=norm, band_indices=bi, band_ratios=br)
        nb   = imgs.shape[-1]

        try:
            model, _ = build_unet_model(
                input_shape     = (256, 256, nb),
                backbone_name   = cfg["model"]["backbone"],
                pretrained      = False,
                decoder_filters = tuple(cfg["model"].get("decoder_filters", [256, 128, 96, 64, 32])),
                decoder_dropout = float(cfg["model"].get("decoder_dropout", 0.1)),
                attention       = bool(cfg["model"].get("attention", False)),
                se              = bool(cfg["model"].get("se", False)),
            )
            load_weights_compat(model, exp_dir / "best.weights.h5")
        except Exception as exc:
            print(f"    SKIP – load error: {exc}")
            tf.keras.backend.clear_session()
            continue

        preds = predict_with_tta(model, imgs, args.batch)
        ens_accum += preds.astype(np.float64)
        tf.keras.backend.clear_session()
        print(f"    chip-mean sigmoid: {preds.mean():.4f}")

    ens_prob   = (ens_accum / len(pool)).astype(np.float32)
    print(f"\n  Ensemble probability range: [{ens_prob.min():.3f}, {ens_prob.max():.3f}]")

    # ------------------------------------------------------------------ #
    # Binary masks                                                         #
    # ------------------------------------------------------------------ #
    t = args.threshold
    best_binary = (best_preds[..., 0] >= t).astype(np.uint8)
    ens_binary  = (ens_prob[..., 0]   >= t).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    col_titles = [
        "RGB (true colour)",
        "Ground-truth mask",
        f"Best single model\n({args.best_exp})",
        f"Ensemble top-{args.top} + 8×TTA\n(≥{t})",
    ]

    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8.5, fontweight="bold")

    for row_idx, rec in enumerate(samples):
        chip_id   = rec["chip_id"]
        raw_hwc   = images_hwc_raw[row_idx]
        gt_mask   = masks_hw[row_idx]
        bst_mask  = best_binary[row_idx]
        ens_mask  = ens_binary[row_idx]

        rgb_u8    = make_rgb(raw_hwc)
        gt_vis    = mask_overlay(rgb_u8, gt_mask)
        bst_vis   = mask_overlay(rgb_u8, bst_mask)
        ens_vis   = mask_overlay(rgb_u8, ens_mask)

        ax0, ax1, ax2, ax3 = axes[row_idx]

        ax0.imshow(rgb_u8)
        ax1.imshow(gt_vis)
        ax2.imshow(bst_vis)
        ax3.imshow(ens_vis)

        ax0.set_ylabel(chip_id, fontsize=6.5, rotation=0, labelpad=85, va="center")
        ax1.set_xlabel(f"panel: {gt_mask.mean()*100:.1f}%",  fontsize=7)
        ax2.set_xlabel(f"panel: {bst_mask.mean()*100:.1f}%", fontsize=7)
        ax3.set_xlabel(f"panel: {ens_mask.mean()*100:.1f}%", fontsize=7)

        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Solar PV – single model vs ensemble+TTA  |  n={n} test chips  |  t={t}",
        fontsize=9.5,
        y=1.002,
    )
    plt.tight_layout()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] Saved → {out_path}")


if __name__ == "__main__":
    main()
