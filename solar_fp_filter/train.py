"""Phase 3 — train the LightGBM 3-class FP classifier.

Loads the extracted time-series parquets + polygon labels, computes
features, does a polygon-GROUPED stratified split (all windows of one
spatial polygon stay in one split), trains a 3-class LightGBM, and
writes the model + metrics + feature-importance plot.

Decision rule at inference: keep a detection if
    P(pv) + P(becoming_pv) >= threshold
so both fully-built and newly-built PV count as real.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solar_fp_filter import features as F

REPO = Path(__file__).resolve().parent.parent
TS_DIR = REPO / "outputs/fp_classifier/timeseries"
POLYGONS_GPKG = REPO / "outputs/fp_classifier/polygons.gpkg"
OUT_DIR = REPO / "outputs/fp_classifier"

CLASSES = ["not_pv", "becoming_pv", "pv"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def _load_features() -> pd.DataFrame:
    import geopandas as gpd
    files = sorted(glob.glob(str(TS_DIR / "*.parquet")))
    ts = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    poly = gpd.read_file(POLYGONS_GPKG)[
        ["sample_id", "polygon_id", "class", "negative_source",
         "lat", "area_m2", "window_name"]]
    feats = F.compute_features(ts, polygons=poly)
    feats = feats.merge(poly[["sample_id", "polygon_id"]], on="sample_id", how="left")
    feats = feats[feats["class"].notna()].reset_index(drop=True)
    return feats


def _grouped_split(feats: pd.DataFrame, seed: int = 42):
    """Split by polygon_id so all windows of one spatial polygon are in the
    same split. Stratify the group assignment by the polygon's dominant class
    and latitude band."""
    rng = np.random.default_rng(seed)
    grp = feats.groupby("polygon_id").agg(
        cls=("class", lambda s: s.mode().iloc[0]),
        lat=("lat", "mean")).reset_index()
    grp["lat_band"] = pd.cut(grp["lat"], bins=[-90, 0, 30, 45, 60, 90],
                             labels=False)
    grp["stratum"] = grp["cls"].astype(str) + "_" + grp["lat_band"].astype(str)
    train_ids, val_ids, test_ids = set(), set(), set()
    for _, sub in grp.groupby("stratum"):
        ids = sub["polygon_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids); n_tr = int(0.70 * n); n_va = int(0.15 * n)
        train_ids.update(ids[:n_tr])
        val_ids.update(ids[n_tr:n_tr + n_va])
        test_ids.update(ids[n_tr + n_va:])
    split = pd.Series("train", index=feats.index)
    split[feats["polygon_id"].isin(val_ids)] = "val"
    split[feats["polygon_id"].isin(test_ids)] = "test"
    return split


def train(seed: int = 42):
    import lightgbm as lgb
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading + featurizing ...", flush=True)
    feats = _load_features()
    print(f"Feature table: {feats.shape[0]:,} samples x {feats.shape[1]} cols", flush=True)

    # Exclude n_months entirely: after truncation it's ~constant, and any
    # residual variance (samples with <12 months due to cloud) would be a
    # data-quality signal, not a PV signal — keep it out of the model.
    cols = [c for c in F.feature_columns(feats) if c != "n_months"]
    feats["y"] = feats["class"].map(CLASS_TO_INT)
    feats["split"] = _grouped_split(feats, seed=seed)
    print("Split sizes:", feats["split"].value_counts().to_dict(), flush=True)
    print("Class balance (train):",
          feats[feats.split == "train"]["class"].value_counts().to_dict(), flush=True)

    def XY(s):
        d = feats[feats.split == s]
        return d[cols].to_numpy(dtype=float), d["y"].to_numpy(), d

    Xtr, ytr, _ = XY("train")
    Xva, yva, _ = XY("val")
    Xte, yte, dte = XY("test")

    # Class weights: pv:becoming:not_pv = 1:1:2 -> weight not_pv samples less,
    # actually we want balanced; use inverse-frequency * plan ratio.
    counts = np.bincount(ytr, minlength=3)
    base_w = counts.sum() / (3 * np.maximum(counts, 1))
    ratio = {CLASS_TO_INT["pv"]: 1.0, CLASS_TO_INT["becoming_pv"]: 1.0,
             CLASS_TO_INT["not_pv"]: 2.0}
    cls_w = {k: base_w[k] * ratio[k] for k in range(3)}
    w_tr = np.array([cls_w[y] for y in ytr])

    dtrain = lgb.Dataset(Xtr, label=ytr, weight=w_tr, feature_name=cols)
    dval = lgb.Dataset(Xva, label=yva, reference=dtrain, feature_name=cols)
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, max_depth=6,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                  seed=seed, verbose=-1)
    print("Training LightGBM ...", flush=True)
    model = lgb.train(params, dtrain, num_boost_round=800,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    # ---- Evaluate on held-out test ----
    proba = model.predict(Xte)
    pred = proba.argmax(1)
    # Binary "PV-or-becoming vs not_pv"
    p_pv = proba[:, CLASS_TO_INT["pv"]] + proba[:, CLASS_TO_INT["becoming_pv"]]
    is_pv_true = (yte != CLASS_TO_INT["not_pv"]).astype(int)
    from sklearn.metrics import roc_auc_score, confusion_matrix
    auc = roc_auc_score(is_pv_true, p_pv)

    # New-build robustness: of held-out becoming_pv, how many pass p>=0.5?
    bc_mask = yte == CLASS_TO_INT["becoming_pv"]
    newbuild_keep = float((p_pv[bc_mask] >= 0.5).mean()) if bc_mask.sum() else None

    cm = confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist()
    metrics = {
        "n_train": int((feats.split == "train").sum()),
        "n_val": int((feats.split == "val").sum()),
        "n_test": int(len(yte)),
        "best_iteration": int(model.best_iteration),
        "binary_auc_pv_vs_notpv": round(float(auc), 4),
        "newbuild_keep_rate": (round(newbuild_keep, 4) if newbuild_keep is not None else None),
        "confusion_matrix_rows_true_notpv_becoming_pv": cm,
        "classes": CLASSES,
    }
    print("\n=== TEST METRICS ===", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    model.save_model(str(OUT_DIR / "model.lgb"))
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "feature_cols.json").write_text(json.dumps(cols, indent=2))

    # Feature importance plot
    imp = pd.Series(model.feature_importance(importance_type="gain"),
                    index=cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 11))
    imp.tail(30).plot.barh(ax=ax)
    ax.set_title("FP classifier — top-30 feature importance (gain)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved model + metrics + feature_importance.png to {OUT_DIR}", flush=True)
    print("\nTop 12 features by gain:")
    print(imp.tail(12)[::-1].to_string(), flush=True)
    return metrics
