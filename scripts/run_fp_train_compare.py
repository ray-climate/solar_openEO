#!/usr/bin/env python
"""Train and compare two FP classifiers on the SAME polygon-grouped split:
  (A) mean-only features  (the current filter)
  (B) mean + percentile features  (idea #1)

Reports held-out test metrics for both so we can see whether the percentile
features improve accuracy. Also saves model B for the EU-unseen blob eval.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from solar_fp_filter import features as F
from solar_fp_filter import train as T

OUT = REPO / "outputs/fp_classifier"
PCT_DIRS = [OUT / "timeseries_pct", OUT / "timeseries_pct_gapfill"]


def _load_pct():
    frames = []
    for d in PCT_DIRS:
        for f in glob.glob(str(d / "*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                pass  # skip empty/bad parquet
    return pd.concat(frames, ignore_index=True)


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, confusion_matrix

    print("Loading mean features ...", flush=True)
    mean_feats = T._load_features()           # mean-based per-sample features + label + split inputs
    mean_feats["y"] = mean_feats["class"].map(T.CLASS_TO_INT)
    mean_feats["split"] = T._grouped_split(mean_feats, seed=42)

    print("Loading + featurizing percentiles ...", flush=True)
    pct = _load_pct()
    pct_feats = F.compute_pct_features(pct)
    print(f"  pct features: {pct_feats.shape[0]:,} samples x {pct_feats.shape[1]-1} cols", flush=True)

    merged = mean_feats.merge(pct_feats, on="sample_id", how="inner")
    print(f"Merged (samples with BOTH mean+pct): {len(merged):,}", flush=True)

    mean_cols = [c for c in F.feature_columns(mean_feats) if c != "n_months"]
    pct_cols = [c for c in pct_feats.columns if c != "sample_id"]
    print(f"  mean cols: {len(mean_cols)}, pct cols: {len(pct_cols)}", flush=True)

    def split_xy(df, cols, s):
        d = df[df.split == s]
        return d[cols].to_numpy(dtype=float), d["y"].to_numpy()

    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, max_depth=6,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                  seed=42, verbose=-1)
    counts = np.bincount(merged[merged.split == "train"]["y"], minlength=3)
    base_w = counts.sum() / (3 * np.maximum(counts, 1))
    ratio = {0: 2.0, 1: 1.0, 2: 1.0}   # not_pv:becoming:pv = 2:1:1
    cls_w = {k: base_w[k] * ratio[k] for k in range(3)}

    def train_eval(cols, label):
        Xtr, ytr = split_xy(merged, cols, "train")
        Xva, yva = split_xy(merged, cols, "val")
        Xte, yte = split_xy(merged, cols, "test")
        w = np.array([cls_w[y] for y in ytr])
        dtr = lgb.Dataset(Xtr, label=ytr, weight=w, feature_name=cols)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr, feature_name=cols)
        model = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                          valid_names=["val"], callbacks=[lgb.early_stopping(50, verbose=False)])
        proba = model.predict(Xte)
        p_keep = proba[:, 1] + proba[:, 2]
        is_pv = (yte != 0).astype(int)
        auc = roc_auc_score(is_pv, p_keep)
        # threshold sweep
        sweep = {}
        for th in [0.2, 0.3, 0.5]:
            neg = (yte == 0); pos = ~neg
            fp_removed = (p_keep[neg] < th).mean()
            tp_kept = (p_keep[pos] >= th).mean()
            nb = (yte == 1)
            nb_kept = (p_keep[nb] >= th).mean()
            sweep[th] = (fp_removed, tp_kept, nb_kept)
        print(f"\n=== {label} ===  (n_features={len(cols)})", flush=True)
        print(f"  binary AUC (PV+becoming vs not_pv): {auc:.4f}", flush=True)
        print(f"  {'thr':>5} {'FP_removed':>11} {'TP_kept':>9} {'newbuild_kept':>14}", flush=True)
        for th, (fr, tk, nk) in sweep.items():
            print(f"  {th:>5} {fr*100:>10.1f}% {tk*100:>8.1f}% {nk*100:>13.1f}%", flush=True)
        return model, auc

    m_mean, auc_mean = train_eval(mean_cols, "A: MEAN-ONLY (current)")
    m_both, auc_both = train_eval(mean_cols + pct_cols, "B: MEAN + PERCENTILE (idea #1)")

    print(f"\n>>> AUC: mean-only {auc_mean:.4f}  vs  mean+pct {auc_both:.4f}  "
          f"(delta {auc_both-auc_mean:+.4f})", flush=True)

    # Save model B + its feature columns for the blob eval
    m_both.save_model(str(OUT / "model_pct.lgb"))
    (OUT / "feature_cols_pct.json").write_text(json.dumps(mean_cols + pct_cols))
    print(f"Saved model_pct.lgb (+ feature_cols_pct.json)", flush=True)


if __name__ == "__main__":
    main()
