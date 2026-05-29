#!/usr/bin/env python
"""Detailed performance diagnostics for the trained FP classifier.

Reproduces the exact train/val/test split, re-scores the held-out test
set with the saved model, and produces a multi-panel figure + printed
stats focused on the question: how well does it remove false positives
while keeping real PV?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from solar_fp_filter import features as F
from solar_fp_filter import train as T

OUT = REPO / "outputs/fp_classifier"


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

    # Rebuild features + split exactly as training did
    feats = T._load_features()
    feats["y"] = feats["class"].map(T.CLASS_TO_INT)
    feats["split"] = T._grouped_split(feats, seed=42)
    cols = json.loads((OUT / "feature_cols.json").read_text())
    model = lgb.Booster(model_file=str(OUT / "model.lgb"))

    # attach negative_source for breakdown
    import geopandas as gpd
    src = gpd.read_file(T.POLYGONS_GPKG)[["sample_id", "negative_source", "window_name"]]
    feats = feats.merge(src, on="sample_id", how="left")

    te = feats[feats.split == "test"].copy()
    proba = model.predict(te[cols].to_numpy(dtype=float))
    te["p_notpv"] = proba[:, T.CLASS_TO_INT["not_pv"]]
    te["p_becoming"] = proba[:, T.CLASS_TO_INT["becoming_pv"]]
    te["p_pv"] = proba[:, T.CLASS_TO_INT["pv"]]
    te["p_keep"] = te["p_pv"] + te["p_becoming"]           # keep-score
    te["is_real_pv"] = (te["y"] != T.CLASS_TO_INT["not_pv"]).astype(int)

    y = te["is_real_pv"].to_numpy()
    s = te["p_keep"].to_numpy()
    auc = roc_auc_score(y, s)

    # ---- Threshold sweep: FP removal vs TP retention ----
    ths = np.linspace(0.05, 0.95, 19)
    rows = []
    neg = te[te.is_real_pv == 0]; pos = te[te.is_real_pv == 1]
    for th in ths:
        fp_removed = (neg["p_keep"] < th).mean()          # rejected negatives
        tp_kept = (pos["p_keep"] >= th).mean()             # kept positives
        nb = pos[pos.window_name == "transition"]
        nb_kept = (nb["p_keep"] >= th).mean() if len(nb) else np.nan
        rows.append((th, fp_removed, tp_kept, nb_kept))
    sweep = pd.DataFrame(rows, columns=["th", "fp_removed", "tp_kept", "newbuild_kept"])

    # ---- Figure ----
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))

    # (1) keep-score distributions
    ax[0, 0].hist(neg["p_keep"], bins=40, alpha=0.55, color="tab:red",
                  density=True, label="not_pv (FP)")
    ax[0, 0].hist(pos["p_keep"], bins=40, alpha=0.55, color="tab:green",
                  density=True, label="real PV (pv+becoming)")
    ax[0, 0].axvline(0.5, ls="--", c="k", lw=1)
    ax[0, 0].set_title("Keep-score  P(pv)+P(becoming_pv)")
    ax[0, 0].set_xlabel("keep-score"); ax[0, 0].legend()

    # (2) ROC
    fpr, tpr, _ = roc_curve(y, s)
    ax[0, 1].plot(fpr, tpr, color="tab:blue")
    ax[0, 1].plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax[0, 1].set_title(f"ROC (real PV vs not_pv)  AUC={auc:.3f}")
    ax[0, 1].set_xlabel("False positive rate"); ax[0, 1].set_ylabel("True positive rate")

    # (3) Precision-Recall
    prec, rec, _ = precision_recall_curve(y, s)
    ax[0, 2].plot(rec, prec, color="tab:purple")
    ax[0, 2].set_title("Precision-Recall (keep real PV)")
    ax[0, 2].set_xlabel("Recall (TP retention)"); ax[0, 2].set_ylabel("Precision")

    # (4) threshold tradeoff
    ax[1, 0].plot(sweep.th, sweep.fp_removed * 100, "-o", c="tab:red", label="FP removed %")
    ax[1, 0].plot(sweep.th, sweep.tp_kept * 100, "-o", c="tab:green", label="TP retained %")
    ax[1, 0].plot(sweep.th, sweep.newbuild_kept * 100, "-o", c="tab:olive", label="new-build kept %")
    ax[1, 0].axvline(0.5, ls="--", c="k", lw=1)
    ax[1, 0].set_title("Threshold tradeoff"); ax[1, 0].set_xlabel("keep threshold")
    ax[1, 0].set_ylabel("%"); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    # (5) FP removal by negative source
    ax[1, 1].set_title("FP removal by negative type (@0.5)")
    sub = neg.copy(); sub["src"] = sub["negative_source"].fillna("pre_build")
    g = sub.groupby("src").apply(lambda d: (d["p_keep"] < 0.5).mean() * 100)
    g.sort_values().plot.barh(ax=ax[1, 1], color="tab:red")
    ax[1, 1].set_xlabel("% correctly rejected"); ax[1, 1].set_xlim(0, 100)
    for i, v in enumerate(g.sort_values().values):
        ax[1, 1].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)

    # (6) TP retention by build recency (pv vs becoming)
    ax[1, 2].set_title("TP retention by type (@0.5)")
    gk = pos.groupby("window_name").apply(lambda d: (d["p_keep"] >= 0.5).mean() * 100)
    gk.sort_values().plot.barh(ax=ax[1, 2], color="tab:green")
    ax[1, 2].set_xlabel("% kept"); ax[1, 2].set_xlim(0, 100)
    for i, v in enumerate(gk.sort_values().values):
        ax[1, 2].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)

    plt.tight_layout()
    out_png = OUT / "classifier_performance.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_png}", flush=True)

    # ---- Printed stats ----
    print("\n=== Threshold sweep (test set) ===")
    print(sweep.assign(fp_removed=(sweep.fp_removed*100).round(1),
                       tp_kept=(sweep.tp_kept*100).round(1),
                       newbuild_kept=(sweep.newbuild_kept*100).round(1)).to_string(index=False))
    print("\n=== FP removal by negative type @0.5 ===")
    print(g.round(1).to_string())
    print("\n=== TP retention by type @0.5 ===")
    print(gk.round(1).to_string())


if __name__ == "__main__":
    main()
