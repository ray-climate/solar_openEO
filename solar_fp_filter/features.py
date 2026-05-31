"""Phase 3 — feature engineering for the FP classifier.

Turns a long-format monthly time-series (one row per sample-month with
band means) into a wide per-sample feature table. Features are designed
around the new-build-friendly logic: heavy weight on the LATE window
(last portion of the trace) plus a late-vs-early step-change indicator,
so a site that only recently became PV still looks PV-like.

Reflectance is in DN (0-10000); we keep DN for brightness features and
use ratio indices (NDVI/NDWI/NDBI) which are scale-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
LATE_FRACTION = 0.5   # last 50% of a sample's months = "late window"

# At inference we always pull the most recent 12 months. Training windows
# differ in length by class (transition is 24mo, others 12mo), so we MUST
# truncate every sample to its last WINDOW_MONTHS to match inference and
# avoid window-length leakage (n_months would otherwise encode the class).
# For a `becoming_pv` (24mo, year-1..year+1) sample the last 12 months span
# build-year..year+1 — exactly the construction->PV signature a freshly
# built site shows at inference.
WINDOW_MONTHS = 12


def _indices(d: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    d = d.copy()
    d["ndvi"] = (d.B8 - d.B4) / (d.B8 + d.B4 + eps)
    d["ndwi"] = (d.B3 - d.B8) / (d.B3 + d.B8 + eps)
    d["ndbi"] = (d.B11 - d.B8) / (d.B11 + d.B8 + eps)
    d["vis"]  = (d.B2 + d.B3 + d.B4) / 3.0
    return d


def _agg_block(g: pd.DataFrame, prefix: str) -> dict:
    """Summary stats over a block of monthly rows."""
    out = {}
    for b in BANDS:
        v = g[b].to_numpy(dtype=float)
        out[f"{prefix}{b}_mean"] = np.nanmean(v)
        out[f"{prefix}{b}_std"]  = np.nanstd(v)
    for idx in ["ndvi", "ndwi", "ndbi"]:
        v = g[idx].to_numpy(dtype=float)
        out[f"{prefix}{idx}_mean"] = np.nanmean(v)
        out[f"{prefix}{idx}_std"]  = np.nanstd(v)
    vis = g["vis"].to_numpy(dtype=float)
    out[f"{prefix}vis_mean"] = np.nanmean(vis)
    return out


def _per_sample(g: pd.DataFrame) -> dict:
    g = g.sort_values("month")
    # Truncate to the most recent WINDOW_MONTHS so every sample spans the
    # same length as inference (see WINDOW_MONTHS note).
    if len(g) > WINDOW_MONTHS:
        g = g.iloc[-WINDOW_MONTHS:]
    n = len(g)
    # n_months kept for diagnostics only; excluded from the model features
    # (it is ~constant after truncation).
    feat = {"n_months": n}

    # Full-window block
    feat.update(_agg_block(g, "full_"))

    # NDVI seasonal amplitude + annual-frequency FFT amplitude
    ndvi = g["ndvi"].to_numpy(dtype=float)
    feat["full_ndvi_amp"] = np.nanmax(ndvi) - np.nanmin(ndvi)
    if n >= 4:
        x = ndvi - np.nanmean(ndvi)
        fft = np.abs(np.fft.rfft(np.nan_to_num(x)))
        feat["ndvi_fft_peak"] = float(fft[1:].max()) / n if len(fft) > 1 else 0.0
    else:
        feat["ndvi_fft_peak"] = 0.0

    # Late window (last LATE_FRACTION of months) — most important at inference
    k = max(1, int(round(n * LATE_FRACTION)))
    late = g.iloc[-k:]
    early = g.iloc[:max(1, n - k)]
    feat.update(_agg_block(late, "late_"))

    # Late-vs-early step change (construction signature)
    feat["step_ndvi"] = float(np.nanmean(late["ndvi"]) - np.nanmean(early["ndvi"]))
    feat["step_vis"]  = float(np.nanmean(late["vis"]) - np.nanmean(early["vis"]))
    feat["step_b11"]  = float(np.nanmean(late["B11"]) - np.nanmean(early["B11"]))
    return feat


def compute_features(ts: pd.DataFrame,
                     polygons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-format ts (sample_id, month, B2..B12) -> wide per-sample features.

    If ``polygons`` (with sample_id, lat, area_m2, class) is provided, the
    metadata columns and the label are merged in.
    """
    ts = _indices(ts)
    rows = []
    for sid, g in ts.groupby("sample_id", sort=False):
        f = _per_sample(g)
        f["sample_id"] = sid
        rows.append(f)
    feats = pd.DataFrame(rows)

    if polygons is not None:
        meta = polygons[["sample_id", "lat", "area_m2", "class"]].copy()
        feats = feats.merge(meta, on="sample_id", how="left")
        feats["abs_lat"] = feats["lat"].abs()
        feats["log_area"] = np.log10(feats["area_m2"].clip(lower=1))
    return feats


PCT_COLS = ["NDVI_p10", "NDVI_p25", "NDVI_p50", "NDVI_p75", "NDVI_p90",
            "NDBI_p10", "NDBI_p25", "NDBI_p50", "NDBI_p75", "NDBI_p90"]


def compute_pct_features(ts_pct: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel percentile time series -> wide per-sample percentile features.

    Captures the panel-pixel TAIL the whole-blob mean dilutes away: low-NDVI
    percentiles (p10/p25 = panel-dominated pixels) and high-NDBI percentiles
    (p75/p90 = built-up pixels). Same last-WINDOW_MONTHS truncation + late
    window as the mean features, so it matches inference.
    """
    rows = []
    for sid, g in ts_pct.groupby("sample_id", sort=False):
        g = g.sort_values("month")
        if len(g) > WINDOW_MONTHS:
            g = g.iloc[-WINDOW_MONTHS:]
        n = len(g)
        k = max(1, int(round(n * LATE_FRACTION)))
        late = g.iloc[-k:]
        f = {"sample_id": sid}
        for c in PCT_COLS:
            v = g[c].to_numpy(dtype=float)
            f[f"{c}_mean"] = np.nanmean(v)
            f[f"{c}_std"] = np.nanstd(v)
            f[f"{c}_late"] = np.nanmean(late[c].to_numpy(dtype=float))
        # spreads: within-footprint heterogeneity (panel+grass mix -> wide spread)
        f["NDVI_spread_mean"] = np.nanmean(
            (g["NDVI_p90"] - g["NDVI_p10"]).to_numpy(dtype=float))
        f["NDBI_spread_mean"] = np.nanmean(
            (g["NDBI_p90"] - g["NDBI_p10"]).to_numpy(dtype=float))
        rows.append(f)
    return pd.DataFrame(rows)


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Columns to feed the model: everything numeric except identifiers,
    label, split, and raw lat/area (we use abs_lat / log_area instead).

    NOTE: explicitly excludes `y` and `split` — if a caller adds the integer
    label `y` to the frame before calling this, it must NOT leak in as a
    feature (that yields a spurious AUC=1.0)."""
    drop = {"sample_id", "class", "lat", "area_m2", "y", "split",
            "polygon_id", "negative_source", "window_name", "year_built"}
    return [c for c in feats.columns
            if c not in drop and pd.api.types.is_numeric_dtype(feats[c])]
