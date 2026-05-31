"""Phase 4 — runtime FP filter.

Takes a U-Net detection mask (+ its geo-referencing) and removes detection
blobs whose temporal signature doesn't look like PV.

Pipeline per call:
  1. Vectorize the binary mask into connected-component polygons.
  2. For each polygon, pull the most recent ``WINDOW_MONTHS`` of monthly
     S2 reflectance via GEE reduceRegions (same backend as training).
  3. Compute the SAME features as training, classify with the LightGBM
     model, keep blobs where P(pv)+P(becoming_pv) >= threshold.
  4. Return the filtered mask + a per-blob verdict table.

The model + feature columns are loaded from outputs/fp_classifier/.
Designed so the same ``classify_polygons`` is reused by the Apex UDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solar_fp_filter import features as F

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/fp_classifier"
DEFAULT_MODEL = OUT / "model.lgb"
DEFAULT_COLS = OUT / "feature_cols.json"

CLASSES = ["not_pv", "becoming_pv", "pv"]
GEE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Keep a detection if P(pv)+P(becoming_pv) >= KEEP_THRESHOLD.
# Set to 0.20 (TP-protective) per the user's priority: remove some false
# positives while keeping true positives least affected. On the held-out
# test set 0.20 removes ~82% of FPs while retaining ~95% of true PV (~95%
# of new builds). The score distribution is bimodal, so this low threshold
# still rejects the bulk of FPs (which cluster near 0) at little TP cost.
# Raise toward 0.5 for more aggressive FP removal at the expense of recall.
KEEP_THRESHOLD = 0.20

# Abstain-on-large safeguard: optionally never DELETE a contiguous detection
# bigger than this. DISABLED by default (None) — the 88-site analysis showed a
# flat size-abstain protects ~65% of false-positive pixel area (FP and TP blob
# sizes fully overlap, so size alone can't separate them). The shipped filter
# is the plain mean classifier at KEEP_THRESHOLD; abstain remains available as
# an opt-in param for callers who want it.
ABSTAIN_ABOVE_PX = None


# --------------------------------------------------------------------------
# Mask -> polygons
# --------------------------------------------------------------------------

def vectorize_mask(mask: np.ndarray,
                   x_min: float, y_max: float,
                   x_extent: float, y_extent: float,
                   crs_epsg: int = 3857,
                   min_pixels: int = 10):
    """Convert a binary mask to connected-component polygons.

    The mask's pixel grid maps to a world bbox: pixel (0,0) is the top-left
    at (x_min, y_max); the full grid spans x_extent x y_extent in CRS units.
    Returns a GeoDataFrame (EPSG:4326) with columns: blob_id, n_pixels,
    pixel_geom_idx (so we can map verdicts back to the mask), geometry.
    """
    import geopandas as gpd
    from rasterio import features as rfeat
    from rasterio.transform import from_bounds
    from shapely.geometry import shape

    h, w = mask.shape
    transform = from_bounds(x_min, y_max - y_extent, x_min + x_extent, y_max, w, h)
    polys, ids, npix = [], [], []
    bid = 0
    for geom, val in rfeat.shapes(mask.astype(np.int16), mask=(mask > 0),
                                  transform=transform):
        if val == 0:
            continue
        shp = shape(geom)
        area_px = shp.area / ((x_extent / w) * (y_extent / h))
        if area_px < min_pixels:
            continue
        polys.append(shp); ids.append(bid); npix.append(int(round(area_px)))
        bid += 1
    if not polys:
        return gpd.GeoDataFrame(
            {"blob_id": [], "n_pixels": []}, geometry=[], crs=f"EPSG:{crs_epsg}"
        ).to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame({"blob_id": ids, "n_pixels": npix},
                           geometry=polys, crs=f"EPSG:{crs_epsg}")
    return gdf.to_crs("EPSG:4326")


# --------------------------------------------------------------------------
# Polygons -> GEE time series -> features -> classify
# --------------------------------------------------------------------------

def _gee_timeseries(blobs_4326, end_date: str, window_months: int):
    """Monthly reduceRegions over the window_months ending at end_date."""
    import ee
    from solar_fp_filter import timeseries_gee as tg

    start = pd.Timestamp(end_date) - pd.DateOffset(months=window_months)
    start_s = start.strftime("%Y-%m-%d")
    feats = []
    for _, r in blobs_4326.iterrows():
        feats.append(ee.Feature(ee.Geometry(r.geometry.__geo_interface__),
                                {"sample_id": str(r.blob_id)}))
    fc = ee.FeatureCollection(feats)
    ic = tg._monthly_ic(fc, start_s, window_months)
    res = tg._reduce_collection(fc, ic)
    info = tg._getinfo_with_retry(res)
    return tg._parse_result(info)


def classify_polygons(blobs_4326, end_date: str,
                      model=None, cols=None,
                      threshold: float = KEEP_THRESHOLD,
                      window_months: int = F.WINDOW_MONTHS) -> pd.DataFrame:
    """Score each polygon. Returns a DataFrame with blob_id, p_keep,
    p_pv, p_becoming, p_notpv, keep (bool). Blobs with no usable time
    series are kept by default (fail-open: don't drop a detection just
    because GEE returned nothing)."""
    import lightgbm as lgb
    if model is None:
        model = lgb.Booster(model_file=str(DEFAULT_MODEL))
    if cols is None:
        cols = json.loads(DEFAULT_COLS.read_text())

    if len(blobs_4326) == 0:
        return pd.DataFrame(columns=["blob_id", "p_keep", "keep"])

    ts = _gee_timeseries(blobs_4326, end_date, window_months)
    out_rows = []
    if len(ts) == 0:
        for _, r in blobs_4326.iterrows():
            out_rows.append({"blob_id": r.blob_id, "p_keep": np.nan,
                             "p_pv": np.nan, "p_becoming": np.nan,
                             "p_notpv": np.nan, "keep": True})  # fail-open
        return pd.DataFrame(out_rows)

    # blob_id was passed as sample_id (string) into GEE
    feats = F.compute_features(ts)
    feats["blob_id"] = feats["sample_id"].astype(int)
    # metadata features the model expects but inference can't get from a blob:
    # abs_lat from centroid; log_area from polygon area.
    cent = blobs_4326.set_index("blob_id").geometry.centroid
    areas = blobs_4326.set_index("blob_id").to_crs("EPSG:3857").geometry.area
    feats["abs_lat"] = feats["blob_id"].map(cent.y.abs())
    feats["log_area"] = np.log10(feats["blob_id"].map(areas).clip(lower=1))

    for c in cols:
        if c not in feats.columns:
            feats[c] = np.nan
    proba = model.predict(feats[cols].to_numpy(dtype=float))
    feats["p_notpv"] = proba[:, CLASSES.index("not_pv")]
    feats["p_becoming"] = proba[:, CLASSES.index("becoming_pv")]
    feats["p_pv"] = proba[:, CLASSES.index("pv")]
    feats["p_keep"] = feats["p_pv"] + feats["p_becoming"]
    feats["keep"] = feats["p_keep"] >= threshold

    scored = set(feats["blob_id"])
    out = feats[["blob_id", "p_keep", "p_pv", "p_becoming", "p_notpv", "keep"]]
    # Fail-open for any blob GEE didn't return
    missing = [b for b in blobs_4326.blob_id if b not in scored]
    if missing:
        extra = pd.DataFrame({"blob_id": missing, "p_keep": np.nan,
                              "p_pv": np.nan, "p_becoming": np.nan,
                              "p_notpv": np.nan, "keep": True})
        out = pd.concat([out, extra], ignore_index=True)
    return out


def _pct_features_for_blobs(blobs_4326, end_date: str,
                            window_months: int = F.WINDOW_MONTHS) -> pd.DataFrame:
    """Per-pixel NDVI/NDBI percentile features per blob, via the scatter-proof
    per-feature extraction. Returns per-sample (blob_id) percentile features."""
    from solar_fp_filter import timeseries_gee as tg
    import ee
    feats = []
    for _, r in blobs_4326.iterrows():
        feats.append(ee.Feature(ee.Geometry(r.geometry.__geo_interface__),
                                 {"sample_id": str(r.blob_id)}))
    fc = ee.FeatureCollection(feats)
    eff_start = (pd.Timestamp(end_date) - pd.DateOffset(months=window_months)).strftime("%Y-%m-%d")
    res = tg._percentile_perfeature(fc, eff_start, window_months)
    ts_pct = tg._parse_pct_result(tg._getinfo_with_retry(res))
    if len(ts_pct) == 0:
        return pd.DataFrame(columns=["blob_id"])
    pf = F.compute_pct_features(ts_pct)
    pf["blob_id"] = pf["sample_id"].astype(int)
    return pf


def classify_polygons_pct(blobs_4326, end_date: str, model, cols,
                          threshold: float = KEEP_THRESHOLD,
                          window_months: int = F.WINDOW_MONTHS) -> pd.DataFrame:
    """Like classify_polygons but using MEAN + PERCENTILE features (idea #1)
    scored with the percentile model. Fail-open for blobs GEE can't score."""
    if len(blobs_4326) == 0:
        return pd.DataFrame(columns=["blob_id", "p_keep", "keep"])
    ts = _gee_timeseries(blobs_4326, end_date, window_months)
    cent = blobs_4326.set_index("blob_id").geometry.centroid
    areas = blobs_4326.set_index("blob_id").to_crs("EPSG:3857").geometry.area
    if len(ts) == 0:
        feats = pd.DataFrame({"blob_id": blobs_4326.blob_id})
    else:
        feats = F.compute_features(ts)
        feats["blob_id"] = feats["sample_id"].astype(int)
    feats["abs_lat"] = feats["blob_id"].map(cent.y.abs())
    feats["log_area"] = np.log10(feats["blob_id"].map(areas).clip(lower=1))
    # merge percentile features
    pf = _pct_features_for_blobs(blobs_4326, end_date, window_months)
    if len(pf):
        feats = feats.merge(pf.drop(columns=["sample_id"], errors="ignore"),
                            on="blob_id", how="left")
    for c in cols:
        if c not in feats.columns:
            feats[c] = np.nan
    out_rows = []
    scored = set()
    if len(feats):
        proba = model.predict(feats[cols].to_numpy(dtype=float))
        feats["p_notpv"] = proba[:, CLASSES.index("not_pv")]
        feats["p_becoming"] = proba[:, CLASSES.index("becoming_pv")]
        feats["p_pv"] = proba[:, CLASSES.index("pv")]
        feats["p_keep"] = feats["p_pv"] + feats["p_becoming"]
        feats["keep"] = feats["p_keep"] >= threshold
        scored = set(feats["blob_id"])
        out = feats[["blob_id", "p_keep", "p_pv", "p_becoming", "p_notpv", "keep"]]
    else:
        out = pd.DataFrame(columns=["blob_id", "p_keep", "p_pv", "p_becoming", "p_notpv", "keep"])
    missing = [b for b in blobs_4326.blob_id if b not in scored]
    if missing:
        extra = pd.DataFrame({"blob_id": missing, "p_keep": np.nan, "p_pv": np.nan,
                              "p_becoming": np.nan, "p_notpv": np.nan, "keep": True})
        out = pd.concat([out, extra], ignore_index=True)
    return out


def apply_filter(mask: np.ndarray, x_min, y_max, x_extent, y_extent,
                 end_date: str, crs_epsg: int = 3857,
                 model=None, cols=None, threshold: float = KEEP_THRESHOLD,
                 use_pct: bool = False, abstain_above_px: int = ABSTAIN_ABOVE_PX):
    """Full filter: mask -> filtered mask + verdict table.

    Returns (filtered_mask, blobs_gdf_with_verdicts).
    """
    import geopandas as gpd
    from rasterio import features as rfeat
    from rasterio.transform import from_bounds

    blobs = vectorize_mask(mask, x_min, y_max, x_extent, y_extent, crs_epsg)
    if len(blobs) == 0:
        return mask.copy(), blobs

    if use_pct:
        verdicts = classify_polygons_pct(blobs, end_date, model=model, cols=cols,
                                         threshold=threshold)
    else:
        verdicts = classify_polygons(blobs, end_date, model=model, cols=cols,
                                     threshold=threshold)
    blobs = blobs.merge(verdicts, on="blob_id", how="left")
    blobs["keep"] = blobs["keep"].fillna(True)
    # Abstain-on-large: never delete a big contiguous detection.
    if abstain_above_px is not None:
        big = blobs["n_pixels"] > abstain_above_px
        blobs.loc[big, "keep"] = True
        blobs.loc[big, "abstained"] = True

    # Burn kept blobs back into a mask
    h, w = mask.shape
    transform = from_bounds(x_min, y_max - y_extent, x_min + x_extent, y_max, w, h)
    keep_blobs = blobs[blobs["keep"]].to_crs(f"EPSG:{crs_epsg}")
    if len(keep_blobs) == 0:
        return np.zeros_like(mask), blobs
    shapes = [(geom, 1) for geom in keep_blobs.geometry]
    filtered = rfeat.rasterize(shapes, out_shape=(h, w), transform=transform,
                               fill=0, dtype=np.uint8)
    # Intersect with the original mask so we keep the exact original pixels
    filtered = (filtered.astype(bool) & (mask > 0)).astype(mask.dtype)
    return filtered, blobs
