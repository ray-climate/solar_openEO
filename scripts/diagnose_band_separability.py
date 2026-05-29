#!/usr/bin/env python
"""Test whether the failing TP blobs are separable from cropland using the
FULL band set (not just NDVI).

For each failing site: recompute the biggest blob's full feature vector,
then (a) place each top feature relative to training pv vs not_pv ranges,
and (b) k-NN class vote in standardized full-feature space.

Conclusion logic:
  - blob lands in not_pv range across MOST top features + kNN votes not_pv
    => spectrally looks like cropland; more bands won't help (coverage problem)
  - some bands place it in pv range / kNN mixed
    => feature/weighting problem; better features could help
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
SITES = ["c-000314_r+001799", "c-000105_r+001859", "c+000355_r+002986"]


def main():
    import ee; ee.Initialize()
    import glob
    import geopandas as gpd
    from solar_fp_filter import inference as I, features as F

    EU = importlib.util.spec_from_file_location("eu", REPO / "scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    from openeo_udp.udf.solar_pv_inference import (
        _build_model, _load_weights_compat, _load_band_stats, _load_registry)
    reg = _load_registry(); m = _build_model(reg); _load_weights_compat(m, eu.POST_WEIGHTS)
    bs = _load_band_stats(REPO / reg["band_stats_local"]); thr = float(reg.get("threshold", 0.85))
    spec2 = importlib.util.spec_from_file_location("rw", REPO / "scripts/19_test_real_world_sites.py")
    rw = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cols = json.loads(I.DEFAULT_COLS.read_text())

    # Training feature table
    ts_tr = pd.concat([pd.read_parquet(f) for f in glob.glob(str(REPO / "outputs/fp_classifier/timeseries/*.parquet"))], ignore_index=True)
    poly = gpd.read_file(REPO / "outputs/fp_classifier/polygons.gpkg")[["sample_id", "class", "lat", "area_m2"]]
    tr_feats = F.compute_features(ts_tr, polygons=poly)
    tr_feats = tr_feats[tr_feats["class"].notna()]
    pv = tr_feats[tr_feats["class"].isin(["pv", "becoming_pv"])]
    npv = tr_feats[tr_feats["class"] == "not_pv"]

    # Standardize for kNN
    mu = tr_feats[cols].mean(); sd = tr_feats[cols].std().replace(0, 1)
    Xtr = ((tr_feats[cols] - mu) / sd).to_numpy()
    ytr = (tr_feats["class"] != "not_pv").astype(int).to_numpy()

    all_sites = eu.select_sites(100)
    TOP = ["full_ndbi_mean", "late_ndbi_mean", "full_B12_mean", "late_B12_mean",
           "full_B8_mean", "full_ndwi_mean", "full_ndvi_mean", "full_vis_mean", "abs_lat"]
    for CID in SITES:
        site = [s for s in all_sites if s["chip_id"] == CID][0]
        spectral, scl, _ = rw.load_stack(REPO / f"docs/europe_unseen_2026MarMay/stacks/{CID}_stack.nc")
        comp, info = create_temporal_mosaic(spectral, scl)
        mask, _ = eu.tiled_predict(m, np.transpose(comp[:13], (1, 2, 0)), bs, thr)
        bbox = eu.aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
        x_min, y_min = tr.transform(bbox["west"], bbox["south"])
        x_max, y_max = tr.transform(bbox["east"], bbox["north"])
        blobs = I.vectorize_mask(mask, x_min, y_max, x_max - x_min, y_max - y_min, crs_epsg=3857)
        big = blobs.sort_values("n_pixels").iloc[[-1]].copy()
        big_id = int(big.iloc[0]["blob_id"])
        bts = I._gee_timeseries(big, "2026-06-01", 12)
        bf = F.compute_features(bts)
        bf["abs_lat"] = abs(site["lat"])
        bf["log_area"] = np.log10(max(big.to_crs("EPSG:3857").geometry.area.iloc[0], 1))
        for c in cols:
            if c not in bf.columns:
                bf[c] = np.nan

        print("\n" + "=" * 78)
        print(f"SITE {CID}  lat={site['lat']:.1f}  biggest blob id={big_id} "
              f"({int(big.iloc[0]['n_pixels'])} px)")
        print(f"{'feature':>16} | {'blob':>9} | {'pv 25-75':>17} | {'not_pv 25-75':>17} | side")
        for f in TOP:
            v = float(bf.iloc[0][f])
            pql, pqh = pv[f].quantile(.25), pv[f].quantile(.75)
            nql, nqh = npv[f].quantile(.25), npv[f].quantile(.75)
            # which class range is the blob value closer to?
            d_pv = 0 if pql <= v <= pqh else min(abs(v - pql), abs(v - pqh))
            d_np = 0 if nql <= v <= nqh else min(abs(v - nql), abs(v - nqh))
            side = "PV" if d_pv < d_np else ("not_pv" if d_np < d_pv else "tie")
            print(f"{f:>16} | {v:9.3f} | [{pql:7.3f},{pqh:7.3f}] | [{nql:7.3f},{nqh:7.3f}] | {side}")

        # kNN vote (k=25) in standardized full-feature space
        xb = ((bf[cols] - mu) / sd).to_numpy()[0]
        d = np.sqrt(np.nansum((Xtr - xb) ** 2, axis=1))
        knn = np.argsort(d)[:25]
        frac_pv = ytr[knn].mean()
        print(f"  kNN(25) in FULL feature space: {frac_pv*100:.0f}% PV neighbours, "
              f"{(1-frac_pv)*100:.0f}% not_pv")


if __name__ == "__main__":
    main()
