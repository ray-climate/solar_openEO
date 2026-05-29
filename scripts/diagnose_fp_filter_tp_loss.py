#!/usr/bin/env python
"""Diagnose why genuine PV plants get filtered out.

For each flagged site: run the U-Net, vectorize blobs, pull GEE time
series, classify, and dump per-blob probabilities + the monthly trace of
the biggest blob. Also compares the blob's mean signature against a
TIGHT (eroded) core to test the dilution hypothesis.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

SITES = ["c-000314_r+001799", "c-000105_r+001859", "c+000355_r+002986"]


def main():
    import ee; ee.Initialize()
    import lightgbm as lgb

    EU = importlib.util.spec_from_file_location("eu", REPO / "scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    from solar_fp_filter import inference as I, features as F, timeseries_gee as tg

    from openeo_udp.udf.solar_pv_inference import (
        _build_model, _load_weights_compat, _load_band_stats, _load_registry)
    reg = _load_registry(); m = _build_model(reg); _load_weights_compat(m, eu.POST_WEIGHTS)
    bs = _load_band_stats(REPO / reg["band_stats_local"]); thr = float(reg.get("threshold", 0.85))
    spec2 = importlib.util.spec_from_file_location("rw", REPO / "scripts/19_test_real_world_sites.py")
    rw = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    fp_model = lgb.Booster(model_file=str(I.DEFAULT_MODEL)); cols = json.loads(I.DEFAULT_COLS.read_text())

    all_sites = eu.select_sites(100)
    for CID in SITES:
        site = [s for s in all_sites if s["chip_id"] == CID][0]
        print("\n" + "=" * 78)
        print(f"SITE {CID}  tier={site['tier']} area={site['area_m2']/1e6:.2f}km2 "
              f"lat={site['lat']:.2f} lon={site['lon']:.2f}")
        spectral, scl, _ = rw.load_stack(REPO / f"docs/europe_unseen_2026MarMay/stacks/{CID}_stack.nc")
        comp, info = create_temporal_mosaic(spectral, scl)
        mask, _ = eu.tiled_predict(m, np.transpose(comp[:13], (1, 2, 0)), bs, thr)
        bbox = eu.aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
        x_min, y_min = tr.transform(bbox["west"], bbox["south"])
        x_max, y_max = tr.transform(bbox["east"], bbox["north"])
        blobs = I.vectorize_mask(mask, x_min, y_max, x_max - x_min, y_max - y_min, crs_epsg=3857)
        print(f"  {len(blobs)} blobs, px sizes (top5): {sorted(blobs.n_pixels.tolist(), reverse=True)[:5]}")

        verdicts = I.classify_polygons(blobs, "2026-06-01", model=fp_model, cols=cols, threshold=0.20)
        vb = verdicts.merge(blobs[["blob_id", "n_pixels"]], on="blob_id")
        print("  Verdicts (top 5 blobs by size):")
        print(vb.sort_values("n_pixels", ascending=False).head(5)[
            ["blob_id", "n_pixels", "p_notpv", "p_becoming", "p_pv", "p_keep", "keep"]
        ].round(3).to_string(index=False))

        # Biggest blob: full vs eroded-core signature
        big_id = int(blobs.sort_values("n_pixels").iloc[-1]["blob_id"])
        big_geom = blobs.set_index("blob_id").loc[big_id, "geometry"]
        ts = I._gee_timeseries(blobs[blobs.blob_id == big_id], "2026-06-01", 12)
        ts["ndvi"] = (ts.B8 - ts.B4) / (ts.B8 + ts.B4 + 1e-9)
        ts["vis"] = (ts.B2 + ts.B3 + ts.B4) / 3
        print(f"  Biggest blob id={big_id}: ndvi mean={ts.ndvi.mean():.3f} "
              f"std={ts.ndvi.std():.3f} vis={ts.vis.mean():.0f}  n_months={len(ts)}")

        # Eroded core (inner 40%) signature to test dilution
        import geopandas as gpd
        g3857 = gpd.GeoSeries([big_geom], crs="EPSG:3857")
        # erode by ~30% of equivalent radius
        eq_r = (g3857.area.iloc[0] / np.pi) ** 0.5
        core = g3857.buffer(-0.35 * eq_r)
        if core.area.iloc[0] > 0:
            core4326 = core.to_crs("EPSG:4326")
            cg = gpd.GeoDataFrame({"blob_id": [9999]}, geometry=[core4326.iloc[0]], crs="EPSG:4326")
            ts_core = I._gee_timeseries(cg, "2026-06-01", 12)
            if len(ts_core):
                ts_core["ndvi"] = (ts_core.B8 - ts_core.B4) / (ts_core.B8 + ts_core.B4 + 1e-9)
                ts_core["vis"] = (ts_core.B2 + ts_core.B3 + ts_core.B4) / 3
                print(f"  ERODED CORE (-35% r): ndvi mean={ts_core.ndvi.mean():.3f} "
                      f"std={ts_core.ndvi.std():.3f} vis={ts_core.vis.mean():.0f}")
                cg2 = cg.copy()
                vc = I.classify_polygons(cg2, "2026-06-01", model=fp_model, cols=cols, threshold=0.20)
                print(f"  ERODED CORE verdict: p_keep={vc.iloc[0]['p_keep']:.3f} keep={vc.iloc[0]['keep']}")
        # Monthly trace
        print("  Monthly trace (full blob):")
        print(ts.sort_values("month")[["month", "ndvi", "vis", "B11", "B12"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
