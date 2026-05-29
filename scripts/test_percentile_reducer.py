#!/usr/bin/env python
"""Test idea #1: recover the diluted panel signal with a PERCENTILE reducer.

For each failing blob, instead of the whole-blob MEAN reflectance, compute
the per-pixel NDVI / NDBI distribution inside the blob and pull the
panel-dominated tail (low-NDVI / high-NDBI percentiles). Panels are the
darkest, most built-up pixels, so their signature should survive in the
tail even when grass dominates the mean.

Compares, per site (averaged over 12 months):
  mean NDVI/NDBI  (what the current filter uses -> failed)
  vs p10/p25 NDVI and p75/p90 NDBI (panel-dominated pixels)
against the training pv-class mean ranges.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
SITES = ["c-000314_r+001799", "c-000105_r+001859", "c+000355_r+002986"]


def monthly_index_ic(fc, start, n_months, _tg):
    import ee
    months = ee.List.sequence(0, n_months - 1)
    base = ee.Date(start)
    empty = ee.Image.constant([0, 0]).rename(["NDVI", "NDBI"]) \
        .updateMask(ee.Image.constant(0)).toFloat()

    def make(m):
        s = base.advance(m, "month"); e = s.advance(1, "month")
        coll = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(fc).filterDate(s, e)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
                .map(_tg._mask_qa60).select(["B4", "B8", "B11"]))
        comp = coll.median()

        def idx(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
            return ndvi.addBands(ndbi)
        out = ee.Image(ee.Algorithms.If(coll.size().gt(0), idx(comp), empty))
        return out.rename(["NDVI", "NDBI"]).set("month", s.format("YYYY-MM"))

    return ee.ImageCollection(months.map(make))


def main():
    import ee; ee.Initialize()
    import glob
    import geopandas as gpd
    from solar_fp_filter import inference as I, features as F, timeseries_gee as tg

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

    # Training pv-class mean NDVI/NDBI reference (tight polygons)
    ts_tr = pd.concat([pd.read_parquet(f) for f in glob.glob(str(REPO / "outputs/fp_classifier/timeseries/*.parquet"))], ignore_index=True)
    poly = gpd.read_file(REPO / "outputs/fp_classifier/polygons.gpkg")[["sample_id", "class"]]
    ts_tr = ts_tr.merge(poly, on="sample_id")
    ts_tr["ndvi"] = (ts_tr.B8 - ts_tr.B4) / (ts_tr.B8 + ts_tr.B4 + 1e-9)
    ts_tr["ndbi"] = (ts_tr.B11 - ts_tr.B8) / (ts_tr.B11 + ts_tr.B8 + 1e-9)
    pvm = ts_tr[ts_tr["class"].isin(["pv", "becoming_pv"])]
    npvm = ts_tr[ts_tr["class"] == "not_pv"]
    print("=== Training reference (per-sample mean, tight polygons) ===")
    print(f"  pv    : NDVI {pvm.groupby('sample_id').ndvi.mean().median():.3f}  "
          f"NDBI {pvm.groupby('sample_id').ndbi.mean().median():.3f}")
    print(f"  not_pv: NDVI {npvm.groupby('sample_id').ndvi.mean().median():.3f}  "
          f"NDBI {npvm.groupby('sample_id').ndbi.mean().median():.3f}")

    all_sites = eu.select_sites(100)
    for CID in SITES:
        site = [s for s in all_sites if s["chip_id"] == CID][0]
        spectral, scl, _ = rw.load_stack(REPO / f"docs/europe_unseen_2026MarMay/stacks/{CID}_stack.nc")
        comp, info = create_temporal_mosaic(spectral, scl)
        mask, _ = eu.tiled_predict(m, np.transpose(comp[:13], (1, 2, 0)), bs, thr)
        bbox = eu.aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
        x_min, y_min = tr.transform(bbox["west"], bbox["south"])
        x_max, y_max = tr.transform(bbox["east"], bbox["north"])
        blobs = I.vectorize_mask(mask, x_min, y_max, x_max - x_min, y_max - y_min, crs_epsg=3857)
        big = blobs.sort_values("n_pixels").iloc[[-1]]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(big.iloc[0].geometry.__geo_interface__), {"id": 1})])
        ic = monthly_index_ic(fc, "2025-06-01", 12, tg)
        red = ee.Reducer.mean().combine(
            ee.Reducer.percentile([10, 25, 75, 90]), sharedInputs=True)

        def per_img(img):
            r = img.reduceRegions(collection=fc, reducer=red, scale=10)
            return r.map(lambda f: f.set("month", img.get("month")))
        flat = ic.map(per_img).flatten().select(
            ["NDVI_mean", "NDVI_p10", "NDVI_p25", "NDBI_mean", "NDBI_p75", "NDBI_p90", "month"],
            retainGeometry=False)
        info2 = tg._getinfo_with_retry(flat)
        recs = [f["properties"] for f in info2["features"]]
        d = pd.DataFrame(recs).apply(pd.to_numeric, errors="coerce")
        print(f"\n=== {CID} (lat {site['lat']:.1f}, {int(big.iloc[0]['n_pixels'])} px) — avg over months ===")
        print(f"  NDVI: mean={d.NDVI_mean.mean():.3f}  p25={d.NDVI_p25.mean():.3f}  "
              f"p10={d.NDVI_p10.mean():.3f}   (panel pixels = LOW NDVI; pv ref ~0.20)")
        print(f"  NDBI: mean={d.NDBI_mean.mean():.3f}  p75={d.NDBI_p75.mean():.3f}  "
              f"p90={d.NDBI_p90.mean():.3f}   (panel pixels = HIGH NDBI; pv ref ~0.13)")


if __name__ == "__main__":
    main()
