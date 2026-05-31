#!/usr/bin/env python
"""Decisive test: do the MEAN filter and the MEAN+PERCENTILE filter (idea #1)
keep the real plants that the mean filter previously KILLED?

For each flagged TP-loss site, get the U-Net's biggest detection blob, extract
BOTH mean and percentile features for it via GEE, score with both models, and
print the keep decision (threshold 0.2).
"""
from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
SITES = ["c-000314_r+001799", "c-000105_r+001859", "c+000355_r+002986"]
END = "2026-06-01"


def main():
    import ee; ee.Initialize()
    import lightgbm as lgb
    import geopandas as gpd
    from solar_fp_filter import inference as I, features as F, timeseries_gee as tg

    EU = importlib.util.spec_from_file_location("eu", REPO / "scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    from openeo_udp.udf.solar_pv_inference import (_build_model,_load_weights_compat,_load_band_stats,_load_registry)
    reg=_load_registry(); m=_build_model(reg); _load_weights_compat(m,eu.POST_WEIGHTS)
    bs=_load_band_stats(REPO/reg["band_stats_local"]); thr=float(reg.get("threshold",0.85))
    spec2=importlib.util.spec_from_file_location("rw",REPO/"scripts/19_test_real_world_sites.py")
    rw=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    from pyproj import Transformer
    tr=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)

    m_mean=lgb.Booster(model_file=str(I.DEFAULT_MODEL)); cols_mean=json.loads(I.DEFAULT_COLS.read_text())
    m_pct=lgb.Booster(model_file=str(REPO/"outputs/fp_classifier/model_pct.lgb"))
    cols_pct=json.loads((REPO/"outputs/fp_classifier/feature_cols_pct.json").read_text())

    base=ee.Date(f"{int(END[:4])-1}{END[4:]}"); months=ee.List.sequence(0,11)
    empty=ee.Image.constant([0]*len(tg._PCT_PROPS)).rename(tg._PCT_PROPS).updateMask(ee.Image.constant(0)).toFloat()
    red=ee.Reducer.percentile(tg.PCTS,maxBuckets=200,minBucketWidth=0.005,maxRaw=1000)

    def pct_ts_for_geom(geom4326):
        feat=ee.Feature(ee.Geometry(geom4326.__geo_interface__),{"sample_id":"blob"})
        fc=ee.FeatureCollection([feat])
        res=tg._percentile_perfeature(fc, f"{int(END[:4])-1}{END[4:]}", 12)
        return tg._parse_pct_result(res.getInfo())

    all_sites=eu.select_sites(100)
    for CID in SITES:
        site=[s for s in all_sites if s["chip_id"]==CID][0]
        spectral,scl,_=rw.load_stack(REPO/f"docs/europe_unseen_2026MarMay/stacks/{CID}_stack.nc")
        comp,info=create_temporal_mosaic(spectral,scl)
        mask,_=eu.tiled_predict(m,np.transpose(comp[:13],(1,2,0)),bs,thr)
        bbox=eu.aoi_bbox_4326(site["lat"],site["lon"],site["half_size_km"])
        x_min,y_min=tr.transform(bbox["west"],bbox["south"]); x_max,y_max=tr.transform(bbox["east"],bbox["north"])
        blobs=I.vectorize_mask(mask,x_min,y_max,x_max-x_min,y_max-y_min,crs_epsg=3857)
        big=blobs.sort_values("n_pixels").iloc[[-1]]
        # mean features (existing inference path)
        mean_ts=I._gee_timeseries(big,END,12)
        mf=F.compute_features(mean_ts)
        mf["abs_lat"]=abs(site["lat"]); mf["log_area"]=np.log10(max(big.to_crs(3857).geometry.area.iloc[0],1))
        # pct features
        pts=pct_ts_for_geom(big.iloc[0].geometry)
        pf=F.compute_pct_features(pts)
        comb=mf.merge(pf,on="sample_id",how="left") if "sample_id" in pf else mf.assign(**{c:np.nan for c in cols_pct})
        for c in cols_mean:
            if c not in mf: mf[c]=np.nan
        for c in cols_pct:
            if c not in comb: comb[c]=np.nan
        p_mean=m_mean.predict(mf[cols_mean].to_numpy(float))[0]
        p_comb=m_pct.predict(comb[cols_pct].to_numpy(float))[0]
        keep_mean=p_mean[1]+p_mean[2]; keep_pct=p_comb[1]+p_comb[2]
        print(f"\n{CID} (lat {site['lat']:.1f}, {int(big.iloc[0]['n_pixels'])}px):", flush=True)
        print(f"   MEAN filter   p_keep={keep_mean:.3f}  -> {'KEEP' if keep_mean>=0.2 else 'REJECT'}", flush=True)
        print(f"   MEAN+PCT (#1) p_keep={keep_pct:.3f}  -> {'KEEP' if keep_pct>=0.2 else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
