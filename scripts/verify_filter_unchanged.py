#!/usr/bin/env python
"""Verify the deliverable filter (mean model, thr 0.20, abstain OFF) reproduces
the Phase 5 EU-unseen results recorded in filter_summary.csv. Spot-checks a few
sites by re-running U-Net + filter and comparing det_px / filt_px.
"""
from __future__ import annotations
import csv, importlib.util, sys
from pathlib import Path
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
NC_DIR = REPO / "docs/europe_unseen_2026MarMay/stacks"
SUMMARY = REPO / "docs/europe_unseen_2026MarMay_filtered/filter_summary.csv"
CHECK = ["c-000250_r+001899", "c+000773_r+002852", "c-000060_r+001788"]  # varied removal


def main():
    import ee; ee.Initialize()
    import lightgbm as lgb, json, geopandas as gpd
    from pyproj import Transformer
    from solar_fp_filter import inference as I

    rec = {r["chip_id"]: r for r in csv.DictReader(SUMMARY.open())}
    EU = importlib.util.spec_from_file_location("eu", REPO/"scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    from openeo_udp.udf.solar_pv_inference import (_build_model,_load_weights_compat,_load_band_stats,_load_registry)
    reg=_load_registry(); m=_build_model(reg); _load_weights_compat(m,eu.POST_WEIGHTS)
    bs=_load_band_stats(REPO/reg["band_stats_local"]); thr=float(reg.get("threshold",0.85))
    spec2=importlib.util.spec_from_file_location("rw",REPO/"scripts/19_test_real_world_sites.py")
    rw=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    tr=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
    model=lgb.Booster(model_file=str(I.DEFAULT_MODEL)); cols=json.loads(I.DEFAULT_COLS.read_text())
    sites={s["chip_id"]:s for s in eu.select_sites(100)}

    print(f"{'chip_id':>20} {'det(rec)':>9} {'det(new)':>9} {'filt(rec)':>9} {'filt(new)':>9} match", flush=True)
    for cid in CHECK:
        s=sites[cid]
        spectral,scl,_=rw.load_stack(NC_DIR/f"{cid}_stack.nc")
        comp,info=create_temporal_mosaic(spectral,scl)
        mask,_=eu.tiled_predict(m,np.transpose(comp[:13],(1,2,0)),bs,thr)
        bbox=eu.aoi_bbox_4326(s["lat"],s["lon"],s["half_size_km"])
        x0,y0=tr.transform(bbox["west"],bbox["south"]); x1,y1=tr.transform(bbox["east"],bbox["north"])
        filt,_=I.apply_filter(mask,x0,y1,x1-x0,y1-y0,end_date="2026-06-01",crs_epsg=3857,
                              model=model,cols=cols,threshold=0.20)  # abstain default None
        det_new=int(mask.sum()); filt_new=int(filt.sum())
        r=rec[cid]; det_rec=int(r["det_px"]); filt_rec=int(r["filt_px"])
        ok = abs(det_new-det_rec)/max(det_rec,1)<0.02 and abs(filt_new-filt_rec)/max(filt_rec,1)<0.05
        print(f"{cid:>20} {det_rec:>9} {det_new:>9} {filt_rec:>9} {filt_new:>9} {'OK' if ok else 'DIFF'}", flush=True)


if __name__ == "__main__":
    main()
