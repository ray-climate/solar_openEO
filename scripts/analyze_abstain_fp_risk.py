#!/usr/bin/env python
"""Quantify the abstain-on-large risk across the 88 EU-unseen sites.

For every site: run the U-Net, vectorize detection blobs, and label each blob
TP-like (intersects a polygon-DB feature) or FP-like (does not). Then report
the SIZE distribution of FP-like vs TP-like blobs, focused on the abstain
threshold (5000px): how much FP-like detection would the abstain wrongly
PROTECT from deletion, vs how much it correctly shields real plants.

No GEE — U-Net + vectorize + spatial overlap only.
"""
from __future__ import annotations

import csv, importlib.util, sys
from pathlib import Path
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
NC_DIR = REPO / "docs/europe_unseen_2026MarMay/stacks"
ABSTAIN = 5000
OUT_CSV = REPO / "outputs/fp_classifier/abstain_fp_risk_blobs.csv"


def main():
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    from pyproj import Transformer
    from solar_fp_filter import inference as I

    EU = importlib.util.spec_from_file_location("eu", REPO / "scripts/test_europe_unseen.py")
    eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)
    from openeo_udp.udf.solar_pv_inference import (_build_model,_load_weights_compat,_load_band_stats,_load_registry)
    reg=_load_registry(); m=_build_model(reg); _load_weights_compat(m,eu.POST_WEIGHTS)
    bs=_load_band_stats(REPO/reg["band_stats_local"]); thr=float(reg.get("threshold",0.85))
    spec2=importlib.util.spec_from_file_location("rw",REPO/"scripts/19_test_real_world_sites.py")
    rw=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    tr=Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)

    polys=gpd.read_file(eu.GPKG)
    if polys.crs.to_epsg()!=3857: polys=polys.to_crs("EPSG:3857")
    bad=~polys.geometry.is_valid
    if bad.any(): polys.loc[bad,"geometry"]=polys.loc[bad,"geometry"].buffer(0)

    sites=[s for s in eu.select_sites(100) if (NC_DIR/f"{s['chip_id']}_stack.nc").exists()]
    print(f"{len(sites)} cached sites", flush=True)
    rows=[]
    for idx,site in enumerate(sites,1):
        cid=site["chip_id"]
        try:
            spectral,scl,_=rw.load_stack(NC_DIR/f"{cid}_stack.nc")
            comp,info=create_temporal_mosaic(spectral,scl)
            mask,_=eu.tiled_predict(m,np.transpose(comp[:13],(1,2,0)),bs,thr)
        except Exception as e:
            print(f"[{idx}] FAIL {cid}: {e}", flush=True); continue
        bbox=eu.aoi_bbox_4326(site["lat"],site["lon"],site["half_size_km"])
        x_min,y_min=tr.transform(bbox["west"],bbox["south"]); x_max,y_max=tr.transform(bbox["east"],bbox["north"])
        blobs=I.vectorize_mask(mask,x_min,y_max,x_max-x_min,y_max-y_min,crs_epsg=3857)
        if len(blobs)==0: continue
        blobs3857=blobs.to_crs("EPSG:3857")
        for (_,b),(_,b3) in zip(blobs.iterrows(), blobs3857.iterrows()):
            cand=list(polys.sindex.intersection(b3.geometry.bounds))
            is_tp=False
            if cand:
                is_tp=any(polys.iloc[c].geometry.intersects(b3.geometry) for c in cand)
            rows.append({"chip_id":cid,"tier":site["tier"],"n_pixels":int(b.n_pixels),
                         "label":"TP" if is_tp else "FP"})
        print(f"[{idx}/{len(sites)}] {cid}: {len(blobs)} blobs ({len(rows)} total)", flush=True)

    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["chip_id","tier","n_pixels","label"]); w.writeheader(); w.writerows(rows)

    # ---- Analysis ----
    import pandas as pd
    df=pd.DataFrame(rows)
    print(f"\n=== {len(df)} blobs across {df.chip_id.nunique()} sites ===", flush=True)
    for lab in ["TP","FP"]:
        d=df[df.label==lab]
        big=d[d.n_pixels>ABSTAIN]
        print(f"\n{lab}-like blobs: {len(d)} (total {d.n_pixels.sum():,}px)", flush=True)
        print(f"  >{ABSTAIN}px (abstain-protected): {len(big)} blobs, {big.n_pixels.sum():,}px "
              f"({100*big.n_pixels.sum()/max(d.n_pixels.sum(),1):.0f}% of {lab} px)", flush=True)
    fp=df[df.label=="FP"]; fp_big=fp[fp.n_pixels>ABSTAIN]
    print(f"\n>>> ABSTAIN RISK: {len(fp_big)} FP-like blobs >{ABSTAIN}px would be PROTECTED "
          f"from deletion ({fp_big.n_pixels.sum():,}px of FP area).", flush=True)
    print(f">>> Those are on {fp_big.chip_id.nunique()} of {df.chip_id.nunique()} sites.", flush=True)
    # show the largest FP-like blobs that abstain would protect
    print("\nLargest FP-like blobs abstain would protect:", flush=True)
    print(fp_big.nlargest(10,"n_pixels")[["chip_id","tier","n_pixels"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
