#!/usr/bin/env python
"""Render 4-panel figures for the 30 new EU-unseen sites with the SHIPPED
filter (mean classifier, thr 0.20, abstain OFF). Panels: RGB | GT polygons |
2026 detection | detection after filter. Pushes each PNG.
Run after eu_unseen30_download.py. GEE-reachable compute (SLURM ok).
"""
from __future__ import annotations
import csv, importlib.util, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))
OUT = REPO / "docs/europe_unseen30_2026MarMay"
NC = OUT / "stacks"


def main():
    import ee; ee.Initialize()
    import lightgbm as lgb, json, geopandas as gpd
    from shapely.geometry import box as shp_box
    from pyproj import Transformer
    from solar_fp_filter import inference as I, eval_eu_unseen as E

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
    polys=gpd.read_file(eu.GPKG)
    if polys.crs.to_epsg()!=3857: polys=polys.to_crs("EPSG:3857")
    bad=~polys.geometry.is_valid
    if bad.any(): polys.loc[bad,"geometry"]=polys.loc[bad,"geometry"].buffer(0)

    sites=list(csv.DictReader((OUT/"sites.csv").open()))
    sites=[s for s in sites if (NC/f"{s['chip_id']}_stack.nc").exists()]
    print(f"{len(sites)} sites with stacks", flush=True)
    rows=[]; t0=time.time()
    for i,s in enumerate(sites,1):
        cid=s["chip_id"]; lat=float(s["lat"]); lon=float(s["lon"])
        half=float(s["half_size_km"]); area=float(s["area_m2"])
        out_png=OUT/f"eu30_{s['tier']}_{cid}.png"
        if out_png.exists(): print(f"[{i}] skip {cid}",flush=True); continue
        try:
            spectral,scl,_=rw.load_stack(NC/f"{cid}_stack.nc")
            comp,info=create_temporal_mosaic(spectral,scl)
            rgb=eu.make_rgb_robust(comp)
            mask,_=eu.tiled_predict(m,np.transpose(comp[:13],(1,2,0)),bs,thr)
        except Exception as e:
            print(f"[{i}] INFER_FAIL {cid}: {e}",flush=True); continue
        bbox=eu.aoi_bbox_4326(lat,lon,half)
        x0,y0=tr.transform(bbox["west"],bbox["south"]); x1,y1=tr.transform(bbox["east"],bbox["north"])
        site={"chip_id":cid,"tier":s["tier"],"unseen_class":s.get("unseen_class",""),
              "lat":lat,"lon":lon,"area_m2":area,"half_size_km":half,
              "lon_xmin_3857":x0,"lon_xmax_3857":x1,"lat_ymin_3857":y0,"lat_ymax_3857":y1}
        try:
            filt,blobs=I.apply_filter(mask,x0,y1,x1-x0,y1-y0,end_date="2026-06-01",crs_epsg=3857,
                                      model=model,cols=cols,threshold=0.20)  # shipped filter
        except Exception as e:
            print(f"[{i}] FILTER_FAIL {cid}: {e} (unfiltered)",flush=True); filt=mask.copy()
        cand=list(polys.sindex.intersection((x0,y0,x1,y1))); chip=shp_box(x0,y0,x1,y1); pin=[]
        for g in polys.iloc[cand].geometry if cand else []:
            inter=g.intersection(chip) if g.is_valid else g.buffer(0).intersection(chip)
            if not inter.is_empty: pin.append(inter)
        E.render_4panel(site,rgb,mask,filt,pin,out_png)
        det=int(mask.sum()); fdet=int(filt.sum())
        print(f"[{i}/{len(sites)}] {cid}: det={det}->{fdet} (-{100*(det-fdet)/max(det,1):.0f}%) [{(time.time()-t0)/60:.1f}m]",flush=True)
        try:
            subprocess.run(["git","-C",str(REPO),"add",str(out_png)],check=True,capture_output=True)
            subprocess.run(["git","-C",str(REPO),"commit","-m",f"Add EU-unseen30 figure: {cid}"],check=True,capture_output=True)
            subprocess.run(["git","-C",str(REPO),"push","origin","main"],check=True,capture_output=True,timeout=120)
        except Exception as e: print(f"  GIT_FAIL {str(e)[:60]}",flush=True)
        rows.append({"chip_id":cid,"tier":s["tier"],"det_px":det,"filt_px":fdet})
    if rows:
        with (OUT/"summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nDone: {len(rows)} figures",flush=True)


if __name__ == "__main__":
    main()
