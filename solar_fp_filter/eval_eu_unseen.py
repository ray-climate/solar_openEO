"""Phase 5 — evaluate the FP filter on the 88 cached EU-unseen sites.

For each site with a cached 2026 Mar-May stack:
  1. Run the best U-Net (exp_round3_r101_reviewed) -> 2026 detection mask.
  2. Apply the temporal FP filter (GEE, last 12 months) -> filtered mask.
  3. Render a 4-panel figure:
       (1) RGB
       (2) RGB + 2024 ground-truth polygons (green)
       (3) RGB + 2026 U-Net detection (red)
       (4) RGB + 2026 detection AFTER filtering (red)
  4. Push each PNG to git and accumulate stats.

Runs on a SLURM compute node: cached stacks are local, the model loads
locally, GEE is reachable from compute nodes, git push works.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

EU = importlib.util.spec_from_file_location(
    "eu", REPO / "scripts/test_europe_unseen.py")
eu = importlib.util.module_from_spec(EU); EU.loader.exec_module(eu)

from solar_fp_filter import inference as I  # noqa: E402

OUT_DIR = REPO / "docs/europe_unseen_2026MarMay_filtered"
NC_DIR  = REPO / "docs/europe_unseen_2026MarMay/stacks"
FILTER_END_DATE = "2026-06-01"   # pull last 12 months (2025-06 .. 2026-06)


def _overlay_mask(ax, rgb, mask, color):
    """Show RGB then blend a semi-transparent colour where mask>0."""
    blend = rgb.copy()
    m = mask > 0
    for c in range(3):
        blend[..., c][m] = 0.45 * rgb[..., c][m] + 0.55 * color[c]
    ax.imshow(blend)


def _draw_polys(ax, polys_in, site, w, h, color="lime"):
    xmin = site["lon_xmin_3857"]; ymax = site["lat_ymax_3857"]
    x_extent = site["lon_xmax_3857"] - xmin
    y_extent = ymax - site["lat_ymin_3857"]
    for geom in polys_in:
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for g in geoms:
            xy = np.array(g.exterior.coords)
            px = (xy[:, 0] - xmin) / x_extent * w
            py = (ymax - xy[:, 1]) / y_extent * h
            ax.add_patch(MplPolygon(np.column_stack([px, py]), closed=True,
                                    fill=False, edgecolor=color, linewidth=2.0,
                                    alpha=0.95))


def render_4panel(site, rgb, mask, filt_mask, polys_in, out_path):
    h, w = mask.shape
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))

    ax[0].imshow(rgb); ax[0].set_title(f"2026 Mar-May RGB ({h*10/1000:.0f} km)", fontsize=11)

    ax[1].imshow(rgb); _draw_polys(ax[1], polys_in, site, w, h, "lime")
    ax[1].set_xlim(0, w); ax[1].set_ylim(h, 0)
    ax[1].set_title("Ground truth (polygon DB, green)", fontsize=11)

    _overlay_mask(ax[2], rgb, mask, [1.0, 0.15, 0.15])
    ax[2].set_title(f"2026 detection (red)  {int(mask.sum())} px", fontsize=11)

    _overlay_mask(ax[3], rgb, filt_mask, [1.0, 0.15, 0.15])
    removed = int(mask.sum()) - int(filt_mask.sum())
    pct = (100.0 * removed / max(int(mask.sum()), 1))
    ax[3].set_title(f"2026 detection AFTER filter  {int(filt_mask.sum())} px "
                    f"(-{pct:.0f}%)", fontsize=11)

    for a in ax:
        a.axis("off")
    fig.suptitle(
        f"{site['chip_id']}  [{site['tier'].upper()}, {site['unseen_class']}]  "
        f"area={site['area_m2']/1e6:.2f} km²  lat={site['lat']:+.3f} lon={site['lon']:+.3f}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate(threshold: float = I.KEEP_THRESHOLD, push: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sites = eu.select_sites(100)
    # Keep only sites whose cached stack exists
    sites = [s for s in sites if (NC_DIR / f"{s['chip_id']}_stack.nc").exists()]
    print(f"{len(sites)} sites with cached stacks", flush=True)

    # Model
    from openeo_udp.udf.solar_pv_inference import (
        _build_model, _load_weights_compat, _load_band_stats, _load_registry)
    reg = _load_registry()
    m = _build_model(reg); _load_weights_compat(m, eu.POST_WEIGHTS)
    band_stats = _load_band_stats(REPO / reg["band_stats_local"])
    thr = float(reg.get("threshold", 0.85))
    print(f"Model loaded. U-Net threshold={thr}, filter keep-threshold={threshold}", flush=True)

    # GEE + classifier model (loaded once)
    import ee; ee.Initialize()
    import lightgbm as lgb, json
    fp_model = lgb.Booster(model_file=str(I.DEFAULT_MODEL))
    fp_cols = json.loads(I.DEFAULT_COLS.read_text())

    # Stack loader + mosaic + polygons
    spec2 = importlib.util.spec_from_file_location(
        "rw", REPO / "scripts/19_test_real_world_sites.py")
    rw = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rw)
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    import geopandas as gpd
    from shapely.geometry import box as shp_box
    from pyproj import Transformer
    polys = gpd.read_file(eu.GPKG)
    if polys.crs.to_epsg() != 3857:
        polys = polys.to_crs("EPSG:3857")
    bad = ~polys.geometry.is_valid
    if bad.any():
        polys.loc[bad, "geometry"] = polys.loc[bad, "geometry"].buffer(0)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    rows = []
    t0 = time.time()
    total = len(sites)
    for idx, site in enumerate(sites, 1):
        cid = site["chip_id"]
        out_png = OUT_DIR / f"eu_{site['tier']}_{site['unseen_class']}_{cid}.png"
        if out_png.exists():
            print(f"[{idx}/{total}] skip-existing {out_png.name}", flush=True); continue
        nc = NC_DIR / f"{cid}_stack.nc"
        try:
            spectral, scl, _ = rw.load_stack(nc)
            composite, info = create_temporal_mosaic(spectral, scl)
            image_hwc = np.transpose(composite[:13], (1, 2, 0))
            rgb = eu.make_rgb_robust(composite)
            mask, _ = eu.tiled_predict(m, image_hwc, band_stats, thr)
        except Exception as e:
            print(f"[{idx}/{total}] INFER_FAIL {cid}: {e}", flush=True); continue

        h, w = mask.shape
        bbox = eu.aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
        x_min, y_min = tr.transform(bbox["west"], bbox["south"])
        x_max, y_max = tr.transform(bbox["east"], bbox["north"])
        site["lon_xmin_3857"] = x_min; site["lon_xmax_3857"] = x_max
        site["lat_ymin_3857"] = y_min; site["lat_ymax_3857"] = y_max

        # FP filter
        try:
            filt_mask, blobs = I.apply_filter(
                mask, x_min, y_max, x_max - x_min, y_max - y_min,
                end_date=FILTER_END_DATE, crs_epsg=3857,
                model=fp_model, cols=fp_cols, threshold=threshold)
        except Exception as e:
            print(f"[{idx}/{total}] FILTER_FAIL {cid}: {e} (keeping unfiltered)", flush=True)
            filt_mask, blobs = mask.copy(), None

        # GT polygons in AOI
        cand = list(polys.sindex.intersection((x_min, y_min, x_max, y_max)))
        chip_geom = shp_box(x_min, y_min, x_max, y_max)
        polys_in = []
        if cand:
            for g in polys.iloc[cand].geometry:
                inter = g.intersection(chip_geom) if g.is_valid else g.buffer(0).intersection(chip_geom)
                if not inter.is_empty:
                    polys_in.append(inter)

        render_4panel(site, rgb, mask, filt_mask, polys_in, out_png)

        n_blobs = 0 if blobs is None else len(blobs)
        n_removed = 0 if blobs is None else int((~blobs["keep"]).sum())
        det = int(mask.sum()); fdet = int(filt_mask.sum())
        elapsed = (time.time() - t0) / 60
        print(f"[{idx}/{total}] {cid}: det={det} -> {fdet} "
              f"(-{100*(det-fdet)/max(det,1):.0f}%)  blobs={n_blobs} removed={n_removed}  "
              f"[{elapsed:.1f}m]", flush=True)
        if push:
            _git_push(out_png, f"Add EU-unseen filtered 4-panel: {cid}")
        rows.append({"chip_id": cid, "tier": site["tier"],
                     "unseen_class": site["unseen_class"],
                     "det_px": det, "filt_px": fdet,
                     "removed_px": det - fdet, "n_blobs": n_blobs,
                     "n_blobs_removed": n_removed})

    # Summary CSV
    if rows:
        summ = OUT_DIR / "filter_summary.csv"
        with summ.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
            wr.writerows(rows)
        if push:
            _git_push(summ, f"Add EU-unseen filter summary ({len(rows)} sites)")
    print(f"\nDone: {len(rows)} sites rendered in {(time.time()-t0)/60:.1f}m", flush=True)


def _git_push(path: Path, msg: str):
    try:
        subprocess.run(["git", "-C", str(REPO), "add", str(path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "push", "origin", "main"],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:
        print(f"  GIT_FAIL: {str(e)[:100]}", flush=True)
