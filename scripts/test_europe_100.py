#!/usr/bin/env python
"""Test the post-retrained model on 100 Europe medium/large solar PV sites
NOT in the master_manifest (i.e., never used for training extraction).

Inputs (read-only):
  - data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg
  - outputs/final/master_manifest.csv  (to filter out already-extracted chips)

Workflow:
  1. Select 100 polygons in Europe with geometry area >= 500K m², not in
     master_manifest (truly unseen by training).
  2. Compute adaptive AOI size from polygon area so each site's AOI covers
     the whole plant.
  3. Throttled OpenEO pulls (max 25 concurrent) for the 2026-03 to 2026-05
     mosaic of each site.
  4. Per-site: build SLIC mosaic -> tiled R3 R101 reviewed inference ->
     render 3-panel PNG (RGB + polygon overlay + detection) -> git push.
  5. Final summary CSV + console report.

Uses the IMPROVED RGB stretch (5-95 percentile + saturation clip) so sites
with cloud / scene-boundary contamination don't render as all-black.
"""
from __future__ import annotations

import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

REPO = Path("/gws/ssde/j25b/gbov/solar_openEO")
sys.path.insert(0, str(REPO))

from extraction_pipeline.tiling import chip_id_to_bounds  # noqa: E402

GPKG    = REPO / "data/solar_panels_buffered_v4_10_dissolved_with_year_predicted.gpkg"
MASTER  = REPO / "outputs/final/master_manifest.csv"
OUT_DIR = REPO / "docs/europe_100_2026MarMay"
NC_DIR  = OUT_DIR / "stacks"

POST_WEIGHTS = REPO / "experiments/exp_round3_r101_reviewed/best.weights.h5"
TEMPORAL = ["2026-03-01", "2026-05-31"]
BACKEND_URL = "https://openeo.dataspace.copernicus.eu"

CELL = 2560  # m / chip in EPSG:3857
S2_L1C_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", "B10", "B11", "B12"]


def _fmt_chip(col: int, row: int) -> str:
    return f"c{'+' if col >= 0 else '-'}{abs(col):06d}_r{'+' if row >= 0 else '-'}{abs(row):06d}"


def select_sites(n: int = 100, seed: int = 42) -> list[dict]:
    """Pick n Europe medium/large polygons not already in master_manifest."""
    import geopandas as gpd
    gdf = gpd.read_file(GPKG)
    g = gdf.to_crs("EPSG:3857").copy()
    g["area_m2"] = g.geometry.area
    g["lat"] = gdf["latitude"]; g["lon"] = gdf["longitude"]
    # Loose Europe bbox
    eu = g[(g.lat >= 35) & (g.lat <= 72) & (g.lon >= -10) & (g.lon <= 45)]
    eu = eu[eu.area_m2 >= 500_000].copy()  # medium and up
    eu["cx"] = eu.geometry.centroid.x
    eu["cy"] = eu.geometry.centroid.y
    eu["chip_col"] = (eu["cx"] // CELL).astype(int)
    eu["chip_row"] = (eu["cy"] // CELL).astype(int)
    eu["chip_id"]  = [_fmt_chip(c, r) for c, r in zip(eu["chip_col"], eu["chip_row"])]

    seen = set()
    with MASTER.open() as f:
        for r in csv.DictReader(f):
            seen.add(r["chip_id_str"])
    eu = eu[~eu["chip_id"].isin(seen)]
    eu = eu.drop_duplicates(subset=["chip_id"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    # Stratified pick: 25 LARGE (>=2M), 75 MEDIUM (500K-2M)
    large = eu[eu.area_m2 >= 2_000_000]
    med   = eu[eu.area_m2 <  2_000_000]
    n_lg = min(25, len(large), n // 4)
    n_md = min(n - n_lg, len(med))
    pick_lg = large.iloc[rng.choice(len(large), n_lg, replace=False)]
    pick_md = med.iloc[rng.choice(len(med), n_md, replace=False)]
    chosen = pick_lg._append(pick_md) if hasattr(pick_lg, "_append") else \
             gpd.GeoDataFrame(pick_lg).append(pick_md)
    chosen = chosen.reset_index(drop=True)

    sites: list[dict] = []
    for _, r in chosen.iterrows():
        a = float(r["area_m2"])
        if a < 1_000_000:  half_km = 2.5
        elif a < 5_000_000: half_km = 3.5
        elif a < 25_000_000: half_km = 5.0
        else: half_km = 7.5
        sites.append({
            "chip_id":      r["chip_id"],
            "chip_col":     int(r["chip_col"]),
            "chip_row":     int(r["chip_row"]),
            "lat":          float(r["lat"]),
            "lon":          float(r["lon"]),
            "area_m2":      a,
            "tier":         "large" if a >= 2_000_000 else "medium",
            "half_size_km": half_km,
            "year":         int(r["year"]) if "year" in r and r["year"] is not None else None,
            "geometry":     r["geometry"],
        })
    return sites


def aoi_bbox_4326(lat: float, lon: float, half_km: float) -> dict:
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))
    return {"west": lon - dlon, "east": lon + dlon,
            "south": lat - dlat, "north": lat + dlat, "crs": "EPSG:4326"}


def make_rgb_robust(composite: np.ndarray) -> np.ndarray:
    """5-95 percentile stretch with saturation clip, robust to scene artifacts."""
    rgb = np.transpose(composite[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
    # clip insane high values (S2 L1C reflectance is typically 0-10000;
    # values >5000 are usually clouds/snow/saturated boundaries)
    clipped = np.clip(rgb, 0, 5000)
    valid = clipped[clipped > 100]   # also drop near-zero (masked) pixels
    if valid.size == 0:
        return np.zeros_like(rgb)
    lo, hi = np.percentile(valid, 5), np.percentile(valid, 95)
    out = (clipped - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0, 1)


def tiled_predict(model, image_hwc, band_stats, thr):
    from openeo_udp.udf.solar_pv_inference import normalize_zscore
    h, w, c = image_hwc.shape
    probs = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, 256):
        for x0 in range(0, w, 256):
            tile = image_hwc[y0:y0+256, x0:x0+256, :]
            th, tw = tile.shape[0], tile.shape[1]
            if th < 256 or tw < 256:
                pad = np.zeros((256, 256, c), dtype=np.float32)
                pad[:th, :tw, :] = tile; tile = pad
            normed = normalize_zscore(tile[np.newaxis], band_stats)
            p = model.predict(normed, verbose=0)[0, :, :, 0]
            probs[y0:y0+th, x0:x0+tw] = p[:th, :tw]
    return (probs > thr).astype(np.uint8), probs


def render_site(site, rgb, mask, polys_in_chip, out_path):
    """3-panel: RGB | RGB + polygon overlay | RGB + detection overlay."""
    h, w = mask.shape
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))

    ax[0].imshow(rgb)
    ax[0].set_title(f"2026 Mar–May mosaic RGB ({h}×{w} ≈ {h*10/1000:.0f} km)", fontsize=11)
    ax[0].axis("off")

    ax[1].imshow(rgb)
    if polys_in_chip is not None:
        from shapely.geometry import MultiPolygon
        # Pixel transform: AOI is centered on polygon centroid, EPSG:3857 bbox
        xmin = site["lon_xmin_3857"]; ymax = site["lat_ymax_3857"]
        # Compute scale
        x_extent = site["lon_xmax_3857"] - xmin
        y_extent = ymax - site["lat_ymin_3857"]
        for geom in polys_in_chip:
            geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for g in geoms:
                xy = np.array(g.exterior.coords)
                px = (xy[:, 0] - xmin) / x_extent * w
                py = (ymax - xy[:, 1]) / y_extent * h
                ax[1].add_patch(MplPolygon(np.column_stack([px, py]),
                                            closed=True, fill=False,
                                            edgecolor="lime", linewidth=2.0, alpha=0.95))
    ax[1].set_xlim(0, w); ax[1].set_ylim(h, 0)
    ax[1].set_title("Polygon database (green outline)", fontsize=11)
    ax[1].axis("off")

    det = rgb.copy()
    det[mask > 0] = [1.0, 0.15, 0.15]
    ax[2].imshow(det)
    ax[2].set_title(f"Post-fine-tune detection (red)  —  {int(mask.sum())} px "
                    f"({mask.sum()*100/mask.size:.2f}%)", fontsize=11)
    ax[2].axis("off")

    fig.suptitle(
        f"{site['chip_id']}  [{site['tier'].upper()}]  EU  "
        f"area={site['area_m2']/1e6:.2f} km²  AOI={2*site['half_size_km']:.0f} km  "
        f"lat={site['lat']:+.3f} lon={site['lon']:+.3f}  "
        f"polygon_year={site.get('year','?')}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def git_push_png(png_path: Path, msg: str) -> bool:
    """Add + commit + push a single PNG. Returns True on success."""
    try:
        subprocess.run(["git", "-C", str(REPO), "add", str(png_path)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "push", "origin", "main"],
                       check=True, capture_output=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  GIT_FAIL {e.cmd}: {e.stderr.decode()[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"  GIT_EXC: {e}", flush=True)
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NC_DIR.mkdir(parents=True, exist_ok=True)

    sites = select_sites(100)
    print(f"Selected {len(sites)} Europe medium/large sites", flush=True)
    n_lg = sum(1 for s in sites if s["tier"] == "large")
    n_md = sum(1 for s in sites if s["tier"] == "medium")
    print(f"  large (>=2 M m²): {n_lg}    medium (0.5-2 M m²): {n_md}", flush=True)

    # Auth + submit all jobs (throttled)
    import openeo
    conn = openeo.connect(BACKEND_URL)
    conn.authenticate_oidc()
    print("Authenticated.", flush=True)

    # Reuse load_stack from existing module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rw", REPO / "scripts/19_test_real_world_sites.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.TEMPORAL = TEMPORAL

    # Load post-retrained model up front (so first finished download can be
    # processed immediately without re-loading on every call)
    from openeo_udp.udf.solar_pv_inference import (
        _build_model, _load_weights_compat, _load_band_stats, _load_registry,
    )
    reg = _load_registry()
    print(f"Loading post-retrained model ({reg['backbone']}) ...", flush=True)
    m = _build_model(reg)
    _load_weights_compat(m, POST_WEIGHTS)
    band_stats = _load_band_stats(REPO / reg["band_stats_local"])
    thr = float(reg.get("threshold", 0.85))
    print(f"Threshold: {thr}", flush=True)

    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    import geopandas as gpd
    from shapely.geometry import box as shp_box

    polys = gpd.read_file(GPKG)
    if polys.crs.to_epsg() != 3857:
        polys = polys.to_crs("EPSG:3857")
    invalid = ~polys.geometry.is_valid
    if invalid.any():
        polys.loc[invalid, "geometry"] = polys.loc[invalid, "geometry"].buffer(0)

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def process_and_push(site, idx, total, t_start):
        """Build mosaic + run inference + render + git push for one site."""
        cid = site["chip_id"]
        nc = NC_DIR / f"{cid}_stack.nc"
        out_png = OUT_DIR / f"eu_{site['tier']}_{cid}.png"
        if out_png.exists():
            print(f"[{idx:>3}/{total}] [skip-existing] {out_png.name}", flush=True)
            return None
        try:
            spectral, scl, _ = mod.load_stack(nc)
            composite, info = create_temporal_mosaic(spectral, scl)
            image_hwc = np.transpose(composite[:13], (1, 2, 0))
            rgb = make_rgb_robust(composite)
            mask, _ = tiled_predict(m, image_hwc, band_stats, thr)
        except Exception as e:
            print(f"[{idx:>3}/{total}] FAIL {cid}: {e}", flush=True)
            return None

        h, w = mask.shape
        bbox = aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
        x_min, y_min = tr.transform(bbox["west"], bbox["south"])
        x_max, y_max = tr.transform(bbox["east"], bbox["north"])
        site["lon_xmin_3857"] = x_min; site["lon_xmax_3857"] = x_max
        site["lat_ymin_3857"] = y_min; site["lat_ymax_3857"] = y_max
        cand = list(polys.sindex.intersection((x_min, y_min, x_max, y_max)))
        chip_geom = shp_box(x_min, y_min, x_max, y_max)
        polys_in = []
        if cand:
            sub = polys.iloc[cand]
            for g in sub.geometry:
                try:
                    inter = g.intersection(chip_geom)
                except Exception:
                    inter = g.buffer(0).intersection(chip_geom)
                if not inter.is_empty:
                    polys_in.append(inter)
        render_site(site, rgb, mask, polys_in, out_png)
        det_px = int(mask.sum())
        elapsed = (time.time() - t_start) / 60
        eta = elapsed / idx * (total - idx) if idx else 0
        print(f"[{idx:>3}/{total}] {cid} ({site['tier']:>6s}, lat={site['lat']:>+5.1f})  "
              f"det={det_px:>7d}  scenes={info['n_scenes_used']}  "
              f"[{elapsed:5.1f}m elapsed, ETA {eta:5.1f}m]", flush=True)
        ok = git_push_png(out_png, f"Add EU-100 test PNG [{idx}/{total}]: {cid}")
        if not ok:
            print(f"  (continuing despite git failure)", flush=True)
        return {
            "chip_id": cid, "tier": site["tier"],
            "lat": site["lat"], "lon": site["lon"], "area_m2": site["area_m2"],
            "polygon_year": site.get("year", ""),
            "half_size_km": site["half_size_km"],
            "aoi_px_h": h, "aoi_px_w": w,
            "det_px": det_px, "det_frac": det_px / mask.size,
            "n_polys_in_aoi": len(polys_in),
            "n_scenes_used": info["n_scenes_used"],
        }

    # ------------- STREAMING LOOP: download + process + push interleaved -------------
    pending = list(sites)
    in_flight: dict[str, dict] = {}
    rows = []
    processed_n = 0
    t_start = time.time()
    print(f"Streaming download+inference for {len(sites)} sites ...", flush=True)

    # First sweep: skip already-cached sites by processing them immediately
    cached_now: list[dict] = []
    still_pending = []
    for s in pending:
        if (NC_DIR / f"{s['chip_id']}_stack.nc").exists():
            cached_now.append(s)
        else:
            still_pending.append(s)
    pending = still_pending
    print(f"  {len(cached_now)} already-downloaded stacks will be processed first; "
          f"{len(pending)} need OpenEO.", flush=True)
    for s in cached_now:
        processed_n += 1
        r = process_and_push(s, processed_n, len(sites), t_start)
        if r: rows.append(r)

    # Now the streaming download+poll loop. If submission fails (concurrent
    # limit / rate limit), put the site BACK on the front of the queue and
    # back off — we only consume the submission slot when CDSE accepts it.
    backoff_until = 0.0  # epoch seconds; if > now, skip new submissions
    submit_fails = 0
    while pending or in_flight:
        now = time.time()
        if now >= backoff_until:
            while pending and len(in_flight) < 25:
                s = pending.pop(0)
                try:
                    job = mod._build_job(conn, s)
                    job.start_job()
                except Exception as exc:
                    submit_fails += 1
                    pending.insert(0, s)   # put back at front for retry
                    msg = str(exc)[:120]
                    print(f"  SUBMIT_FAIL {s['chip_id']}: {msg}", flush=True)
                    if "ConcurrentJobLimit" in msg or "429" in msg or "Too Many" in msg:
                        backoff_until = now + 60  # back off 60s
                    elif submit_fails % 5 == 0:
                        backoff_until = now + 30
                    break
                in_flight[s["chip_id"]] = {"site": s, "job": job,
                                           "nc_path": NC_DIR / f"{s['chip_id']}_stack.nc"}
                elapsed = (time.time() - t_start) / 60
                print(f"  [{elapsed:5.1f}m] [submit] {s['chip_id']} ({s['tier']:>6s}) "
                      f"in_flight={len(in_flight)} pending={len(pending)}", flush=True)

        finished_cids = []
        for cid, p in list(in_flight.items()):
            try:
                status = p["job"].status()
            except Exception as exc:
                print(f"  POLL_FAIL {cid}: {exc}", flush=True)
                continue
            if status == "finished":
                try:
                    tmp = NC_DIR / f"_tmp_{cid}"; tmp.mkdir(parents=True, exist_ok=True)
                    p["job"].get_results().download_files(tmp)
                    ncs = list(tmp.glob("*.nc"))
                    if not ncs:
                        raise FileNotFoundError(f"no nc for {cid}")
                    ncs[0].rename(p["nc_path"])
                    for f in tmp.iterdir(): f.unlink()
                    tmp.rmdir()
                    elapsed = (time.time() - t_start) / 60
                    print(f"  [{elapsed:5.1f}m] FINISHED {cid} ({p['nc_path'].stat().st_size/1e6:.0f} MB)", flush=True)
                    # IMMEDIATELY process + push this site
                    processed_n += 1
                    r = process_and_push(p["site"], processed_n, len(sites), t_start)
                    if r: rows.append(r)
                except Exception as exc:
                    print(f"  DOWNLOAD_FAIL {cid}: {exc}", flush=True)
                finished_cids.append(cid)
            elif status in ("error", "canceled"):
                print(f"  FAILED {cid} -> {status}", flush=True)
                finished_cids.append(cid)
        for cid in finished_cids:
            del in_flight[cid]
        if in_flight or pending:
            time.sleep(20)
    print(f"\nProcessed {processed_n}/{len(sites)} sites.", flush=True)

    # Final summary CSV
    summary = OUT_DIR / "summary.csv"
    if rows:
        with summary.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        git_push_png(summary, f"Add EU-100 summary CSV ({len(rows)} sites)")
        print(f"\nWrote {summary.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
