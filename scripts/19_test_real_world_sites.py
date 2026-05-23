#!/usr/bin/env python
"""Real-world deployment test: 6 PV sites at varied sizes, Feb-Apr 2026.

Workflow:
  1. Pick 6 stratified sites from master_manifest.csv (small/medium/large).
  2. Pull multi-temporal S2 L1C+SCL stacks via CDSE OpenEO over adaptive AOIs
     (5 km for small/medium, 10 km for large).
  3. Build temporal median mosaics (SLIC-based, same algorithm as production UDF).
  4. Run R3 R101 detection (registry-driven) tiled across each mosaic.
  5. Generate RGB + detection quickview PNGs.

Phases (use --phase to control):
  - select   : print proposed sites + AOIs only (no network)
  - download : submit OpenEO jobs, poll until done, save .nc stacks
  - infer    : mosaic + tile + predict + render PNGs (no network)
  - all      : run all three sequentially (default)

Usage:
  conda run -n base python scripts/19_test_real_world_sites.py --phase select
  conda run -n base python scripts/19_test_real_world_sites.py --phase download
  conda run -n base python scripts/19_test_real_world_sites.py --phase infer
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- Paths ---
MASTER_MANIFEST = REPO / "outputs/final/master_manifest.csv"
REPACKED_H5 = REPO / "outputs/final/final_dataset_repacked.h5"
OUTPUT_DIR = REPO / "docs/realworld_test_2026Q1"

# --- Test config ---
BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
TEMPORAL = ["2026-02-01", "2026-04-30"]
S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]
CHIP_SIZE_M = 2560  # 256 px at 10 m


# ============================================================
# Phase 1: site selection
# ============================================================

def select_sites() -> list[dict]:
    """Pick 6 sites: 2 small, 2 medium, 2 large.

    Stratification keys on `panel_frac` from master_manifest.csv.  Within
    each tier we prefer geographic diversity (different continents) and
    `status=EXTRACTED` (i.e. chips we actually have masks for).
    """
    rows = []
    with open(MASTER_MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["chip_type"] != "positive":
                continue
            if r["status"] != "EXTRACTED":
                continue
            try:
                pf = float(r["panel_frac"])
            except (TypeError, ValueError):
                continue
            rows.append({
                "chip_id": r["chip_id_str"],
                "col": int(r["chip_col"]),
                "row": int(r["chip_row"]),
                "lat": float(r["chip_center_lat"]),
                "lon": float(r["chip_center_lon"]),
                "continent": r["continent"],
                "panel_frac": pf,
            })

    # Stratify
    small = [r for r in rows if 0.02 <= r["panel_frac"] < 0.08]
    medium = [r for r in rows if 0.20 <= r["panel_frac"] < 0.40]
    large = [r for r in rows if r["panel_frac"] >= 0.50]

    def pick_diverse(pool, n):
        # Prefer one from each continent, then fill
        pool_sorted = sorted(pool, key=lambda r: r["panel_frac"], reverse=True)
        chosen, seen = [], set()
        for r in pool_sorted:
            if r["continent"] not in seen:
                chosen.append(r); seen.add(r["continent"])
            if len(chosen) >= n:
                break
        for r in pool_sorted:
            if r in chosen:
                continue
            chosen.append(r)
            if len(chosen) >= n:
                break
        return chosen[:n]

    sites = []
    for tier, pool, n, half_km in [
        ("small",  small,  2, 2.5),
        ("medium", medium, 2, 2.5),
        ("large",  large,  2, 5.0),
    ]:
        for s in pick_diverse(pool, n):
            sites.append({**s, "tier": tier, "half_size_km": half_km})
    return sites


def aoi_bbox_4326(lat: float, lon: float, half_km: float) -> dict:
    """Build a square EPSG:4326 bbox centered at lat/lon with side 2*half_km."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))
    return {
        "west":  lon - dlon,
        "east":  lon + dlon,
        "south": lat - dlat,
        "north": lat + dlat,
        "crs":   "EPSG:4326",
    }


# ============================================================
# Phase 2: OpenEO download
# ============================================================

def download_stack(conn, site: dict, out_dir: Path) -> Path:
    """Submit OpenEO job for one AOI, wait, save netCDF stack."""
    nc_path = out_dir / f"{site['chip_id']}_stack.nc"
    if nc_path.exists():
        print(f"  [cached] {nc_path.name} ({nc_path.stat().st_size/1e6:.1f} MB)")
        return nc_path

    bbox = aoi_bbox_4326(site["lat"], site["lon"], site["half_size_km"])
    s2_l1c = conn.load_collection(
        "SENTINEL2_L1C", spatial_extent=bbox,
        temporal_extent=TEMPORAL, bands=S2_L1C_BANDS,
    ).resample_spatial(resolution=10, projection="EPSG:3857")
    s2_scl = conn.load_collection(
        "SENTINEL2_L2A", spatial_extent=bbox,
        temporal_extent=TEMPORAL, bands=["SCL"],
    ).resample_spatial(resolution=10, projection="EPSG:3857")
    merged = s2_l1c.merge_cubes(s2_scl)
    job = merged.create_job(title=f"rw_{site['chip_id']}", out_format="netCDF")
    job.start_job()
    print(f"  [job] {site['chip_id']} ({site['tier']:>6s}, {site['half_size_km']:.1f} km) -> {job.job_id}")
    return {"site": site, "job": job, "nc_path": nc_path}


def wait_for_jobs(pending: list[dict], out_dir: Path) -> None:
    """Poll all pending OpenEO jobs in parallel until each completes."""
    started = time.time()
    done: dict[str, bool] = {}
    while len(done) < len(pending):
        for p in pending:
            cid = p["site"]["chip_id"]
            if cid in done:
                continue
            status = p["job"].status()
            if status == "finished":
                tmp = out_dir / f"_tmp_{cid}"
                tmp.mkdir(parents=True, exist_ok=True)
                p["job"].get_results().download_files(tmp)
                ncs = list(tmp.glob("*.nc"))
                if not ncs:
                    raise FileNotFoundError(f"No netCDF for {cid}")
                ncs[0].rename(p["nc_path"])
                for f in tmp.iterdir():
                    f.unlink()
                tmp.rmdir()
                size_mb = p["nc_path"].stat().st_size / 1e6
                print(f"  [{(time.time()-started)/60:5.1f} min] FINISHED {cid} ({size_mb:.1f} MB)")
                done[cid] = True
            elif status in ("error", "canceled"):
                print(f"  [{(time.time()-started)/60:5.1f} min] FAILED   {cid} -> {status}")
                done[cid] = True
        if len(done) < len(pending):
            time.sleep(20)


# ============================================================
# Phase 3: mosaic + inference + render
# ============================================================

def load_stack(nc_path: Path):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    band_arrs = [ds[b].values for b in S2_L1C_BANDS if b in ds]
    spectral = np.stack(band_arrs, axis=1).astype(np.float32)  # (T, 13, H, W)
    scl = ds["SCL"].values.astype(np.int32)
    t_dim = next((d for d in ds[S2_L1C_BANDS[0]].dims if d in ("t", "time")), None)
    dates = ([str(np.datetime_as_string(t, unit="D")) for t in ds.coords[t_dim].values]
             if t_dim and t_dim in ds.coords else [f"s{i}" for i in range(spectral.shape[0])])
    ds.close()
    return spectral, scl, dates


def run_tiled_inference(image_hwc: np.ndarray, threshold: float | None = None):
    """Tiled 256x256 inference with non-overlapping windows + edge padding."""
    from openeo_udp.udf.solar_pv_inference import _get_model_and_stats, normalize_zscore
    model, band_stats, registry = _get_model_and_stats()
    thr = float(threshold if threshold is not None else registry.get("threshold", 0.85))
    h, w, c = image_hwc.shape
    probs = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, 256):
        for x0 in range(0, w, 256):
            tile = image_hwc[y0:y0+256, x0:x0+256, :]
            th, tw = tile.shape[0], tile.shape[1]
            if th < 256 or tw < 256:
                pad = np.zeros((256, 256, c), dtype=np.float32)
                pad[:th, :tw, :] = tile
                tile = pad
            normed = normalize_zscore(tile[np.newaxis], band_stats)
            pred = model.predict(normed, verbose=0)
            probs[y0:y0+th, x0:x0+tw] = pred[0, :th, :tw, 0]
    return (probs > thr).astype(np.uint8), probs, thr


def make_rgb(composite: np.ndarray) -> np.ndarray:
    rgb = np.transpose(composite[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
    valid = rgb[rgb > 0]
    lo, hi = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size > 0 else (0, 1)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def render_quickview(site: dict, rgb: np.ndarray, mask: np.ndarray, probs: np.ndarray,
                     thr: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = mask.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title(f"RGB ({h}×{w} px, 10 m)", fontsize=11); axes[0].axis("off")
    overlay = rgb.copy()
    red = mask > 0
    overlay[red] = [1.0, 0.15, 0.15]
    axes[1].imshow(overlay)
    axes[1].set_title(f"RGB + detection overlay (thr={thr:.2f})", fontsize=11); axes[1].axis("off")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    n_det = int(mask.sum())
    axes[2].set_title(f"Detection mask — {n_det} px ({n_det*100/mask.size:.2f}%)", fontsize=11)
    axes[2].axis("off")
    fig.suptitle(
        f"{site['chip_id']}  [{site['tier'].upper()}]  "
        f"{site['continent']}  panel_frac={site['panel_frac']:.2f}  "
        f"AOI={2*site['half_size_km']:.0f} km",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_site(site: dict, nc_path: Path, out_dir: Path) -> dict:
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic
    print(f"  {site['chip_id']} ({site['tier']:>6s}, {site['continent']}, "
          f"{site['half_size_km']:.1f} km, panel_frac={site['panel_frac']:.2f})")
    if not nc_path.exists():
        print(f"    SKIP: no stack at {nc_path.name}")
        return {"site": site, "status": "missing"}
    spectral, scl, dates = load_stack(nc_path)
    print(f"    stack: {spectral.shape[0]} scenes × 13 × {spectral.shape[2]}×{spectral.shape[3]}")
    t0 = time.time()
    composite, info = create_temporal_mosaic(spectral, scl)
    print(f"    mosaic: {info['n_scenes_used']} scenes used, "
          f"{info['fill_mode'].sum()*100/info['fill_mode'].size:.1f}% rescue fill, "
          f"{time.time()-t0:.1f}s")
    image_hwc = np.transpose(composite[:13], (1, 2, 0))
    mask, probs, thr = run_tiled_inference(image_hwc)
    rgb = make_rgb(composite)
    png_path = out_dir / f"realworld_{site['tier']}_{site['chip_id']}.png"
    render_quickview(site, rgb, mask, probs, thr, png_path)
    print(f"    -> {png_path.name}  ({int(mask.sum())} px detected)")
    return {"site": site, "status": "ok", "n_det": int(mask.sum()),
            "n_scenes": info["n_scenes_used"], "png": str(png_path)}


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["select", "download", "infer", "all"],
                        default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sites = select_sites()
    print(f"\nSelected {len(sites)} sites:")
    print(f"  {'chip_id':<22} {'tier':<7} {'continent':<10} {'panel_frac':>10}  {'AOI km':>7}  {'lat/lon':<22}")
    for s in sites:
        print(f"  {s['chip_id']:<22} {s['tier']:<7} {s['continent']:<10} "
              f"{s['panel_frac']:>10.3f}  {2*s['half_size_km']:>5.1f}    "
              f"{s['lat']:>+8.3f}, {s['lon']:>+8.3f}")

    if args.phase == "select":
        print("\n(select-only; not downloading)")
        return

    if args.phase in ("download", "all"):
        import openeo
        print(f"\nConnecting to {BACKEND_URL} (device-code auth if no cached token)...")
        conn = openeo.connect(BACKEND_URL)
        conn.authenticate_oidc_device()
        print("Authenticated.")
        print(f"\nSubmitting {len(sites)} OpenEO jobs (Feb-Apr 2026)...")
        pending = []
        for s in sites:
            res = download_stack(conn, s, OUTPUT_DIR)
            if isinstance(res, dict):
                pending.append(res)
        if pending:
            print(f"\nWaiting for {len(pending)} job(s) to finish...")
            wait_for_jobs(pending, OUTPUT_DIR)
        print("\nAll downloads complete.")

    if args.phase in ("infer", "all"):
        print("\nMosaic + inference + render:")
        results = []
        for s in sites:
            nc = OUTPUT_DIR / f"{s['chip_id']}_stack.nc"
            results.append(process_site(s, nc, OUTPUT_DIR))
        ok = [r for r in results if r.get("status") == "ok"]
        print(f"\n{len(ok)}/{len(sites)} sites processed. PNGs in {OUTPUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
