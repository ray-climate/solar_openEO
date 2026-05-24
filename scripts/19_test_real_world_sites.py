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
V2_SPLIT_MANIFEST = REPO / "outputs/training_prep/final_v2/split_manifest.csv"
DEFAULT_OUTPUT_DIR = REPO / "docs/realworld_test_2026Q1"

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

def _load_v2_splits() -> dict[str, str]:
    splits = {}
    with open(V2_SPLIT_MANIFEST) as f:
        for r in csv.DictReader(f):
            splits[r["chip_id"]] = r["split"]
    return splits


def select_sites(n_total: int = 6, unseen_only: bool = False, seed: int = 42) -> list[dict]:
    """Pick stratified positive sites.

    Stratifies into small/medium/large by `panel_frac` and round-robins by
    continent within each tier for geographic diversity.

    Tier counts scale with `n_total` (default 6 → 2/2/2; 50 → 20/20/10).

    If `unseen_only`, restricts to chips whose v2 split is `val` or `test`
    (i.e. never trained on).  Honest real-world evaluation requires this.
    """
    import random
    rng = random.Random(seed)

    splits = _load_v2_splits() if unseen_only else {}

    rows = []
    with open(MASTER_MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["chip_type"] != "positive":
                continue
            if r["status"] != "EXTRACTED":
                continue
            cid = r["chip_id_str"]
            if unseen_only and splits.get(cid) not in ("val", "test"):
                continue
            try:
                pf = float(r["panel_frac"])
            except (TypeError, ValueError):
                continue
            if pf < 0.02:
                continue
            rows.append({
                "chip_id":    cid,
                "col":        int(r["chip_col"]),
                "row":        int(r["chip_row"]),
                "lat":        float(r["chip_center_lat"]),
                "lon":        float(r["chip_center_lon"]),
                "continent":  r["continent"],
                "panel_frac": pf,
                "split":      splits.get(cid, "?"),
            })

    # Stratify
    def in_tier(pf, tier):
        if tier == "small":  return 0.02 <= pf < 0.10
        if tier == "medium": return 0.10 <= pf < 0.40
        if tier == "large":  return pf >= 0.40
    # Tier counts scale with n_total (default 6 splits 2/2/2; 50 splits 20/20/10)
    if n_total <= 6:
        tier_counts = {"small": 2, "medium": 2, "large": 2}
        half_km = {"small": 2.5, "medium": 2.5, "large": 5.0}
    else:
        # weights: 0.4 / 0.4 / 0.2
        tier_counts = {
            "small":  int(round(n_total * 0.40)),
            "medium": int(round(n_total * 0.40)),
            "large":  n_total - int(round(n_total * 0.40)) - int(round(n_total * 0.40)),
        }
        half_km = {"small": 2.5, "medium": 2.5, "large": 5.0}

    sites = []
    for tier, n_pick in tier_counts.items():
        pool = [r for r in rows if in_tier(r["panel_frac"], tier)]
        rng.shuffle(pool)
        # group by continent, then round-robin pick for diversity
        by_cont: dict[str, list] = {}
        for r in pool:
            by_cont.setdefault(r["continent"], []).append(r)
        order = sorted(by_cont.keys(), key=lambda c: -len(by_cont[c]))
        chosen: list = []
        while len(chosen) < n_pick:
            progressed = False
            for c in order:
                if by_cont[c]:
                    chosen.append(by_cont[c].pop()); progressed = True
                    if len(chosen) >= n_pick:
                        break
            if not progressed:
                break
        for s in chosen[:n_pick]:
            sites.append({**s, "tier": tier, "half_size_km": half_km[tier]})
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

def _build_job(conn, site: dict):
    """Construct (but do not start) the OpenEO job for one site."""
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
    return merged.create_job(title=f"rw_{site['chip_id']}", out_format="netCDF")


def download_all_throttled(conn, sites: list[dict], out_dir: Path,
                           max_concurrent: int = 25,
                           poll_sec: int = 20) -> None:
    """Submit + download S2 stacks with a rolling window so we never exceed
    CDSE's concurrent-job limit (30 as of 2026-05).
    """
    pending = list(sites)            # not yet submitted
    in_flight: dict[str, dict] = {}  # chip_id -> {site, job, nc_path}
    started = time.time()
    n_done = 0; n_failed = 0; n_cached = 0

    while pending or in_flight:
        # Refill submissions up to max_concurrent
        while pending and len(in_flight) < max_concurrent:
            s = pending.pop(0)
            cid = s["chip_id"]
            nc_path = out_dir / f"{cid}_stack.nc"
            if nc_path.exists():
                print(f"  [cached] {cid} ({nc_path.stat().st_size/1e6:.1f} MB)")
                n_cached += 1
                continue
            try:
                job = _build_job(conn, s)
                job.start_job()
            except Exception as exc:
                print(f"  [{(time.time()-started)/60:5.1f} min] SUBMIT_FAIL {cid}: {exc}")
                n_failed += 1
                continue
            in_flight[cid] = {"site": s, "job": job, "nc_path": nc_path}
            print(f"  [{(time.time()-started)/60:5.1f} min] [submit] {cid} ({s['tier']:>6s}, {s['half_size_km']:.1f} km) "
                  f"-> {job.job_id}   (in_flight={len(in_flight)}, pending={len(pending)})")

        # Poll in_flight
        finished_cids = []
        for cid, p in list(in_flight.items()):
            try:
                status = p["job"].status()
            except Exception as exc:
                print(f"  [{(time.time()-started)/60:5.1f} min] POLL_FAIL {cid}: {exc}")
                continue
            if status == "finished":
                try:
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
                    n_done += 1
                    print(f"  [{(time.time()-started)/60:5.1f} min] FINISHED {cid} ({size_mb:.1f} MB)  "
                          f"[done={n_done}, failed={n_failed}, remaining={len(pending)+len(in_flight)-1}]")
                except Exception as exc:
                    print(f"  [{(time.time()-started)/60:5.1f} min] DOWNLOAD_FAIL {cid}: {exc}")
                    n_failed += 1
                finished_cids.append(cid)
            elif status in ("error", "canceled"):
                n_failed += 1
                print(f"  [{(time.time()-started)/60:5.1f} min] FAILED {cid} -> {status}")
                finished_cids.append(cid)
        for cid in finished_cids:
            del in_flight[cid]

        if pending or in_flight:
            time.sleep(poll_sec)
    print(f"\nDownload summary: {n_done} done, {n_failed} failed, {n_cached} cached, "
          f"total {n_done+n_failed+n_cached}")


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
    parser.add_argument("--n-sites", type=int, default=6, help="Total number of sites")
    parser.add_argument("--unseen-only", action="store_true",
                        help="Restrict picks to chips in v2 val/test (never trained on)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = select_sites(n_total=args.n_sites, unseen_only=args.unseen_only, seed=args.seed)
    label = "unseen (val/test only)" if args.unseen_only else "all positives"
    print(f"\nSelected {len(sites)} sites from {label}:")
    print(f"  {'chip_id':<22} {'tier':<7} {'split':<5} {'continent':<10} {'panel_frac':>10}  {'AOI km':>7}  lat/lon")
    for s in sites:
        print(f"  {s['chip_id']:<22} {s['tier']:<7} {s.get('split','?'):<5} {s['continent']:<10} "
              f"{s['panel_frac']:>10.3f}  {2*s['half_size_km']:>5.1f}    "
              f"{s['lat']:>+8.3f}, {s['lon']:>+8.3f}")

    if args.phase == "select":
        print("\n(select-only; not downloading)")
        return

    if args.phase in ("download", "all"):
        import openeo
        print(f"\nConnecting to {BACKEND_URL} (device-code auth if no cached token)...")
        conn = openeo.connect(BACKEND_URL)
        # authenticate_oidc() tries cached refresh token first, falls back to
        # device-code interactive (and stores the new refresh token on success).
        conn.authenticate_oidc()
        print("Authenticated.")
        print(f"\nSubmitting {len(sites)} OpenEO jobs (Feb-Apr 2026) "
              f"with rolling window (max_concurrent=25 to respect CDSE limit of 30)...")
        download_all_throttled(conn, sites, output_dir, max_concurrent=25)
        print("\nAll downloads complete.")

    if args.phase in ("infer", "all"):
        print("\nMosaic + inference + render:")
        results = []
        for s in sites:
            nc = output_dir / f"{s['chip_id']}_stack.nc"
            results.append(process_site(s, nc, output_dir))
        ok = [r for r in results if r.get("status") == "ok"]
        print(f"\n{len(ok)}/{len(sites)} sites processed. PNGs in {output_dir.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
