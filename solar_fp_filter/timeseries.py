"""Phase 2 — extract aggregate_spatial time series via CDSE OpenEO.

For each (sample_id, window_start, window_end) in the polygon GeoPackage,
pull the per-band mean reflectance per timestamp from Sentinel-2 L2A
after SCL-dilation cloud masking. Batched by shared temporal window so
each OpenEO job runs a single load_collection.

Outputs a long-format parquet at ``outputs/fp_classifier/timeseries/<batch_id>.parquet``
with one row per (sample_id, date, band).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import openeo
import pandas as pd
from shapely.geometry import box as shp_box, mapping

REPO = Path(__file__).resolve().parent.parent
POLYGONS_GPKG = REPO / "outputs/fp_classifier/polygons.gpkg"
TIMESERIES_DIR = REPO / "outputs/fp_classifier/timeseries"
BACKEND_URL = "https://openeo.dataspace.copernicus.eu"

L2A_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
             "B08", "B8A", "B11", "B12"]   # SCL is loaded separately for masking

# Geo-binning + batch sizing.
# aggregate_spatial scales well across many polygons in one job — each
# OpenEO batch job has fixed startup overhead (~5-10 min) regardless of
# how few polygons are in it, so we want batches as LARGE as possible
# subject to:
#   - bbox not so wide that the load_collection cost dominates
#   - polygon count fits comfortably in a single FeatureCollection
MAX_BBOX_DEG       = 15.0   # union bbox per batch (was 5° — too fragmented)
MIN_POLYS_PER_BATCH = 30    # merge geo-bins below this until we hit max
MAX_POLYS_PER_BATCH = 250   # split if a bin has more than this


def _connect() -> openeo.Connection:
    conn = openeo.connect(BACKEND_URL)
    conn.authenticate_oidc()
    return conn


def _build_process_graph(conn, bbox_4326: dict, window_start: str, window_end: str,
                         fc: dict) -> openeo.DataCube:
    """Build the OpenEO process graph for a single window batch."""
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox_4326,
        temporal_extent=[window_start, window_end],
        bands=L2A_BANDS + ["SCL"],
    )
    s2_masked = s2.process(
        "mask_scl_dilation",
        data=s2, scl_band_name="SCL",
        kernel1_size=3, kernel2_size=5,
    )
    # Drop SCL after masking (we don't want SCL in the output)
    s2_masked = s2_masked.filter_bands(bands=L2A_BANDS)
    ts = s2_masked.aggregate_spatial(geometries=fc, reducer="mean")
    return ts


def _sub_batch_polygons(group: pd.DataFrame,
                        max_deg: float = MAX_BBOX_DEG,
                        min_per_batch: int = MIN_POLYS_PER_BATCH,
                        max_per_batch: int = MAX_POLYS_PER_BATCH):
    """Yield sub-groups balancing bbox extent and polygon count.

    Strategy: bin by (max_deg)-cells of lat/lon first, then merge any
    cell with fewer than min_per_batch polygons into the next non-empty
    cell along longitude. Any cell exceeding max_per_batch is split
    further by latitude.
    """
    g = group.copy()
    g["lat_bin"] = (g["lat"] // max_deg).astype(int)
    g["lon_bin"] = (g["lon"] // max_deg).astype(int)
    bins = list(g.groupby(["lat_bin", "lon_bin"], sort=True))
    # Merge undersized bins by greedy accumulation
    merged: list[pd.DataFrame] = []
    buffer = []
    buffer_count = 0
    for _, sub in bins:
        buffer.append(sub)
        buffer_count += len(sub)
        if buffer_count >= min_per_batch:
            merged.append(pd.concat(buffer, ignore_index=True))
            buffer, buffer_count = [], 0
    if buffer:
        # Append leftover to the last merged group, or yield as its own
        if merged:
            merged[-1] = pd.concat([merged[-1], pd.concat(buffer, ignore_index=True)],
                                   ignore_index=True)
        else:
            merged.append(pd.concat(buffer, ignore_index=True))
    # Split oversized batches
    for sub in merged:
        if len(sub) <= max_per_batch:
            yield sub.drop(columns=["lat_bin", "lon_bin"], errors="ignore").reset_index(drop=True)
        else:
            n_chunks = (len(sub) + max_per_batch - 1) // max_per_batch
            for chunk in [sub.iloc[i::n_chunks] for i in range(n_chunks)]:
                yield chunk.drop(columns=["lat_bin", "lon_bin"], errors="ignore").reset_index(drop=True)


def _polygons_to_featurecollection(sub: pd.DataFrame) -> dict:
    features = []
    for _, r in sub.iterrows():
        features.append({
            "type": "Feature",
            "properties": {"sample_id": r["sample_id"]},
            "geometry": mapping(r["geometry"]),
        })
    return {"type": "FeatureCollection", "features": features}


def _bbox_4326(sub: pd.DataFrame, pad_deg: float = 0.02) -> dict:
    lon_min = sub["lon"].min() - pad_deg
    lon_max = sub["lon"].max() + pad_deg
    lat_min = sub["lat"].min() - pad_deg
    lat_max = sub["lat"].max() + pad_deg
    return {"west": lon_min, "east": lon_max,
            "south": lat_min, "north": lat_max,
            "crs": "EPSG:4326"}


def _parse_aggregate_result(result_path: Path, sub: pd.DataFrame) -> pd.DataFrame:
    """OpenEO aggregate_spatial JSON output: a nested dict
        {"<date_iso>": [[band1, band2, ...], ...]}
    where the outer list index matches the order of polygons in the FC.
    Reflectance values are in DN scale (0-10000), divide by 10000 for
    physical reflectance.
    """
    with result_path.open() as f:
        data = json.load(f)
    # Some backends return metadata at the top level — must reject anything
    # that isn't a date->list-of-lists mapping.
    sample_ids = sub["sample_id"].tolist()
    rows = []
    for key, per_geom in data.items():
        # Date keys look like "2018-01-03T00:00:00Z" or "2018-01-03"
        if not (isinstance(key, str) and (len(key) >= 10 and key[4] == "-")):
            continue
        if not isinstance(per_geom, list):
            continue
        for i, vals in enumerate(per_geom):
            if i >= len(sample_ids):
                continue
            if vals is None or not isinstance(vals, list):
                continue
            # Skip rows where all bands are NaN/null (fully cloud-masked)
            if all(v is None for v in vals):
                continue
            row = {"sample_id": sample_ids[i], "date": key}
            for j, band in enumerate(L2A_BANDS):
                v = vals[j] if j < len(vals) else None
                row[band] = v
            rows.append(row)
    return pd.DataFrame(rows)


def extract_for_polygons(polygons_gpkg: Path = POLYGONS_GPKG,
                         out_dir: Path = TIMESERIES_DIR,
                         smoke_test: bool = False,
                         smoke_n_per_class: int = 15,
                         max_concurrent_jobs: int = 4) -> None:
    """Drive the extraction. If smoke_test=True, only run on ~60 samples
    (smoke_n_per_class per (class × negative_source) combo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(polygons_gpkg)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    if smoke_test:
        # Pick a manageable subset: keep all (class × negative_source) combos,
        # restrict to mid-latitude Europe so the bbox is small per batch.
        gdf = gdf[(gdf["lat"].between(36, 50)) & (gdf["lon"].between(-10, 30))].copy()
        gdf["combo"] = (gdf["class"].astype(str) + "_"
                        + gdf["negative_source"].fillna("positive"))
        picks = []
        for combo, sub in gdf.groupby("combo"):
            picks.append(sub.head(smoke_n_per_class))
        gdf = pd.concat(picks, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        print(f"SMOKE TEST: {len(gdf)} samples", flush=True)
        print(gdf.groupby(["class", "negative_source"], dropna=False).size().to_string())

    conn = _connect()
    print(f"Connected to {BACKEND_URL}", flush=True)

    # Group by (window_start, window_end), sub-batch by geographic bbox
    window_groups = list(gdf.groupby(["window_start", "window_end"], sort=False))
    print(f"\nTotal window groups: {len(window_groups)}", flush=True)

    batches = []
    for (ws, we), grp in window_groups:
        for sub in _sub_batch_polygons(grp):
            batches.append((ws, we, sub))
    print(f"Total geo-binned batches: {len(batches)}", flush=True)

    # Process batches with bounded concurrency
    pending = list(enumerate(batches))
    in_flight: dict[str, dict] = {}
    n_done = 0; n_failed = 0
    t0 = time.time()

    while pending or in_flight:
        # Submit up to max_concurrent_jobs
        while pending and len(in_flight) < max_concurrent_jobs:
            idx, (ws, we, sub) = pending.pop(0)
            bbox = _bbox_4326(sub)
            fc = _polygons_to_featurecollection(sub)
            batch_id = f"batch_{idx:04d}_{ws[:7]}_to_{we[:7]}"
            out_pq = out_dir / f"{batch_id}.parquet"
            if out_pq.exists():
                print(f"  [skip-existing] {batch_id} ({len(sub)} polys)", flush=True)
                n_done += 1
                continue
            try:
                ts = _build_process_graph(conn, bbox, ws, we, fc)
                job = ts.save_result(format="JSON").create_job(
                    title=f"fp_filter_{batch_id}")
                job.start_job()
            except Exception as e:
                msg = str(e)[:160]
                pending.insert(0, (idx, (ws, we, sub)))
                print(f"  SUBMIT_FAIL {batch_id}: {msg}", flush=True)
                if "ConcurrentJobLimit" in msg or "429" in msg or "Too Many" in msg:
                    time.sleep(60)
                else:
                    time.sleep(15)
                break
            in_flight[batch_id] = {"job": job, "sub": sub, "out_pq": out_pq,
                                   "submitted_at": time.time()}
            elapsed = (time.time() - t0) / 60
            print(f"  [{elapsed:5.1f}m] submit {batch_id} bbox={bbox['west']:.1f},{bbox['south']:.1f}->"
                  f"{bbox['east']:.1f},{bbox['north']:.1f}  n={len(sub)}  "
                  f"in_flight={len(in_flight)} pending={len(pending)}", flush=True)

        # Poll in-flight jobs
        finished_ids = []
        for batch_id, info in list(in_flight.items()):
            try:
                status = info["job"].status()
            except Exception as e:
                print(f"  POLL_FAIL {batch_id}: {e}", flush=True); continue
            if status == "finished":
                tmp = out_dir / f"_tmp_{batch_id}"
                tmp.mkdir(parents=True, exist_ok=True)
                try:
                    info["job"].get_results().download_files(tmp)
                    # OpenEO writes both `timeseries.json` (the aggregate_spatial
                    # output) and `job-results.json` (metadata). Pick the right one.
                    jsons = [p for p in tmp.glob("*.json") if p.name != "job-results.json"]
                    if not jsons:
                        raise FileNotFoundError("no data JSON in result")
                    df = _parse_aggregate_result(jsons[0], info["sub"])
                    df.to_parquet(info["out_pq"], index=False)
                    n_done += 1
                    elapsed = (time.time() - t0) / 60
                    print(f"  [{elapsed:5.1f}m] FINISHED {batch_id}  "
                          f"rows={len(df):,}  dates={df['date'].nunique() if len(df) else 0}  "
                          f"[done={n_done}/{len(batches)}]", flush=True)
                except Exception as e:
                    n_failed += 1
                    print(f"  DOWNLOAD_FAIL {batch_id}: {e}", flush=True)
                finally:
                    for f in tmp.iterdir():
                        try: f.unlink()
                        except: pass
                    try: tmp.rmdir()
                    except: pass
                finished_ids.append(batch_id)
            elif status in ("error", "canceled"):
                n_failed += 1
                print(f"  FAILED {batch_id} -> {status}", flush=True)
                finished_ids.append(batch_id)
        for bid in finished_ids:
            del in_flight[bid]
        if in_flight or pending:
            time.sleep(20)

    print(f"\n=== Phase 2 complete ===  done={n_done}  failed={n_failed}", flush=True)
