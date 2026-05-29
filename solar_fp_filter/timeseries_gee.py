"""Phase 2 (GEE backend) — extract monthly reflectance time series per
polygon via Google Earth Engine reduceRegions.

Much faster than OpenEO aggregate_spatial for this workload: GEE reduces
server-side and returns scalars. For each polygon we get one row per
month per band over its labelled window.

Uses COPERNICUS/S2_HARMONIZED (L1C, same as the U-Net training data),
QA60 bitmask cloud masking, monthly median composites.

Output: one parquet per chunk at
``outputs/fp_classifier/timeseries/<window>_<chunk>.parquet``
long format (sample_id, month, B2..B12).
"""
from __future__ import annotations

import time
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
POLYGONS_GPKG = REPO / "outputs/fp_classifier/polygons.gpkg"
TIMESERIES_DIR = REPO / "outputs/fp_classifier/timeseries"

# S2_HARMONIZED band names (GEE). Maps to our canonical B02.. via rename.
GEE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
MAX_SCENE_CLOUD_PCT = 80   # coarse scene filter; QA60 handles per-pixel
# Chunk size is bounded by the REQUEST payload (10MB): the polygons' full
# geometries are embedded inline in the computation graph uploaded to GEE.
# These are complex MultiPolygons (~18 sub-polygons each); 150 stays ~5MB,
# 300 exceeds the cap. retainGeometry=False only shrinks the *response*.
CHUNK_SIZE = 150
GETINFO_RETRIES = 3


def _init():
    ee.Initialize()


def _mask_qa60(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return img.updateMask(mask)


def _n_months(window_start: str, window_end: str) -> int:
    y0, m0 = int(window_start[:4]), int(window_start[5:7])
    y1, m1 = int(window_end[:4]), int(window_end[5:7])
    return (y1 - y0) * 12 + (m1 - m0)


def _monthly_ic(fc: ee.FeatureCollection, window_start: str,
                n_months: int) -> ee.ImageCollection:
    months = ee.List.sequence(0, n_months - 1)
    base = ee.Date(window_start)

    # A fully-masked fallback image with the right band names, so months with
    # NO scenes still yield an image with bands (median() of an empty
    # collection has 0 bands -> reduceRegions throws "Image has no bands").
    # The fallback's values are masked, so reduceRegions returns null for it
    # and the parser skips those rows.
    empty = ee.Image.constant([0] * len(GEE_BANDS)).rename(GEE_BANDS) \
        .updateMask(ee.Image.constant(0)).toFloat()

    def make(m):
        start = base.advance(m, "month")
        end = start.advance(1, "month")
        coll = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(fc)
                .filterDate(start, end)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PCT))
                .map(_mask_qa60)
                .select(GEE_BANDS))
        comp = ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.median(), empty))
        return comp.rename(GEE_BANDS).set("month", start.format("YYYY-MM"))

    return ee.ImageCollection(months.map(make))


def _reduce_collection(fc: ee.FeatureCollection,
                       ic: ee.ImageCollection) -> ee.FeatureCollection:
    def per_img(img):
        reduced = img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=10)
        return reduced.map(lambda f: f.set("month", img.get("month")))
    flat = ic.map(per_img).flatten()
    # Drop geometry from output — reduceRegions echoes the full input polygon
    # (many vertices) in every row, blowing past the 10MB getInfo cap. We only
    # need the scalar band means. retainGeometry=False reliably strips it
    # (setGeometry(None) inside .map did not).
    props = ["sample_id", "month"] + GEE_BANDS
    return flat.select(propertySelectors=props, retainGeometry=False)


def _chunk_to_fc(sub: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, r in sub.iterrows():
        feats.append(ee.Feature(
            ee.Geometry(r["geometry"].__geo_interface__),
            {"sample_id": r["sample_id"]}))
    return ee.FeatureCollection(feats)


def _getinfo_with_retry(fc_result: ee.FeatureCollection) -> dict:
    last_exc = None
    for attempt in range(GETINFO_RETRIES):
        try:
            return fc_result.getInfo()
        except Exception as e:
            last_exc = e
            wait = 10 * (attempt + 1)
            print(f"    getInfo retry {attempt+1}/{GETINFO_RETRIES} after {wait}s: {str(e)[:100]}",
                  flush=True)
            time.sleep(wait)
    raise last_exc


def _parse_result(info: dict) -> pd.DataFrame:
    rows = []
    for feat in info.get("features", []):
        p = feat.get("properties", {})
        sid = p.get("sample_id")
        month = p.get("month")
        if sid is None or month is None:
            continue
        # Skip fully-masked (no band values) records
        if all(p.get(b) is None for b in GEE_BANDS):
            continue
        row = {"sample_id": sid, "month": month}
        for b in GEE_BANDS:
            row[b] = p.get(b)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Percentile extraction (idea #1): per-pixel NDVI/NDBI distribution inside
# each polygon, to recover the panel signal from diluted detection blobs.
# Panels = lowest-NDVI / highest-NDBI pixels, so the tails carry the signal.
# ---------------------------------------------------------------------------
TIMESERIES_PCT_DIR = REPO / "outputs/fp_classifier/timeseries_pct"
PCTS = [10, 25, 50, 75, 90]
GRID_DEG = 5   # 2D spatial blocking cell size for compact chunks


def _monthly_index_ic(fc, window_start, n_months):
    months = ee.List.sequence(0, n_months - 1)
    base = ee.Date(window_start)
    empty = ee.Image.constant([0, 0]).rename(["NDVI", "NDBI"]) \
        .updateMask(ee.Image.constant(0)).toFloat()

    def make(m):
        s = base.advance(m, "month"); e = s.advance(1, "month")
        coll = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(fc).filterDate(s, e)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PCT))
                .map(_mask_qa60).select(["B4", "B8", "B11"]))
        comp = coll.median()

        def idx(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
            return ndvi.addBands(ndbi)
        out = ee.Image(ee.Algorithms.If(coll.size().gt(0), idx(comp), empty))
        return out.rename(["NDVI", "NDBI"]).set("month", s.format("YYYY-MM"))

    return ee.ImageCollection(months.map(make))


def _reduce_percentiles(fc, ic):
    # Histogram-based percentile (bounded memory). The exact reducer holds
    # every pixel and blows GEE's memory limit on large polygons; capping
    # maxBuckets/maxRaw forces the histogram approximation, which is plenty
    # accurate for NDVI/NDBI percentiles and keeps memory bounded regardless
    # of polygon size.
    red = ee.Reducer.percentile(PCTS, maxBuckets=200, minBucketWidth=0.005,
                                maxRaw=1000)

    def per_img(img):
        r = img.reduceRegions(collection=fc, reducer=red, scale=10)
        return r.map(lambda f: f.set("month", img.get("month")))
    flat = ic.map(per_img).flatten()
    cols = ["sample_id", "month"] + \
           [f"NDVI_p{p}" for p in PCTS] + [f"NDBI_p{p}" for p in PCTS]
    return flat.select(propertySelectors=cols, retainGeometry=False)


def _parse_pct_result(info: dict) -> pd.DataFrame:
    pcols = [f"NDVI_p{p}" for p in PCTS] + [f"NDBI_p{p}" for p in PCTS]
    rows = []
    for feat in info.get("features", []):
        p = feat.get("properties", {})
        sid = p.get("sample_id"); month = p.get("month")
        if sid is None or month is None:
            continue
        if all(p.get(c) is None for c in pcols):
            continue
        row = {"sample_id": sid, "month": month}
        for c in pcols:
            row[c] = p.get(c)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_percentiles_gee(polygons_gpkg: Path = POLYGONS_GPKG,
                            out_dir: Path = TIMESERIES_PCT_DIR,
                            chunk_size: int = CHUNK_SIZE,
                            limit: int | None = None) -> None:
    """Per-pixel NDVI/NDBI percentiles per polygon-month (idea #1)."""
    _init()
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(polygons_gpkg)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    if limit:
        gdf = gdf.head(limit)
    print(f"Extracting PERCENTILE features for {len(gdf):,} samples via GEE", flush=True)
    window_groups = list(gdf.groupby(["window_start", "window_end"], sort=False))
    t0 = time.time(); n_chunks_done = 0; n_rows_total = 0
    for (ws, we), grp in window_groups:
        nm = _n_months(ws, we)
        # Spatially sort within the window so each chunk is geographically
        # COMPACT IN BOTH LAT AND LON. Scattered chunks force the monthly
        # composite to load S2 over a huge bbox -> GEE memory blowup. A
        # latitude-only sort still lets a dense band span all longitudes, so
        # we block by a coarse 2D grid (5deg cells): all polygons in a cell
        # are consecutive, bounding each chunk's span to ~one cell.
        grp = grp.copy()
        grp["_latb"] = (grp["lat"] // GRID_DEG).astype(int)
        grp["_lonb"] = (grp["lon"] // GRID_DEG).astype(int)
        grp = grp.sort_values(["_latb", "_lonb", "lat", "lon"]).reset_index(drop=True)
        n_chunks = (len(grp) + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            sub = grp.iloc[ci * chunk_size:(ci + 1) * chunk_size]
            tag = f"{ws[:7]}_to_{we[:7]}_chunk{ci:03d}"
            out_pq = out_dir / f"{tag}.parquet"
            if out_pq.exists():
                print(f"  [skip-existing] {tag}", flush=True); n_chunks_done += 1; continue
            fc = _chunk_to_fc(sub)
            ic = _monthly_index_ic(fc, ws, nm)
            result = _reduce_percentiles(fc, ic)
            t_chunk = time.time()
            try:
                info = _getinfo_with_retry(result)
            except Exception as e:
                print(f"  CHUNK_FAIL {tag}: {str(e)[:140]}", flush=True); continue
            df = _parse_pct_result(info)
            df.to_parquet(out_pq, index=False)
            n_chunks_done += 1; n_rows_total += len(df)
            print(f"  [{(time.time()-t0)/60:5.1f}m] {tag}: {len(sub)} polys x {nm}mo -> "
                  f"{len(df):,} rows in {time.time()-t_chunk:.1f}s [chunks={n_chunks_done}]", flush=True)
    print(f"\n=== Percentile extraction complete === chunks={n_chunks_done} "
          f"rows={n_rows_total:,} elapsed={(time.time()-t0)/60:.1f}m", flush=True)


def extract_for_polygons_gee(polygons_gpkg: Path = POLYGONS_GPKG,
                             out_dir: Path = TIMESERIES_DIR,
                             chunk_size: int = CHUNK_SIZE,
                             limit: int | None = None) -> None:
    """Extract monthly time series for all polygons. If limit is set,
    only process the first ``limit`` samples (for piloting)."""
    _init()
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(polygons_gpkg)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    if limit:
        gdf = gdf.head(limit)
    print(f"Extracting {len(gdf):,} samples via GEE", flush=True)

    window_groups = list(gdf.groupby(["window_start", "window_end"], sort=False))
    print(f"Window groups: {len(window_groups)}", flush=True)

    t0 = time.time()
    n_chunks_done = 0
    n_rows_total = 0
    for (ws, we), grp in window_groups:
        nm = _n_months(ws, we)
        grp = grp.reset_index(drop=True)
        n_chunks = (len(grp) + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            sub = grp.iloc[ci * chunk_size:(ci + 1) * chunk_size]
            tag = f"{ws[:7]}_to_{we[:7]}_chunk{ci:03d}"
            out_pq = out_dir / f"{tag}.parquet"
            if out_pq.exists():
                print(f"  [skip-existing] {tag}", flush=True)
                n_chunks_done += 1
                continue
            fc = _chunk_to_fc(sub)
            ic = _monthly_ic(fc, ws, nm)
            result = _reduce_collection(fc, ic)
            t_chunk = time.time()
            try:
                info = _getinfo_with_retry(result)
            except Exception as e:
                print(f"  CHUNK_FAIL {tag}: {str(e)[:140]}", flush=True)
                continue
            df = _parse_result(info)
            df.to_parquet(out_pq, index=False)
            n_chunks_done += 1
            n_rows_total += len(df)
            elapsed = (time.time() - t0) / 60
            print(f"  [{elapsed:5.1f}m] {tag}: {len(sub)} polys x {nm}mo -> "
                  f"{len(df):,} rows in {time.time()-t_chunk:.1f}s  "
                  f"[chunks={n_chunks_done}]", flush=True)

    print(f"\n=== GEE extraction complete ===  chunks={n_chunks_done}  "
          f"rows={n_rows_total:,}  elapsed={(time.time()-t0)/60:.1f}m", flush=True)
