#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=geeConn
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO
set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base
python - <<'PY'
import time, ee
print("node:", __import__("socket").gethostname(), flush=True)
t0=time.time()
try:
    ee.Initialize()
    print(f"ee.Initialize OK in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"ee.Initialize FAILED: {e}", flush=True); raise SystemExit(1)
# Tiny getInfo to confirm compute reachability
t1=time.time()
val = ee.Number(41).add(1).getInfo()
print(f"getInfo roundtrip OK ({val}) in {time.time()-t1:.1f}s", flush=True)
# A real reduceRegions on 1 polygon to mimic the workload path
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-3.5,38.0]).buffer(80), {"sid":"t"})])
img = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
       .filterBounds(fc).filterDate("2023-06-01","2023-07-01")
       .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE",60)).median().select(["B2","B3","B4","B8"]))
t2=time.time()
res = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10).getInfo()
print(f"reduceRegions getInfo OK in {time.time()-t2:.1f}s, feats={len(res['features'])}", flush=True)
print("COMPUTE NODE CAN REACH GEE", flush=True)
PY
