#%%
"""Run a batch test of the ONNX solar PV UDP.

Prerequisite: run generate_udp_onnx.py first to produce the UDP JSON.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd() / "openeo_udp" / "tests" / "test_udp_onnx.py"
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import openeo

from openeo_udp.process_graph.solar_pv_detection_onnx import DEFAULT_JOB_OPTIONS

# ---- edit here -------------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"

UDP_JSON = Path("openeo_udp/process_graph/solar_pv_detection_udp.json")
OUT_DIR = Path(__file__).resolve().parent / "test_outputs"
AOI = {
  "west": -120.15,
  "south": 35.33,
  "east": -119.98,
  "north": 35.45,
  "crs": "EPSG:4326"
}
TEMPORAL = ["2024-05-01", "2024-09-30"]
# ---------------------------------------------------------------------------

if not UDP_JSON.exists():
    raise FileNotFoundError(f"UDP JSON not found: {UDP_JSON} — run generate_udp_onnx.py first.")


conn = openeo.connect(BACKEND)
conn.authenticate_oidc()


cube = conn.datacube_from_json(
    str(UDP_JSON),
    parameters={
        "spatial_extent": AOI,
        "temporal_extent": TEMPORAL,
    },
)


job = cube.create_job(
    title="solar_pv_detection_onnx_test_mosaic",
    out_format="GTiff",
    job_options=DEFAULT_JOB_OPTIONS,
)
job.start_and_wait()
print(f"Job ID: {job.job_id}")

