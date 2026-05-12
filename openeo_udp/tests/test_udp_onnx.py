#%%
"""Run a batch test of the ONNX solar PV UDP.

Prerequisite: run generate_udp_onnx.py first to produce the UDP JSON.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openeo

from openeo_udp.process_graph.solar_pv_detection_onnx import DEFAULT_JOB_OPTIONS

# ---- edit here -------------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
UDP_JSON = Path(__file__).resolve().parents[1] / "process_graph" / "solar_pv_detection_onnx_udp.json"
OUT_DIR = Path(__file__).resolve().parent / "test_outputs"
AOI = {"west": -1.85, "south": 51.54, "east": -1.82, "north": 51.56, "crs": "EPSG:4326"}
TEMPORAL = ["2024-05-01", "2024-07-31"]
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
    title="solar_pv_detection_onnx_test",
    out_format="GTiff",
    job_options=DEFAULT_JOB_OPTIONS,
)
job.start_and_wait()
print(f"Job ID: {job.job_id}")

