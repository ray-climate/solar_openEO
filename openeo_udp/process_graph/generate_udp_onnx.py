#%%
"""Generate ONNX solar PV UDP JSON.

Edit the constants below, then run:
    python openeo_udp/process_graph/generate_udp_onnx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

from openeo_udp.process_graph.solar_pv_detection_onnx import (
    DEFAULT_JOB_OPTIONS,
    build_solar_pv_detection_onnx,
)

# ---- edit here -------------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
PROCESS_ID = "solar_pv_detection"
OUTPUT_JSON = Path(__file__).resolve().parent / "solar_pv_detection_udp.json"
REGISTER = True
# ---------------------------------------------------------------------------

conn = openeo.connect(BACKEND)
conn.authenticate_oidc()

spatial_extent = Parameter(
    name="spatial_extent",
    description="Bounding box dict with west/south/east/north.",
    schema={"type": "object"},
)
temporal_extent = Parameter(
    name="temporal_extent",
    description="Date range [start, end] (ISO-8601).",
    schema={"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
)

cube = build_solar_pv_detection_onnx(
    connection=conn,
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
)

udp = build_process_dict(
    process_graph=cube,
    process_id=PROCESS_ID,
    summary="Solar PV detection (ONNX)",
    description=(
        "Sentinel-2 L1C temporal mosaic + ONNX UDF inference. "
        "Returns 'solar_pv' (binary) and 'solar_pv_probability' (float)."
    ),
    parameters=[spatial_extent, temporal_extent],
    default_job_options=DEFAULT_JOB_OPTIONS,
)

OUTPUT_JSON.write_text(json.dumps(udp, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT_JSON}")

if REGISTER:
    conn.save_user_defined_process(
        user_defined_process_id=PROCESS_ID,
        process_graph=udp["process_graph"],
        parameters=udp.get("parameters", []),
        summary=udp.get("summary"),
        description=udp.get("description"),
    )
    print(f"Registered UDP: {PROCESS_ID}")

# %%
