# Solar PV Detection — Deployment Guide for Apex

This guide walks through deploying the solar PV detection model as an
OpenEO User Defined Process (UDP).

## Folder Overview

```
openeo_udp/
├── model_registry.yaml          ← Model config (RS-Oxford manages)
├── udf/
│   ├── solar_pv_inference.py    ← Inference UDF (RS-Oxford manages)
│   └── requirements.txt         ← Python dependencies for UDF runtime
├── process_graph/
│   ├── mosaic_prototype.py      ← Starting point for mosaic (Apex adapts)
│   ├── register_udp.py          ← UDP registration script (Apex finalises)
│   └── solar_pv_detection.json  ← Process graph JSON (Apex creates)
├── tests/
│   ├── test_udf_local.py        ← Local UDF verification (run first)
│   └── test_udp_smoke.py        ← End-to-end smoke test (Apex writes)
├── export/
│   └── export_weights.py        ← Model packaging script (RS-Oxford)
└── docs/
    ├── inference_spec.md         ← Input/output contract
    ├── mosaic_spec.md            ← Mosaic algorithm specification
    └── deployment_guide.md      ← This file
```

## IMPORTANT: TensorFlow Runtime Requirement

The inference UDF requires **TensorFlow >= 2.15** in the OpenEO backend's
UDF runtime.  As of April 2026, CDSE's default Python UDF runtime does
**not** include TensorFlow (only numpy, scipy, xarray, rasterio).

Options for Apex to resolve this:
1. **Request TF in backend runtime** — ask CDSE/backend provider to add it
2. **Custom Docker runtime** — build a UDF container image with TF included
3. **Convert to ONNX** — convert the Keras model to ONNX format for
   numpy-only inference via `onnxruntime` (lighter, more portable)

The temporal mosaic UDF (`temporal_mosaic.py`) only needs numpy + scipy
and runs on any backend without modification.

## Prerequisites

- Access to an OpenEO backend with:
  - Sentinel-2 L1C (or L2A) collection
  - UDF runtime supporting TensorFlow >= 2.15 (see note above)
  - Sufficient memory for model inference (~2 GB per UDF worker)
- Python 3.10+ with `openeo` client library
- GitHub access to download model weights (~1 GB)

## Step-by-Step Deployment

### 1. Verify the UDF Locally

Before touching any backend, verify the UDF works:

```bash
cd /path/to/solar_openEO
python openeo_udp/tests/test_udf_local.py --n-chips 10 --verbose
```

Expected output: all checks pass, Dice > 0.3 on random sample.

This confirms:
- model_registry.yaml is parseable
- Model weights load correctly (using local fallback paths)
- Band stats load and normalisation works
- Inference produces valid binary masks

### 2. Ensure Model Weights Are Accessible

The UDF downloads model weights from a URL at runtime.  Check that the
URLs in `model_registry.yaml` are accessible from the backend:

```yaml
weights_url: https://github.com/ray-climate/solar_openEO/releases/download/v1.0/best.weights.h5
band_stats_url: https://github.com/ray-climate/solar_openEO/releases/download/v1.0/band_stats.npz
```

If the backend cannot reach GitHub (firewall), alternatives:
- Host on the backend's object storage
- Bundle weights into a Docker image
- Pass URLs via `context` dict in `run_udf()`

### 3. Implement the Mosaic Process Graph

Start from `process_graph/mosaic_prototype.py` and adapt:

**What to verify on your backend:**
- [ ] Correct collection ID for Sentinel-2 L1C
- [ ] Band names (B01 vs B1 vs band_01)
- [ ] Cloud masking approach (SCL, s2cloudless, or custom)
- [ ] Whether `mask_scl_dilation` is supported
- [ ] Temporal reduction support (median, min, custom reducers)
- [ ] Spatial resampling to EPSG:3857 at 10 m

**Key constraint:** The output must be a 13-band datacube matching the
band order in `inference_spec.md`.  Incorrect band order will cause
silent accuracy degradation.

### 4. Wire the Full Process Graph

Connect mosaic → UDF inference:

```python
mosaic = create_temporal_mosaic(connection, bbox, dates)

result = mosaic.apply_neighborhood(
    process=openeo.UDF(udf_code, runtime="Python"),
    size=[
        {"dimension": "x", "value": 256, "unit": "px"},
        {"dimension": "y", "value": 256, "unit": "px"},
    ],
    overlap=[
        {"dimension": "x", "value": 0, "unit": "px"},
        {"dimension": "y", "value": 0, "unit": "px"},
    ],
    context={"threshold": 0.80},
)
```

Note: `apply_neighborhood` tiles the AOI into 256×256 px chunks and
passes each to the UDF.  No overlap is needed (the model was trained
on non-overlapping chips).

### 5. Export Process Graph JSON

Save the validated process graph:

```python
import json
pg = result.to_json()
with open("openeo_udp/process_graph/solar_pv_detection.json", "w") as f:
    json.dump(pg, f, indent=2)
```

### 6. Register the UDP

```bash
python openeo_udp/process_graph/register_udp.py --backend https://openeo.cloud
```

### 7. Smoke Test

Run on a small AOI (< 5 km²) with known solar installations:

```python
result = connection.datacube_from_process("solar_pv_detection",
    spatial_extent={"west": -1.3, "south": 51.74, "east": -1.25, "north": 51.77},
    temporal_extent=["2024-05-01", "2024-07-31"],
    threshold=0.80,
)
result.download("test_output.tif")
```

Check:
- [ ] Output is a valid GeoTIFF
- [ ] Values are 0 and 1 (binary)
- [ ] CRS is EPSG:3857
- [ ] Resolution is 10 m
- [ ] Known solar farms are detected
- [ ] Non-solar areas are mostly 0

## Ownership & Update Protocol

| Component              | Owner      | When to change                   |
|------------------------|------------|----------------------------------|
| model_registry.yaml    | RS-Oxford  | New model version trained        |
| udf/                   | RS-Oxford  | Architecture changes (rare)      |
| process_graph/*.json   | Apex       | Backend changes, bug fixes       |
| process_graph/*.py     | Apex       | Backend changes, optimisation    |
| docs/                  | Joint      | Keep in sync with changes        |

## Troubleshooting

**UDF fails with ImportError (tensorflow):**
Check that the backend's UDF runtime includes TF >= 2.15.  Some backends
only support lightweight runtimes.  Contact the backend provider.

**Model weights download timeout:**
The weights file is ~1 GB.  Increase the download timeout or pre-stage
weights on backend storage.

**All predictions are zero:**
- Check band order (most likely cause)
- Check that input values are in raw L1C DN range (0–10000+)
- Verify band_stats.npz matches the training data

**Low accuracy on new regions:**
The model was trained on European solar installations.  Performance may
degrade on:
- Non-European solar farm designs (e.g., concentrated solar power)
- Very large utility-scale installations (larger than 256×256 chip)
- Desert regions (sand/solar confusion)
- Snow-covered regions
