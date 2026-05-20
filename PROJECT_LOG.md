# Solar OpenEO Project Log

## Purpose
Short running memory for quick session resume.

## Current Status (2026-05-20)
- ONNX openEO inference on CDSE runs through a single merged UDF:
  - `openeo_udp/udf/solar_pv_detection.py` — SLIC temporal mosaic + ONNX U-Net inference in one `apply_neighborhood` call.
  - `openeo_udp/process_graph/solar_pv_detection_onnx.py` — loads L1C + SCL, merges to a 14-band cube, runs the merged UDF.
- Normalization mirrors `solar_ml.data.normalize_batch` (percentile mode) using `band_stats.npz` from the model archive.
- Mosaic logic mirrors the GEE SNIC training pipeline: SCL clear mask → scene scoring → SLIC superpixels → per-cluster best-scene assignment → hierarchical fallback → rescue fill.
- Output cube: 2 bands (`solar_pv`, `solar_pv_probability`). Names set both via UDF `apply_metadata` and a `rename_labels` step in the process graph.

## Active Defaults
- Detection threshold: 0.80
- Neighborhood geometry: `size=192`, `overlap=32` (effective input = 256×256)
- Mosaic params: `snic_size_px=20`, `snic_compactness=1.0`, `clear_thresh=0.8`, `top_n_scenes=8`, `top_n_rescue=10`
- Job options: `executor-memory=6g`, `executor-memoryOverhead=3g`
- UDF dependency archives:
  - `onnx_deps_python311.zip#onnx_deps`
  - `solar_pv_rui.zip#onnx_models`

## Latest Relevant Files
- `openeo_udp/udf/solar_pv_detection.py`
- `openeo_udp/udf/temporal_mosaic.py` (standalone mosaic UDF, kept for `openeo_udp/tests/test_temporal_mosaic.py`)
- `openeo_udp/process_graph/solar_pv_detection_onnx.py`
- `openeo_udp/process_graph/generate_udp_onnx.py`
- `openeo_udp/tests/test_udp_onnx.py`

## Immediate Next Steps
1. Run a single-site AOI test and validate prediction distribution / footprint alignment.
2. Calibrate threshold on a small labeled validation set if needed.
3. Profile per-chunk runtime to decide whether to bump executor memory or chunk size.
