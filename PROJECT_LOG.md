# Solar OpenEO Project Log

## Purpose
Short running memory for quick session resume.

## Current Status (2026-05-12)
- ONNX openEO inference is active through:
  - openeo_udp/udf/solar_pv_inference_onnx.py
  - openeo_udp/process_graph/solar_pv_detection_onnx.py
- Current UDF debug outputs: solar_pv, solar_pv_probability, pre_norm_mean, post_norm_mean.
- Optional global normalization is supported: pass `global_mean`/`global_std` (13 values) in UDF context.
- New helper supports dynamic ROI stats calculation for concrete AOI runs: `auto_global_stats=True` in `build_solar_pv_detection_onnx(...)`.
- 2026-05-13 debug update: probability smoothing selects UDF outputs by positional band index (not by label) to avoid metadata mismatch where apply_neighborhood still reports source labels (B01..B12).
- UDF entrypoint/signature issue is resolved (DataArray-based entrypoint).
- Model output is validated as sigmoid probability in [0,1].
- Input normalization now uses per-chip robust histogram alignment (p5�p95 trimmed stats).
- Neighborhood geometry keeps model input at 256x256:
  - size=192, overlap=32 per axis (192 + 2*32 = 256).
- Post-inference Gaussian smoothing is applied on probability via apply_kernel:
  - defaults: kernel size 9, sigma 2.0
  - binary band is re-thresholded from smoothed probabilities.

## Active Defaults
- Threshold: 0.80
- Smoothing: disabled (2026-05-13 debug simplification)
- Export target: Google Drive with explicit folder names

## Latest Relevant Files
- openeo_udp/process_graph/solar_pv_detection_onnx.py
- openeo_udp/udf/solar_pv_inference_onnx.py
- openeo_udp/process_graph/generate_udp_onnx.py
- openeo_udp/tests/test_udp_onnx.py

## Immediate Next Steps
1. Run a single-site AOI test and verify binary/probability outputs.
2. Compare seam artifacts before/after smoothing.
3. If needed, calibrate threshold on a small labeled validation set.
