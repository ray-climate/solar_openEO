# Solar OpenEO Project Log

## Purpose
Short running memory for quick session resume.

## Current Status (2026-05-12)
- ONNX openEO inference is active through:
  - openeo_udp/udf/solar_pv_inference_onnx.py
  - openeo_udp/process_graph/solar_pv_detection_onnx.py
- UDF entrypoint/signature issue is resolved (DataArray-based entrypoint).
- Model output is validated as sigmoid probability in [0,1].
- Input normalization now uses per-chip robust histogram alignment (p5–p95 trimmed stats).
- Neighborhood geometry keeps model input at 256x256:
  - size=192, overlap=32 per axis (192 + 2*32 = 256).
- Post-inference Gaussian smoothing is applied on probability via apply_kernel:
  - defaults: kernel size 9, sigma 2.0
  - binary band is re-thresholded from smoothed probabilities.

## Active Defaults
- Threshold: 0.80
- Smoothing: kernel_size=9, sigma=2.0
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
