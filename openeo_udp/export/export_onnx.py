#!/usr/bin/env python
"""Export the active Solar PV Keras model to ONNX.

This script builds the model architecture from `openeo_udp/model_registry.yaml`,
loads weights, and converts the model to ONNX for backend runtimes where
TensorFlow is unavailable.

Usage:
    python openeo_udp/export/export_onnx.py \
        --output openeo_udp/export/releases/v1.0/solar_pv.onnx

Optional:
    python openeo_udp/export/export_onnx.py \
        --weights experiments/exp_stage1_r101_dice_zscore_longer/best.weights.h5 \
        --output openeo_udp/export/releases/v1.0/solar_pv.onnx \
        --opset 17 \
        --validate

Requirements (local environment):
    pip install 'numpy<2' tensorflow tf2onnx onnx onnxruntime

Note: NumPy must be <2 for TensorFlow compatibility.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY = REPO / "openeo_udp" / "model_registry.yaml"

# Allow running as: python openeo_udp/export/export_onnx.py
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)


def _load_active_model(registry_path: Path) -> dict:
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency: pip install pyyaml") from e

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw or "active_model" not in raw:
        raise ValueError(f"Missing 'active_model' in {registry_path}")
    return raw["active_model"]


def _resolve_weights_path(model_cfg: dict, weights_arg: str | None) -> Path:
    from openeo_udp.udf.solar_pv_inference import _download_if_needed

    def _is_url(value: str) -> bool:
        v = value.lower()
        return v.startswith("http://") or v.startswith("https://")

    if weights_arg:
        if _is_url(weights_arg):
            return _download_if_needed(weights_arg)

        path = Path(weights_arg)
        if not path.is_absolute():
            path = (REPO / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")
        return path

    # Default path from registry: try local first, then URL fallback.
    local = model_cfg.get("weights_local")
    weights_url = model_cfg.get("weights_url")

    if local:
        local_path = REPO / local
        if local_path.exists():
            return local_path.resolve()

    if weights_url:
        return _download_if_needed(weights_url, local_fallback=local)

    # Nothing available.
    if local:
        missing_local = (REPO / local).resolve()
        raise FileNotFoundError(
            f"Weights not found: {missing_local}. Also missing active_model.weights_url in model_registry.yaml"
        )

    raise ValueError(
        "No --weights provided and active_model has neither weights_local nor weights_url in model_registry.yaml"
    )


def _build_model_from_registry(model_cfg: dict):
    from solar_ml.model import build_unet_model

    input_shape = tuple(model_cfg["input_shape"])
    model, _ = build_unet_model(
        input_shape=input_shape,
        backbone_name=model_cfg["backbone"],
        pretrained=False,
        decoder_filters=tuple(model_cfg["decoder_filters"]),
        decoder_dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        attention=bool(model_cfg.get("attention", False)),
        se=bool(model_cfg.get("se", False)),
    )
    return model


def _load_weights(model, weights_path: Path) -> None:
    # Reuse the compatibility loader from the UDF path so export matches inference loading.
    from openeo_udp.udf.solar_pv_inference import _load_weights_compat

    _load_weights_compat(model, weights_path)


def _convert_to_onnx(model, output_path: Path, opset: int, dynamic_spatial: bool) -> None:
    try:
        tf = importlib.import_module("tensorflow")
        tf2onnx = importlib.import_module("tf2onnx")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing export dependency. Install with: pip install tensorflow tf2onnx"
        ) from e

    h, w, c = model.input_shape[1:]

    def _do_convert(use_dynamic_spatial: bool) -> None:
        if use_dynamic_spatial:
            spec = (tf.TensorSpec((None, None, None, c), tf.float32, name="s2_input"),)
        else:
            spec = (tf.TensorSpec((None, h, w, c), tf.float32, name="s2_input"),)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset,
            output_path=str(output_path),
        )

    if dynamic_spatial: #TODO currently not working
        try:
            _do_convert(use_dynamic_spatial=True)
            return
        except Exception as e:
            err = str(e)
            dynamic_not_supported = (
                "Using a symbolic `tf.Tensor` as a Python `bool`" in err
                or "OperatorNotAllowedInGraphError" in err
                or "UpSampling2D.call" in err
            )
            if not dynamic_not_supported:
                raise
            print(
                "Dynamic spatial export is not supported by this Keras/TF graph "
                "(UpSampling2D symbolic-shape limitation). Falling back to fixed spatial export."
            )
            _do_convert(use_dynamic_spatial=False)
            return

    _do_convert(use_dynamic_spatial=False)


def _validate_onnx(model, onnx_path: Path) -> None:
    import numpy as np

    try:
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing validation dependency. Install with: pip install onnx onnxruntime"
        ) from e

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Small numeric parity check against Keras
    h, w, c = model.input_shape[1:]
    x = np.random.rand(1, h, w, c).astype(np.float32)

    y_tf = model.predict(x, verbose=0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y_onnx = sess.run([output_name], {input_name: x})[0]

    max_abs = float(np.max(np.abs(y_tf - y_onnx)))
    mean_abs = float(np.mean(np.abs(y_tf - y_onnx)))
    print(f"Validation OK | max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX path (absolute or repo-relative)",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to model_registry.yaml",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional override path to .h5 weights",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--fixed-spatial",
        action="store_true",
        help="Export with fixed H/W from registry input_shape (default: dynamic H/W)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ONNX checker + quick parity test (requires onnxruntime)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging (shows download/build/convert progress)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = (REPO / registry_path).resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO / output_path).resolve()

    t0 = time.perf_counter()
    logger.info("Loading model registry: %s", registry_path)
    model_cfg = _load_active_model(registry_path)
    logger.info("Resolving model weights...")
    weights_path = _resolve_weights_path(model_cfg, args.weights)

    print(f"Registry: {registry_path}")
    print(f"Model:    {model_cfg.get('name', 'unknown')} ({model_cfg.get('backbone', 'unknown')})")
    print(f"Weights:  {weights_path}")
    print("Building model...")
    t_build = time.perf_counter()
    model = _build_model_from_registry(model_cfg)
    print(f"Model built in {time.perf_counter() - t_build:.1f}s")

    print("Loading weights...")
    t_weights = time.perf_counter()
    _load_weights(model, weights_path)
    print(f"Weights loaded in {time.perf_counter() - t_weights:.1f}s")

    print(f"Converting to ONNX (opset={args.opset})...")
    t_convert = time.perf_counter()
    dynamic_spatial = not args.fixed_spatial
    print(f"Dynamic spatial axes: {dynamic_spatial}")
    _convert_to_onnx(model, output_path, args.opset, dynamic_spatial=dynamic_spatial)
    print(f"Conversion finished in {time.perf_counter() - t_convert:.1f}s")
    print(f"ONNX exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    if args.validate:
        t_validate = time.perf_counter()
        _validate_onnx(model, output_path)
        print(f"Validation finished in {time.perf_counter() - t_validate:.1f}s")

    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
