"""Register the solar PV detection pipeline as an OpenEO UDP.

This script registers the full detection pipeline (mosaic + inference) as
a User Defined Process on an OpenEO backend.  Once registered, anyone
with access to the backend can call it by name.

NOTE FOR APEX:
    This is a skeleton.  The process graph JSON in solar_pv_detection.json
    (which Apex creates) is the actual UDP definition.  This script wraps
    the registration API call.

Usage:
    python openeo_udp/process_graph/register_udp.py --backend https://openeo.cloud
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openeo


def register_udp(
    connection: openeo.Connection,
    process_id: str = "solar_pv_detection",
    description: str = (
        "Detect solar PV panels from Sentinel-2 imagery using a U-Net model. "
        "Builds a cloud-free temporal mosaic and runs pixel-level binary "
        "segmentation.  Returns a mask where 1 = solar panel."
    ),
):
    """Register the UDP on the connected backend.

    Parameters
    ----------
    connection : openeo.Connection
        Authenticated connection to an OpenEO backend.
    process_id : str
        Name for the UDP (how users will reference it).
    description : str
        Human-readable description shown in the backend's process catalogue.
    """
    # Load process graph from JSON (Apex to create this file)
    pg_path = Path(__file__).parent / "solar_pv_detection.json"
    if not pg_path.exists():
        raise FileNotFoundError(
            f"{pg_path} not found. Apex must create this file containing "
            "the validated process graph JSON."
        )

    with pg_path.open() as f:
        process_graph = json.load(f)

    # Define UDP parameters (exposed to users calling the UDP)
    parameters = [
        {
            "name": "spatial_extent",
            "description": "Bounding box (west, south, east, north) with CRS.",
            "schema": {"type": "object"},
        },
        {
            "name": "temporal_extent",
            "description": "Date range [start, end] for Sentinel-2 imagery.",
            "schema": {"type": "array", "items": {"type": "string"}},
        },
        {
            "name": "threshold",
            "description": "Detection confidence threshold (default 0.80).",
            "schema": {"type": "number"},
            "optional": True,
            "default": 0.80,
        },
    ]

    # Register
    connection.save_user_defined_process(
        user_defined_process_id=process_id,
        process_graph=process_graph,
        parameters=parameters,
        summary="Solar PV Panel Detection (U-Net)",
        description=description,
    )
    print(f"UDP '{process_id}' registered successfully.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True,
                        help="OpenEO backend URL (e.g. https://openeo.cloud)")
    parser.add_argument("--process-id", default="solar_pv_detection",
                        help="UDP name (default: solar_pv_detection)")
    args = parser.parse_args()

    connection = openeo.connect(args.backend)
    connection.authenticate_oidc()

    register_udp(connection, process_id=args.process_id)


if __name__ == "__main__":
    main()
