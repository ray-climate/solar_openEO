"""Smoke test runner for temporal mosaic v1."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mosaic_module import create_temporal_mosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal mosaic v1 export test.")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--aoi-km",
        type=float,
        default=10.0,
        help="AOI side length in km (e.g. 1 for 1x1 km)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Top candidate scenes for main assignment (default: 8)",
    )
    parser.add_argument(
        "--top-n-scenes-rescue",
        type=int,
        default=10,
        help="Candidate scenes for rescue fill (default: 10)",
    )
    parser.add_argument(
        "--clear-thresh",
        type=float,
        default=0.8,
        help="Cluster clear-fraction threshold (default: 0.8)",
    )
    parser.add_argument(
        "--max-scene-cloud-pct",
        type=int,
        default=None,
        help="Optional coarse scene cloud metadata filter (0-100). Default: none",
    )
    parser.add_argument(
        "--use-shadow",
        action="store_true",
        help="Enable projected shadow masking (default is cloud-only).",
    )
    parser.add_argument("--name", type=str, default=None, help="Optional export name")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    result = create_temporal_mosaic(
        center_lat=args.lat,
        center_lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        aoi_size_km=args.aoi_km,
        top_n_scenes=args.top_n,
        top_n_scenes_rescue=args.top_n_scenes_rescue,
        clear_thresh=args.clear_thresh,
        max_scene_cloud_pct=args.max_scene_cloud_pct,
        use_shadow_mask=args.use_shadow,
        export_name=args.name,
    )
    print("Export submitted.")
    print(f"Task ID: {result['task_id']}")
    print(f"State: {result['task_state']}")
    print(f"RGB Task ID: {result['rgb_task_id']}")
    print(f"RGB State: {result['rgb_task_state']}")
    print(f"Drive folder: {result['drive_folder']}")
    print(f"Export name: {result['export_name']}")
    print(f"AOI size (km): {result['aoi_size_km']}")
    print(f"Max scene cloud pct: {result['max_scene_cloud_pct']}")
    print(f"Use shadow mask: {result['use_shadow_mask']}")
    print(f"Candidate scenes: {result['candidate_scene_count']}")
    print(f"Rescue scenes: {result['rescue_scene_count']}")


if __name__ == "__main__":
    main()
