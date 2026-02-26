"""Create a diagnostics plot for cloud masking and mosaicing quality."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mosaic_module.diagnostics import create_mosaic_diagnostic_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporal mosaic diagnostics.")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--aoi-km",
        type=float,
        default=1.0,
        help="AOI side length in km (default: 1.0)",
    )
    parser.add_argument("--top-n", type=int, default=6, help="Number of input scenes shown")
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
    parser.add_argument("--thumb-px", type=int, default=900, help="Thumb width/height in px")
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=220,
        help="Saved figure DPI",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional output PNG path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    result = create_mosaic_diagnostic_plot(
        center_lat=args.lat,
        center_lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        aoi_size_km=args.aoi_km,
        top_n_scenes=args.top_n,
        max_scene_cloud_pct=args.max_scene_cloud_pct,
        use_shadow_mask=args.use_shadow,
        thumb_px=args.thumb_px,
        figure_dpi=args.figure_dpi,
        out_png=args.out,
    )

    print("Diagnostics plot created.")
    print(f"Plot path: {result['plot_path']}")
    print(f"Total scenes in window: {result['total_scene_count']}")
    print(f"Candidate scenes shown: {result['candidate_scene_count']}")
    print(f"Max scene cloud pct: {result['max_scene_cloud_pct']}")
    print(f"Use shadow mask: {result['use_shadow_mask']}")
    print(f"Thumb px: {result['thumb_px']}")
    print(f"Figure dpi: {result['figure_dpi']}")


if __name__ == "__main__":
    main()
