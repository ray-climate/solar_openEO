"""Run diagnostics plotting for multiple locations."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mosaic_module.diagnostics import create_mosaic_diagnostic_plot

DEFAULT_LOCATIONS = [
    ("sahara_desert", 23.8000, 10.5000),
    ("atacama_desert", -23.1000, -68.2000),
    ("gobi_steppe", 42.7000, 103.8000),
    ("australia_outback", -25.0000, 133.0000),
    ("mongolia_steppe", 46.5000, 105.5000),
    ("scotland_highlands", 57.2000, -4.7000),
    ("patagonia_steppe", -49.0000, -72.0000),
    ("great_plains", 43.0000, -101.0000),
    ("namib_desert", -24.0000, 15.0000),
    ("alps_rural", 46.6000, 10.4000),
]


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "site"


def _parse_locations(raw: str) -> List[Tuple[str, float, float]]:
    """Parse `name:lat,lon;name2:lat,lon`."""
    out: List[Tuple[str, float, float]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name_part, coord_part = chunk.split(":")
        lat_str, lon_str = coord_part.split(",")
        out.append((_safe_name(name_part), float(lat_str), float(lon_str)))
    if not out:
        raise ValueError("No valid locations parsed.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch diagnostics over multiple locations.")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--aoi-km",
        type=float,
        default=3.0,
        help="AOI side length in km (default: 3.0)",
    )
    parser.add_argument("--top-n", type=int, default=6, help="Number of input scenes shown")
    parser.add_argument("--thumb-px", type=int, default=900, help="Thumb width/height in px")
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=220,
        help="Saved figure DPI",
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
    parser.add_argument(
        "--locations",
        type=str,
        default=None,
        help=(
            "Optional locations as 'name:lat,lon;name2:lat,lon'. "
            "If omitted, uses built-in defaults."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/diagnostics/batch",
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--gallery-name",
        type=str,
        default="index.html",
        help="HTML gallery filename written under out-dir",
    )
    return parser.parse_args()


def _write_gallery(
    out_dir: Path,
    gallery_name: str,
    start: str,
    end: str,
    aoi_km: float,
    items: List[Dict[str, str]],
) -> Path:
    cards = []
    for item in items:
        cards.append(
            (
                f"<article class='card'><h3>{item['name']}</h3>"
                f"<p>{item['meta']}</p>"
                f"<img src='{item['img']}' alt='{item['name']}' loading='lazy'/>"
                f"</article>"
            )
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Mosaic Diagnostics Gallery</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f7fb;color:#111;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;}"
        ".card{background:#fff;border:1px solid #d6d9e0;border-radius:10px;padding:12px;}"
        ".card h3{margin:0 0 6px 0;font-size:18px}"
        ".card p{margin:0 0 10px 0;font-size:13px;color:#444}"
        ".card img{width:100%;height:auto;border:1px solid #ddd;border-radius:6px}"
        "</style></head><body>"
        f"<h1>Mosaic Diagnostics Gallery</h1>"
        f"<p>Window: {start} to {end} | AOI: {aoi_km:.1f} km x {aoi_km:.1f} km</p>"
        f"<p>Locations: {len(items)}</p>"
        f"<section class='grid'>{''.join(cards)}</section>"
        "</body></html>"
    )
    out_path = out_dir / gallery_name
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    locations = (
        _parse_locations(args.locations) if args.locations else list(DEFAULT_LOCATIONS)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running diagnostics for {len(locations)} locations...")
    gallery_items: List[Dict[str, str]] = []
    for name, lat, lon in locations:
        out_png = out_dir / f"{name}_{args.start}_{args.end}_aoi{args.aoi_km:.1f}km.png"
        result = create_mosaic_diagnostic_plot(
            center_lat=lat,
            center_lon=lon,
            start_date=args.start,
            end_date=args.end,
            aoi_size_km=args.aoi_km,
            top_n_scenes=args.top_n,
            max_scene_cloud_pct=args.max_scene_cloud_pct,
            use_shadow_mask=args.use_shadow,
            thumb_px=args.thumb_px,
            figure_dpi=args.figure_dpi,
            out_png=str(out_png),
        )
        gallery_items.append(
            {
                "name": name,
                "img": out_png.name,
                "meta": (
                    f"lat={lat:.4f}, lon={lon:.4f} | "
                    f"total={result['total_scene_count']} shown={result['candidate_scene_count']}"
                ),
            }
        )
        print(
            f"[{name}] plot={result['plot_path']} "
            f"total={result['total_scene_count']} shown={result['candidate_scene_count']}"
        )

    gallery_path = _write_gallery(
        out_dir=out_dir,
        gallery_name=args.gallery_name,
        start=args.start,
        end=args.end,
        aoi_km=args.aoi_km,
        items=gallery_items,
    )
    print(f"Gallery: {gallery_path}")
    print("Batch diagnostics completed.")


if __name__ == "__main__":
    main()
