"""Build an HTML gallery from diagnostic PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostics HTML gallery.")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing diagnostic PNG files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="index.html",
        help="Output HTML filename (inside --dir)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mosaic Diagnostics Gallery",
        help="Gallery title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.dir)
    pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG files found in {out_dir}")

    cards = []
    for p in pngs:
        label = p.stem
        cards.append(
            (
                f"<article class='card'><h3>{label}</h3>"
                f"<img src='{p.name}' alt='{label}' loading='lazy'/>"
                f"</article>"
            )
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{args.title}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f7fb;color:#111;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;}"
        ".card{background:#fff;border:1px solid #d6d9e0;border-radius:10px;padding:12px;}"
        ".card h3{margin:0 0 10px 0;font-size:16px;line-height:1.2;word-break:break-word}"
        ".card img{width:100%;height:auto;border:1px solid #ddd;border-radius:6px}"
        "</style></head><body>"
        f"<h1>{args.title}</h1>"
        f"<p>Images: {len(pngs)}</p>"
        f"<section class='grid'>{''.join(cards)}</section>"
        "</body></html>"
    )

    out_path = out_dir / args.output
    out_path.write_text(html, encoding="utf-8")
    print(f"Gallery written: {out_path}")


if __name__ == "__main__":
    main()
