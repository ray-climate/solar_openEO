#!/usr/bin/env python
"""Convert docs/review_large/*.png to .jpg in-place to shrink the repo footprint.

Updates manifest.json so the HTML viewer points at .jpg files.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

OUT_DIR = Path("/gws/ssde/j25b/gbov/solar_openEO/docs/review_large")
MANIFEST = OUT_DIR / "manifest.json"
QUALITY = 88


def main() -> None:
    pngs = sorted(OUT_DIR.glob("*.png"))
    print(f"Converting {len(pngs)} PNG -> JPG (quality={QUALITY}) ...", flush=True)
    saved = 0
    for i, p in enumerate(pngs):
        jpg_path = p.with_suffix(".jpg")
        if jpg_path.exists():
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.save(jpg_path, "JPEG", quality=QUALITY, optimize=True)
        before = p.stat().st_size
        after = jpg_path.stat().st_size
        saved += (before - after)
        p.unlink()  # remove the PNG
        if (i + 1) % 100 == 0 or (i + 1) == len(pngs):
            print(f"  [{i+1:>4}/{len(pngs)}]  saved so far: {saved/1e6:.1f} MB", flush=True)

    # Update manifest paths
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        for chip in m["chips"]:
            if chip["png"].endswith(".png"):
                chip["png"] = chip["png"][:-4] + ".jpg"
        MANIFEST.write_text(json.dumps(m, indent=1))
        print(f"\nRewrote {MANIFEST.name}: chip references now use .jpg", flush=True)

    # Report
    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.jpg"))
    print(f"Total .jpg size: {total/1e6:.1f} MB  (saved ~{saved/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
