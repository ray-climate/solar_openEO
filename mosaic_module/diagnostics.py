"""Diagnostic plotting for cloud masking and mosaic quality checks."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Optional

import ee
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from .cloud_mask import attach_cloud_prob, compute_clear_mask, load_sentinel_l1c
from .io_helpers import initialize_ee, sanitize_export_name
from .main import create_aoi
from .scene_scoring import score_scenes

LOGGER = logging.getLogger(__name__)

SCENE_PALETTE = [
    "1f77b4",
    "ff7f0e",
    "2ca02c",
    "d62728",
    "9467bd",
    "8c564b",
    "e377c2",
    "7f7f7f",
    "bcbd22",
    "17becf",
]


def _thumb_array(
    image: ee.Image,
    region: ee.Geometry,
    dimensions: int,
    crs: str = "EPSG:3857",
) -> np.ndarray:
    params = {
        "region": region,
        "dimensions": f"{dimensions}x{dimensions}",
        "format": "png",
        "crs": crs,
    }
    url = image.getThumbURL(params)
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Thumbnail request failed ({response.status_code}): {response.text[:400]}"
        )
    return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))


def _save_default_path(
    center_lat: float,
    center_lon: float,
    start_date: str,
    end_date: str,
) -> Path:
    stem = sanitize_export_name(
        f"diag_{start_date}_{end_date}_{center_lat:.5f}_{center_lon:.5f}"
    )
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.png"


def create_mosaic_diagnostic_plot(
    center_lat: float,
    center_lon: float,
    start_date: str,
    end_date: str,
    aoi_size_km: float = 1.0,
    out_scale: int = 10,
    top_n_scenes: int = 6,
    max_scene_cloud_pct: int | None = None,
    use_shadow_mask: bool = False,
    thumb_px: int = 900,
    figure_dpi: int = 220,
    out_png: Optional[str] = None,
) -> Dict[str, object]:
    """Create a diagnostic figure with per-date RGB/cloud masks + mosaic QA."""
    if aoi_size_km <= 0:
        raise ValueError("aoi_size_km must be > 0.")
    initialize_ee()

    aoi = create_aoi(
        center_lat=center_lat,
        center_lon=center_lon,
        half_size_km=aoi_size_km / 2.0,
    )

    s2 = load_sentinel_l1c(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_scene_cloud_pct=max_scene_cloud_pct,
    )
    s2 = attach_cloud_prob(s2)
    s2 = s2.map(lambda img: compute_clear_mask(img, aoi=aoi, use_shadow=use_shadow_mask))
    scored = score_scenes(s2, aoi=aoi, start_date=start_date, end_date=end_date)

    total_scene_count = int(scored.size().getInfo())
    if total_scene_count == 0:
        raise RuntimeError("No scenes found for diagnostics.")

    n = min(top_n_scenes, total_scene_count)
    candidates = scored.limit(n, "score", False)
    candidate_list = [ee.Image(candidates.toList(n).get(i)) for i in range(n)]

    quality_layers = []
    for idx, scene in enumerate(candidate_list):
        layer = (
            scene.select("clear_mask")
            .rename("score")
            .addBands(scene.select(["B4", "B3", "B2"]))
            .addBands(ee.Image.constant(idx).toInt16().rename("source_scene_id"))
        )
        quality_layers.append(layer.clip(aoi))

    best = ee.ImageCollection(quality_layers).qualityMosaic("score").clip(aoi)
    mosaic_rgb = best.select(["B4", "B3", "B2"])
    source_scene_id = best.select("source_scene_id").toInt16()
    assigned_valid = best.select("score").gt(0).rename("assigned_valid_mask")

    rows = n + 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4.8), dpi=figure_dpi)
    if rows == 1:
        axes = np.array([axes])

    for i, scene in enumerate(candidate_list):
        scene_date = ee.Date(scene.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        clear_fraction = float(ee.Number(scene.get("clear_fraction")).getInfo())

        rgb_vis = scene.select(["B4", "B3", "B2"]).visualize(
            min=200,
            max=3000,
            gamma=1.2,
        )
        cloud_vis = scene.select("cloud_shadow_mask").visualize(
            min=0,
            max=1,
            palette=["000000", "ff0000"],
        )
        clear_vis = scene.select("clear_mask").visualize(
            min=0,
            max=1,
            palette=["000000", "00ff00"],
        )

        axes[i, 0].imshow(_thumb_array(rgb_vis, aoi, thumb_px))
        axes[i, 1].imshow(_thumb_array(cloud_vis, aoi, thumb_px))
        axes[i, 2].imshow(_thumb_array(clear_vis, aoi, thumb_px))

        axes[i, 0].set_title(f"{scene_date} RGB")
        axes[i, 1].set_title(f"{scene_date} Cloud/Shadow")
        axes[i, 2].set_title(f"Clear mask (frac={clear_fraction:.3f})")

    mosaic_row = n
    mosaic_rgb_vis = mosaic_rgb.visualize(
        min=200,
        max=3000,
        gamma=1.2,
    )
    src_vis = source_scene_id.visualize(
        min=0,
        max=max(1, n - 1),
        palette=SCENE_PALETTE[: max(2, n)],
    )
    assigned_vis = assigned_valid.visualize(
        min=0,
        max=1,
        palette=["cc0000", "00aa00"],
    )

    axes[mosaic_row, 0].imshow(_thumb_array(mosaic_rgb_vis, aoi, thumb_px))
    axes[mosaic_row, 1].imshow(_thumb_array(src_vis, aoi, thumb_px))
    axes[mosaic_row, 2].imshow(_thumb_array(assigned_vis, aoi, thumb_px))
    axes[mosaic_row, 0].set_title("Final mosaic RGB")
    axes[mosaic_row, 1].set_title("source_scene_id map")
    axes[mosaic_row, 2].set_title("assigned_valid_mask")

    for r in range(rows):
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(
        (
            f"Temporal mosaic diagnostics (quick preview) | "
            f"AOI {aoi_size_km:.2f} km x {aoi_size_km:.2f} km\n"
            f"{start_date} to {end_date} | top_n={n}"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    out_path = Path(out_png) if out_png else _save_default_path(
        center_lat=center_lat,
        center_lon=center_lon,
        start_date=start_date,
        end_date=end_date,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved diagnostics plot: %s", out_path)
    return {
        "plot_path": str(out_path),
        "candidate_scene_count": n,
        "total_scene_count": total_scene_count,
        "max_scene_cloud_pct": max_scene_cloud_pct,
        "use_shadow_mask": use_shadow_mask,
        "thumb_px": thumb_px,
        "figure_dpi": figure_dpi,
    }
