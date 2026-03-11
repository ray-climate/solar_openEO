"""Plot all individual Sentinel-2 scenes + cloud masks for a chip AOI.

Usage:
  conda run -n geospatial python scripts/plot_chip_scenes.py --chip-id c+000572_r+002106
  conda run -n geospatial python scripts/plot_chip_scenes.py --chip-id c+000080_r+002449
"""

import argparse
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.tiling import chip_id_to_bounds
from mosaic_module.cloud_mask import attach_cloud_prob, compute_clear_mask, load_sentinel_l1c
from mosaic_module.io_helpers import initialize_ee
from mosaic_module.scene_scoring import score_scenes


FIGURES_DIR = cfg.OUTPUT_DIR / "figures"

N_SCENES_PER_ROW = 3   # scenes per row in the figure


def fetch_scene_arrays(img: ee.Image, aoi: ee.Geometry) -> dict | None:
    """Download RGB, cloud_prob and clear_mask arrays for one scene."""
    try:
        sample = (
            img.select(["B4", "B3", "B2", "cloud_prob", "clear_mask"])
            .sampleRectangle(region=aoi, defaultValue=0)
            .getInfo()
        )
        props = sample["properties"]
        b4 = np.array(props["B4"], dtype=np.float32)
        b3 = np.array(props["B3"], dtype=np.float32)
        b2 = np.array(props["B2"], dtype=np.float32)
        cloud = np.array(props["cloud_prob"], dtype=np.float32)
        clear = np.array(props["clear_mask"], dtype=np.float32)
        return dict(b4=b4, b3=b3, b2=b2, cloud=cloud, clear=clear)
    except Exception as exc:
        print(f"    [fetch failed: {exc}]")
        return None


def make_rgb(b4, b3, b2, lo=2, hi=98):
    """Percentile-stretch to uint8 RGB."""
    def stretch(band):
        valid = band[band > 0]
        if valid.size == 0:
            return np.zeros_like(band)
        p_lo, p_hi = np.percentile(valid, [lo, hi])
        p_hi = max(p_hi, p_lo + 1)
        return np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)
    return np.stack([stretch(b4), stretch(b3), stretch(b2)], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip-id", required=True)
    args = p.parse_args()

    chip_id = args.chip_id
    # Parse col/row from chip_id_str like c+000572_r+002106 or c-000084_r+002660
    parts = chip_id.split("_")
    col = int(parts[0][1:])   # strip leading 'c'
    row = int(parts[1][1:])   # strip leading 'r'

    bounds = chip_id_to_bounds(col, row)
    xmin, ymin, xmax, ymax = bounds

    initialize_ee()
    aoi = ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax],
        proj=ee.Projection("EPSG:3857"),
        evenOdd=False,
    )

    print(f"Loading scenes for {chip_id} ...")
    s2 = load_sentinel_l1c(aoi=aoi, start_date=cfg.START_DATE, end_date=cfg.END_DATE)
    s2 = attach_cloud_prob(s2)
    s2 = s2.map(lambda img: compute_clear_mask(img, aoi=aoi, use_shadow=False))

    # Score so we know chip-level clear fraction for every scene
    scored = score_scenes(s2, aoi=aoi,
                          start_date=cfg.START_DATE, end_date=cfg.END_DATE)
    n_total = int(s2.size().getInfo())
    print(f"  {n_total} scenes found")

    # Build system:index → chip-level score map from the scored collection
    scored_list = scored.toList(n_total).getInfo()
    chip_score_map = {}   # system:index → (chip_score, rank)
    for rank, item in enumerate(scored_list):
        sid = item["properties"].get("system:index", "")
        score = item["properties"].get("score", -1)
        chip_score_map[sid] = (score, rank)

    # Collect scene metadata (using unscored list to preserve original images)
    scene_list = s2.toList(n_total)
    scenes_meta = []
    for i in range(n_total):
        img = ee.Image(scene_list.get(i))
        props = img.getInfo()["properties"]
        sid = props.get("system:index", "")
        date = sid[:8]
        chip_score, rank = chip_score_map.get(sid, (-1, -1))
        scenes_meta.append(dict(img=img, date=date, chip_score=chip_score,
                                rank=rank, idx=i))

    # Sort chronologically
    scenes_meta.sort(key=lambda x: x["date"])

    # Figure layout: N_SCENES_PER_ROW scenes per row, 2 columns per scene
    n = len(scenes_meta)
    n_rows = int(np.ceil(n / N_SCENES_PER_ROW))
    n_cols = N_SCENES_PER_ROW * 2   # RGB + cloud_mask per scene

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(N_SCENES_PER_ROW * 5, n_rows * 2.8),
        squeeze=False,
    )
    fig.suptitle(
        f"{chip_id}  —  individual scenes + cloud mask  ({cfg.START_DATE} → {cfg.END_DATE})",
        fontsize=11, y=1.01,
    )

    for i, meta in enumerate(scenes_meta):
        row_idx = i // N_SCENES_PER_ROW
        col_rgb  = (i % N_SCENES_PER_ROW) * 2
        col_cld  = col_rgb + 1

        ax_rgb = axes[row_idx][col_rgb]
        ax_cld = axes[row_idx][col_cld]

        date       = meta["date"]
        chip_score = meta["chip_score"]
        rank       = meta["rank"]
        print(f"  Fetching scene {i+1}/{n}: {date}  chip_score={chip_score:.3f}  rank={rank} ...")
        arrays = fetch_scene_arrays(meta["img"], aoi)

        if arrays is None:
            ax_rgb.text(0.5, 0.5, "fetch\nfailed", ha="center", va="center",
                        transform=ax_rgb.transAxes)
            ax_cld.text(0.5, 0.5, "fetch\nfailed", ha="center", va="center",
                        transform=ax_cld.transAxes)
        else:
            rgb = make_rgb(arrays["b4"], arrays["b3"], arrays["b2"])
            ax_rgb.imshow(rgb)
            ax_cld.imshow(arrays["cloud"], cmap="Reds", vmin=0, vmax=100)

            # Overlay clear mask boundary on cloud panel
            ax_cld.contour(arrays["clear"], levels=[0.5], colors=["lime"],
                           linewidths=0.8)

        ax_rgb.axis("off")
        ax_cld.axis("off")
        rank_str = f" [rank#{rank}]" if rank < 8 else ""
        title_col = "navy" if rank < 8 else "black"
        ax_rgb.set_title(f"{date}{rank_str}\nchip_score={chip_score:.3f}",
                         fontsize=7, pad=2, color=title_col)
        ax_cld.set_title("cloud prob + clear mask", fontsize=7, pad=2)

    # Hide unused axes
    for i in range(n, n_rows * N_SCENES_PER_ROW):
        row_idx = i // N_SCENES_PER_ROW
        col_rgb  = (i % N_SCENES_PER_ROW) * 2
        axes[row_idx][col_rgb].axis("off")
        axes[row_idx][col_rgb + 1].axis("off")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{chip_id}_scenes.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
