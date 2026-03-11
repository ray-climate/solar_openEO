"""QA visualization for extracted Stage-1 chips.

Produces three figures saved to outputs/stage1/figures/:

  1. chip_grid.png      — RGB + mask overlay grid for all extracted chips
  2. chip_locations.png — world map of chip locations coloured by continent
  3. chip_stats.png     — panel fraction histogram + continent bar chart

Usage:
  conda run -n geospatial python scripts/06_visualize_chips.py
  conda run -n geospatial python scripts/06_visualize_chips.py --max-chips 30
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

FIGURES_DIR = cfg.OUTPUT_DIR / "figures"

CONTINENT_COLORS = {
    "Asia":      "#e41a1c",
    "Europe":    "#377eb8",
    "N_America": "#4daf4a",
    "S_America": "#984ea3",
    "Africa":    "#ff7f00",
    "Oceania":   "#a65628",
    "Other":     "#999999",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Linear percentile stretch to [0, 1]."""
    p_lo, p_hi = np.percentile(band[band > 0], [lo, hi]) if (band > 0).any() else (0, 1)
    p_hi = max(p_hi, p_lo + 1)
    stretched = (band.astype(float) - p_lo) / (p_hi - p_lo)
    return np.clip(stretched, 0, 1)


def make_rgb(image_arr: np.ndarray) -> np.ndarray:
    """Build an 8-bit RGB array (H, W, 3) from a (13, H, W) S2 chip.

    Uses B4 (red), B3 (green), B2 (blue) with per-band percentile stretch.
    Band order from config: B1,B2,B3,B4,... so indices 3,2,1.
    """
    r = percentile_stretch(image_arr[3])   # B4
    g = percentile_stretch(image_arr[2])   # B3
    b = percentile_stretch(image_arr[1])   # B2
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def load_chip(
    chip_id_str: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load (image_arr, rgb_arr, mask_arr) for a chip. Returns (None,None,None) if missing."""
    img_path = cfg.CHIPS_DIR / f"{chip_id_str}_image.tif"
    msk_path = cfg.CHIPS_DIR / f"{chip_id_str}_mask.tif"
    if not img_path.exists() or not msk_path.exists():
        return None, None, None
    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(msk_path) as src:
        mask = src.read(1).astype(np.uint8)
    rgb = make_rgb(image)
    return image, rgb, mask


# ---------------------------------------------------------------------------
# Figure 1: chip grid
# ---------------------------------------------------------------------------

def plot_chip_grid(chip_metadata: pd.DataFrame, max_chips: int, out_path: Path) -> None:
    """Grid of chips: each column shows RGB and RGB+mask overlay."""
    valid = chip_metadata[chip_metadata["coverage_ok"] == True].head(max_chips)  # noqa: E712
    n = len(valid)
    if n == 0:
        LOGGER.warning("No valid chips to plot.")
        return

    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(ncols * 2.8, nrows * 5.2),
        squeeze=False,
    )
    fig.suptitle(
        f"Stage-1 positive chips — RGB (top) and mask overlay (bottom)\n"
        f"({n} chips, May–Jul 2024)",
        fontsize=13, y=1.01,
    )

    mask_color = np.array([1.0, 0.15, 0.15])   # red overlay

    for i, (_, row) in enumerate(valid.iterrows()):
        col_idx = i % ncols
        row_idx = i // ncols

        ax_rgb  = axes[row_idx * 2][col_idx]
        ax_mask = axes[row_idx * 2 + 1][col_idx]

        image, rgb, mask = load_chip(row["chip_id_str"])
        if image is None:
            ax_rgb.axis("off")
            ax_mask.axis("off")
            continue

        # rgb is always computed in load_chip; no fallback needed

        # RGB
        ax_rgb.imshow(rgb)
        ax_rgb.axis("off")
        continent = row.get("continent", "")
        panel_pct = float(row["panel_frac"]) * 100
        ax_rgb.set_title(
            f"{row['chip_id_str']}\n{continent}  |  {panel_pct:.2f}% panel",
            fontsize=6.5, pad=2,
        )

        # RGB + mask overlay
        overlay = rgb.astype(float) / 255.0
        if mask.any():
            panel_pixels = mask == 1
            overlay[panel_pixels] = (
                0.55 * overlay[panel_pixels] + 0.45 * mask_color
            )
        ax_mask.imshow(overlay)
        ax_mask.axis("off")
        n_px = int(row["n_panel_px"])
        ax_mask.set_title(f"panel px: {n_px}", fontsize=6.5, pad=2)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        col_idx = i % ncols
        row_idx = i // ncols
        axes[row_idx * 2][col_idx].axis("off")
        axes[row_idx * 2 + 1][col_idx].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved chip grid → %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2: world map
# ---------------------------------------------------------------------------

def plot_chip_locations(chip_metadata: pd.DataFrame, out_path: Path) -> None:
    """World map with chip locations coloured by continent."""
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        world = None

    valid = chip_metadata[chip_metadata["coverage_ok"] == True]  # noqa: E712
    if valid.empty:
        LOGGER.warning("No valid chips to plot on map.")
        return

    # Reconstruct lat/lon from chip_col/chip_row
    import pyproj
    tr = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x_centers = valid["chip_col"] * cfg.CHIP_SIZE_M + cfg.CHIP_SIZE_M / 2
    y_centers = valid["chip_row"] * cfg.CHIP_SIZE_M + cfg.CHIP_SIZE_M / 2
    lons, lats = tr.transform(x_centers.values, y_centers.values)

    fig, ax = plt.subplots(figsize=(14, 7))
    if world is not None:
        world.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.4)

    for cont, color in CONTINENT_COLORS.items():
        mask = valid["continent"] == cont
        if mask.any():
            ax.scatter(
                lons[mask.values], lats[mask.values],
                s=18, color=color, alpha=0.85, linewidths=0,
                zorder=5, label=f"{cont} ({mask.sum()})",
            )

    ax.set_title(
        f"Stage-1 chip locations  ({len(valid)} chips extracted)", fontsize=13
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 80)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved location map → %s", out_path)


# ---------------------------------------------------------------------------
# Figure 3: statistics
# ---------------------------------------------------------------------------

def plot_chip_stats(chip_metadata: pd.DataFrame, out_path: Path) -> None:
    """Panel fraction histogram + continent bar chart."""
    valid = chip_metadata[chip_metadata["coverage_ok"] == True]  # noqa: E712
    if valid.empty:
        LOGGER.warning("No valid chips to plot stats.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Stage-1 chip statistics", fontsize=13)

    # Panel fraction histogram
    panel_pcts = valid["panel_frac"].astype(float) * 100
    ax1.hist(panel_pcts, bins=30, color="#377eb8", edgecolor="white", linewidth=0.5)
    ax1.axvline(panel_pcts.median(), color="#e41a1c", linestyle="--", linewidth=1.5,
                label=f"median={panel_pcts.median():.2f}%")
    ax1.set_xlabel("Panel coverage within chip (%)")
    ax1.set_ylabel("Number of chips")
    ax1.set_title("Panel pixel fraction distribution")
    ax1.legend(fontsize=9)

    # Continent bar chart
    counts = valid["continent"].value_counts()
    colors = [CONTINENT_COLORS.get(c, "#999999") for c in counts.index]
    ax2.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Continent")
    ax2.set_ylabel("Number of chips")
    ax2.set_title("Chips by continent")
    ax2.tick_params(axis="x", rotation=30)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
        ax1.spines[spine].set_visible(False)

    # Summary text
    total_panel_px = valid["n_panel_px"].sum()
    total_px = len(valid) * cfg.CHIP_SIZE_PX ** 2
    fig.text(
        0.5, -0.04,
        f"Total chips: {len(valid)}  |  "
        f"Total panel pixels: {total_panel_px:,} / {total_px:,}  "
        f"({100*total_panel_px/total_px:.3f}%)",
        ha="center", fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved stats figure → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-chips", type=int, default=30,
                   help="Max chips to show in the grid figure (default 30).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not cfg.CHIP_METADATA_CSV.exists():
        LOGGER.error("Chip metadata not found: %s", cfg.CHIP_METADATA_CSV)
        LOGGER.error("Run scripts/04_extract_chips.py first.")
        sys.exit(1)

    chip_metadata = pd.read_csv(cfg.CHIP_METADATA_CSV)
    n_valid = (chip_metadata["coverage_ok"] == True).sum()  # noqa: E712
    LOGGER.info("Chip metadata: %d total, %d valid", len(chip_metadata), n_valid)

    if n_valid == 0:
        LOGGER.error("No valid chips found. Check mosaics were downloaded and extracted.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_chip_grid(
        chip_metadata, max_chips=args.max_chips,
        out_path=FIGURES_DIR / "chip_grid.png",
    )
    plot_chip_locations(
        chip_metadata,
        out_path=FIGURES_DIR / "chip_locations.png",
    )
    plot_chip_stats(
        chip_metadata,
        out_path=FIGURES_DIR / "chip_stats.png",
    )

    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"  chip_grid.png      — RGB + mask overlay for up to {args.max_chips} chips")
    print(f"  chip_locations.png — world map of chip locations")
    print(f"  chip_stats.png     — panel fraction + continent breakdown")


if __name__ == "__main__":
    main()
