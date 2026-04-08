#!/usr/bin/env python
"""Visual test: download multi-temporal S2 stacks and run SLIC mosaic UDF.

For a few random TEST chips, downloads L1C + SCL multi-temporal data from
CDSE OpenEO, applies the SLIC-based temporal mosaic, and generates a
diagnostic figure showing:
  - Per-scene RGB thumbnails (one per acquisition date)
  - Scene clear fraction overlay
  - Final mosaic RGB
  - Scene assignment map (which scene provided each pixel)

Modes:
  Default:    Random test chips from training manifest (solar panel areas)
  --tropical: Stress-test sites in cloudy tropical / monsoon regions

Usage:
    python openeo_udp/tests/test_mosaic_visual.py [--skip-download] [--n-chips 3]
    python openeo_udp/tests/test_mosaic_visual.py --tropical [--skip-download]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
OUTPUT_DIR = REPO / "openeo_udp" / "tests" / "mosaic_visual_outputs"
MANIFEST_PATH = REPO / "outputs/training_prep/stage1_v3/split_manifest.csv"

CELL_SIZE = 2560  # metres per chip in EPSG:3857
TEST_TEMPORAL = ["2024-05-01", "2024-07-31"]

S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]


# --- Tropical / cloudy stress-test sites ---
# ~2.5 km x 2.5 km boxes in regions with persistent cloud cover during
# May–July (monsoon, ITCZ, or tropical convection).
# These are NOT solar panel areas — purely for mosaic quality testing.
TROPICAL_SITES = [
    {
        "chip_id": "tropical_amazon",
        "label": "Amazon basin, Brazil",
        "bbox": {"west": -60.05, "south": -3.12, "east": -60.025, "north": -3.10,
                 "crs": "EPSG:4326"},
    },
    {
        "chip_id": "tropical_congo",
        "label": "Congo basin, DRC",
        "bbox": {"west": 21.45, "south": -0.50, "east": 21.475, "north": -0.48,
                 "crs": "EPSG:4326"},
    },
    {
        "chip_id": "tropical_borneo",
        "label": "Borneo, Malaysia",
        "bbox": {"west": 110.35, "south": 1.55, "east": 110.375, "north": 1.57,
                 "crs": "EPSG:4326"},
    },
    {
        "chip_id": "tropical_india_monsoon",
        "label": "Western Ghats, India (monsoon)",
        "bbox": {"west": 75.75, "south": 12.30, "east": 75.775, "north": 12.32,
                 "crs": "EPSG:4326"},
    },
    {
        "chip_id": "tropical_nigeria",
        "label": "Southern Nigeria",
        "bbox": {"west": 3.38, "south": 6.45, "east": 3.405, "north": 6.47,
                 "crs": "EPSG:4326"},
    },
    {
        "chip_id": "tropical_colombia",
        "label": "Chocó, Colombia (wettest region on Earth)",
        "bbox": {"west": -76.65, "south": 5.68, "east": -76.625, "north": 5.70,
                 "crs": "EPSG:4326"},
    },
]


def select_tropical_sites(n: int) -> list[dict]:
    """Select tropical cloudy stress-test sites."""
    sites = TROPICAL_SITES[:n]
    return [{"chip_id": s["chip_id"], "bbox": s["bbox"], "label": s["label"]} for s in sites]


def chip_to_bbox_4326(col: int, row: int) -> dict:
    """Convert chip col/row to EPSG:4326 bounding box."""
    xmin = col * CELL_SIZE
    ymin = row * CELL_SIZE
    xmax = xmin + CELL_SIZE
    ymax = ymin + CELL_SIZE

    def m_to_lon(x):
        return x / 20037508.34 * 180.0

    def m_to_lat(y):
        return math.degrees(2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)

    return {
        "west": m_to_lon(xmin),
        "south": m_to_lat(ymin),
        "east": m_to_lon(xmax),
        "north": m_to_lat(ymax),
        "crs": "EPSG:4326",
    }


def select_test_chips(n: int, seed: int = 42) -> list[dict]:
    """Select n random TEST chips from the manifest."""
    test_chips = []
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                test_chips.append(row)

    random.seed(seed)
    selected = random.sample(test_chips, min(n, len(test_chips)))

    result = []
    for c in selected:
        col, row = int(c["col"]), int(c["row"])
        bbox = chip_to_bbox_4326(col, row)
        result.append({
            "chip_id": c["chip_id"],
            "col": col,
            "row": row,
            "index": int(c["index"]),
            "bbox": bbox,
        })
    return result


def download_chip_stack(conn, chip: dict, output_dir: Path) -> Path:
    """Download multi-temporal L1C + SCL stack for one chip."""
    nc_path = output_dir / f"{chip['chip_id']}_stack.nc"
    if nc_path.exists():
        print(f"      Already downloaded: {nc_path.name}")
        return nc_path

    bbox = chip["bbox"]

    # Load L1C (13 spectral bands), resampled to 10m EPSG:3857
    s2_l1c = conn.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=bbox,
        temporal_extent=TEST_TEMPORAL,
        bands=S2_L1C_BANDS,
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    # Load L2A SCL, resampled to match
    s2_scl = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=TEST_TEMPORAL,
        bands=["SCL"],
    ).resample_spatial(resolution=10, projection="EPSG:3857")

    # Merge into 14-band temporal cube
    merged = s2_l1c.merge_cubes(s2_scl)

    # Submit batch job
    job = merged.create_job(
        title=f"mosaic_test_{chip['chip_id']}",
        out_format="netCDF",
    )
    job.start_job()
    print(f"      Job ID: {job.job_id}")

    # Poll
    t0 = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - t0
        if elapsed > 10 or status != "queued":
            print(f"      [{elapsed:5.0f}s] {status}")

        if status == "finished":
            break
        elif status in ("error", "canceled"):
            try:
                for log in job.logs()[-5:]:
                    msg = log.get("message", str(log)) if isinstance(log, dict) else str(log)
                    print(f"      LOG: {msg}")
            except Exception:
                pass
            raise RuntimeError(f"Job failed: {status}")

        time.sleep(15)

    # Download
    tmp_dir = output_dir / f"_tmp_{chip['chip_id']}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job.get_results().download_files(tmp_dir)

    nc_files = list(tmp_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No netCDF files for {chip['chip_id']}")

    nc_files[0].rename(nc_path)
    # Cleanup temp dir
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"      Downloaded: {nc_path.name} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    return nc_path


def load_stack(nc_path: Path):
    """Load multi-temporal stack from netCDF.

    Returns
    -------
    spectral : (T, 13, H, W) float32
    scl : (T, H, W) int32
    dates : list of str (date labels per scene)
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    band_vars = [v for v in ds.data_vars if v in S2_L1C_BANDS or v == "SCL"]

    if len(band_vars) >= 14:
        # Each band is a separate variable with dims (t, y, x)
        spectral_bands = []
        for bname in S2_L1C_BANDS:
            if bname in ds:
                spectral_bands.append(ds[bname].values)

        spectral = np.stack(spectral_bands, axis=1).astype(np.float32)
        scl = ds["SCL"].values.astype(np.int32)

        # Extract date labels from time coordinate
        t_dim = next((d for d in ds[S2_L1C_BANDS[0]].dims if d in ("t", "time")), None)
        if t_dim and t_dim in ds.coords:
            dates = [str(np.datetime_as_string(t, unit="D")) for t in ds.coords[t_dim].values]
        else:
            dates = [f"scene_{i}" for i in range(spectral.shape[0])]
    else:
        # Single variable with bands dimension
        for varname in ds.data_vars:
            var = ds[varname]
            if len(var.dims) >= 3:
                data = var.values.astype(np.float32)
                break
        else:
            raise ValueError(f"No suitable variable in {nc_path}")

        if data.ndim == 4:
            # Assume (t, bands, y, x) or similar
            dims = list(var.dims)
            t_dim = next((d for d in dims if d in ("t", "time")), dims[0])

            if t_dim in ds.coords:
                dates = [str(np.datetime_as_string(t, unit="D")) for t in ds.coords[t_dim].values]
            else:
                dates = [f"scene_{i}" for i in range(data.shape[0])]

            spectral = data[:, :13, :, :]
            if data.shape[1] >= 14:
                scl = data[:, 13, :, :].astype(np.int32)
            else:
                scl = np.full(data[:, 0, :, :].shape, 4, dtype=np.int32)
        else:
            raise ValueError(f"Unexpected shape: {data.shape}")

    ds.close()
    return spectral, scl, dates


def make_rgb(data_chw: np.ndarray, percentile_stretch: tuple = (2, 98)) -> np.ndarray:
    """Convert (C, H, W) or (H, W, C) L1C data to RGB uint8 image.

    Uses B04 (red, idx 3), B03 (green, idx 2), B02 (blue, idx 1).
    """
    if data_chw.ndim == 3 and data_chw.shape[0] <= 13:
        # (C, H, W) → (H, W, C)
        data = np.transpose(data_chw, (1, 2, 0))
    else:
        data = data_chw

    rgb = data[:, :, [3, 2, 1]].astype(np.float32)

    # Percentile stretch
    valid = rgb[rgb > 0]
    if valid.size > 0:
        lo = np.percentile(valid, percentile_stretch[0])
        hi = np.percentile(valid, percentile_stretch[1])
    else:
        lo, hi = 0, 1

    rgb = np.clip((rgb - lo) / max(hi - lo, 1), 0, 1)
    return rgb


def generate_mosaic_figure(
    chip_id: str,
    spectral: np.ndarray,
    scl: np.ndarray,
    dates: list[str],
    composite: np.ndarray,
    info: dict,
    output_path: Path,
):
    """Generate diagnostic figure for one chip.

    Layout:
      Top row:    Per-scene RGBs with clear fraction + cloud overlay
      Bottom row: Mosaic RGB | Scene assignment map | Clear fraction bar chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import ListedColormap
    from openeo_udp.udf.temporal_mosaic import compute_clear_masks, DEFAULT_PARAMS

    t_scenes = spectral.shape[0]
    scene_scores = info["scene_scores"]
    scene_order = info["scene_order"]
    scene_assignment = info["scene_assignment"]

    # Compute clear masks for cloud overlay
    clear_masks = compute_clear_masks(scl, DEFAULT_PARAMS["scl_clear_classes"])

    # --- Compute consistent percentile stretch across all scenes + mosaic ---
    all_rgb_vals = []
    for t in range(t_scenes):
        rgb_raw = spectral[t, [3, 2, 1], :, :]  # (3, H, W)
        valid = rgb_raw[rgb_raw > 0]
        if valid.size > 0:
            all_rgb_vals.append(valid)

    comp_rgb_raw = composite[[3, 2, 1], :, :]
    valid_comp = comp_rgb_raw[comp_rgb_raw > 0]
    if valid_comp.size > 0:
        all_rgb_vals.append(valid_comp)

    if all_rgb_vals:
        all_vals = np.concatenate(all_rgb_vals)
        global_lo = np.percentile(all_vals, 2)
        global_hi = np.percentile(all_vals, 98)
    else:
        global_lo, global_hi = 0, 1

    def stretch_rgb(chw):
        rgb = np.transpose(chw[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
        return np.clip((rgb - global_lo) / max(global_hi - global_lo, 1), 0, 1)

    # --- Figure layout ---
    # Top rows: per-scene RGBs sorted by score (best first), up to 10
    # Bottom row: Mosaic RGB | Scene assignment | Fill mode
    n_show = min(t_scenes, 10)
    top_scenes = scene_order[:n_show]

    n_top_cols = min(n_show, 5)
    n_top_rows = math.ceil(n_show / n_top_cols)
    n_bot_cols = 3
    total_cols = max(n_top_cols, n_bot_cols)

    fig = plt.figure(figsize=(total_cols * 3.2, (n_top_rows + 1) * 3.5 + 1.2))
    gs = gridspec.GridSpec(
        n_top_rows + 1, total_cols,
        figure=fig, hspace=0.4, wspace=0.2,
    )

    # --- Top rows: per-scene RGBs ---
    cloud_overlay_cmap = ListedColormap(["none", "#ff000060"])

    for i, s_idx in enumerate(top_scenes):
        row_i = i // n_top_cols
        col_i = i % n_top_cols
        ax = fig.add_subplot(gs[row_i, col_i])

        rgb = stretch_rgb(spectral[s_idx])
        ax.imshow(rgb)

        # Cloud overlay (semi-transparent red)
        cloud = ~clear_masks[s_idx]
        if np.any(cloud):
            cloud_rgba = np.zeros((*cloud.shape, 4), dtype=np.float32)
            cloud_rgba[cloud] = [1, 0, 0, 0.35]
            ax.imshow(cloud_rgba)

        clear_pct = scene_scores[s_idx] * 100
        date_label = dates[s_idx] if s_idx < len(dates) else f"#{s_idx}"

        # Mark if this scene is used in the mosaic
        used_pixels = np.sum(scene_assignment == s_idx)
        used_pct = used_pixels / scene_assignment.size * 100

        title = f"{date_label}\n{clear_pct:.0f}% clear"
        if used_pct > 0:
            title += f" | {used_pct:.0f}% used"
            ax.set_title(title, fontsize=8, fontweight="bold", color="darkgreen")
        else:
            ax.set_title(title, fontsize=8, color="gray")

        ax.axis("off")

    # --- Bottom row ---
    bot_row = n_top_rows

    # Bottom left: Mosaic RGB
    ax_mosaic = fig.add_subplot(gs[bot_row, 0])
    mosaic_rgb = stretch_rgb(composite)
    ax_mosaic.imshow(mosaic_rgb)
    ax_mosaic.set_title("SLIC Mosaic", fontsize=11, fontweight="bold")
    ax_mosaic.axis("off")

    # Bottom centre: Scene assignment map
    ax_assign = fig.add_subplot(gs[bot_row, 1])
    n_unique = len(np.unique(scene_assignment[scene_assignment >= 0]))
    cmap = plt.cm.get_cmap("tab10", max(n_unique, 2))
    assign_display = scene_assignment.astype(np.float32)
    assign_display[assign_display < 0] = np.nan
    im = ax_assign.imshow(assign_display, cmap=cmap, interpolation="nearest")
    ax_assign.set_title(f"Scene Assignment\n({n_unique} scenes used)", fontsize=9)
    ax_assign.axis("off")

    # Legend: map scene index → date (as text below the map)
    used_scenes = np.unique(scene_assignment[scene_assignment >= 0])
    if len(used_scenes) <= 10:
        legend_parts = []
        for s in used_scenes:
            date_str = dates[s] if s < len(dates) else f"#{s}"
            pct = np.sum(scene_assignment == s) / scene_assignment.size * 100
            legend_parts.append(f"{date_str} ({pct:.0f}%)")
        ax_assign.text(
            0.5, -0.05, " | ".join(legend_parts),
            transform=ax_assign.transAxes, fontsize=7,
            ha="center", va="top",
        )

    # Bottom right: Rescue fill map
    ax_fill = fig.add_subplot(gs[bot_row, 2])
    fill_mode = info["fill_mode"]
    fill_cmap = ListedColormap(["#2ecc71", "#e74c3c"])  # green=primary, red=rescue
    ax_fill.imshow(fill_mode, cmap=fill_cmap, vmin=0, vmax=1, interpolation="nearest")
    rescue_pct = fill_mode.sum() / fill_mode.size * 100
    ax_fill.set_title(f"Fill Mode\nGreen=primary, Red=rescue ({rescue_pct:.1f}%)", fontsize=9)
    ax_fill.axis("off")

    # Suptitle
    bbox = chip_to_bbox_4326(0, 0)  # placeholder
    fig.suptitle(
        f"Temporal Mosaic Diagnostic — {chip_id}\n"
        f"{t_scenes} scenes, {info['n_clusters']} superpixels, "
        f"{info['n_scenes_used']} scenes used, {rescue_pct:.1f}% rescue-filled",
        fontsize=12, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      Figure saved: {output_path.name}")


def process_chip(chip: dict, nc_path: Path, output_dir: Path):
    """Run mosaic UDF on a downloaded chip and generate diagnostic figure."""
    from openeo_udp.udf.temporal_mosaic import create_temporal_mosaic

    chip_id = chip["chip_id"]
    print(f"   Processing {chip_id}...")

    # Load stack
    spectral, scl, dates = load_stack(nc_path)
    t_scenes, n_bands, h, w = spectral.shape
    print(f"      Stack: {t_scenes} scenes x {n_bands} bands x {h}x{w}")
    print(f"      Dates: {dates}")

    # Run mosaic
    t0 = time.time()
    composite, info = create_temporal_mosaic(spectral, scl)
    elapsed = time.time() - t0
    print(f"      Mosaic: {info['n_clusters']} clusters, "
          f"{info['n_scenes_used']} scenes used, "
          f"{info['fill_mode'].sum() / info['fill_mode'].size * 100:.1f}% rescue, "
          f"{elapsed:.1f}s")

    # Generate figure
    fig_path = output_dir / f"mosaic_diagnostic_{chip_id}.png"
    generate_mosaic_figure(
        chip_id=chip_id,
        spectral=spectral,
        scl=scl,
        dates=dates,
        composite=composite,
        info=info,
        output_path=fig_path,
    )

    return composite, info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OpenEO download, use existing .nc files")
    parser.add_argument("--n-chips", type=int, default=3,
                        help="Number of test chips to process (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chip selection")
    parser.add_argument("--tropical", action="store_true",
                        help="Use tropical cloudy stress-test sites instead of manifest chips")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    if args.tropical:
        print("Temporal Mosaic — TROPICAL CLOUD Stress Test")
    else:
        print("Temporal Mosaic — Visual Diagnostic Test")
    print("=" * 60)

    # Select chips
    if args.tropical:
        chips = select_tropical_sites(args.n_chips)
        print(f"\nSelected {len(chips)} tropical cloudy sites:")
        for c in chips:
            bbox = c["bbox"]
            label = c.get("label", "")
            print(f"   {c['chip_id']:30s} {label}")
            print(f"      lat={bbox['south']:.2f}–{bbox['north']:.2f}  "
                  f"lon={bbox['west']:.2f}–{bbox['east']:.2f}")
    else:
        chips = select_test_chips(args.n_chips, seed=args.seed)
        print(f"\nSelected {len(chips)} TEST chips:")
        for c in chips:
            bbox = c["bbox"]
            print(f"   {c['chip_id']:25s} lat={bbox['south']:.2f}–{bbox['north']:.2f} "
                  f"lon={bbox['west']:.2f}–{bbox['east']:.2f}")

    # Download multi-temporal stacks from OpenEO
    if not args.skip_download:
        import openeo

        print(f"\nConnecting to {BACKEND_URL}...")
        conn = openeo.connect(BACKEND_URL)
        conn.authenticate_oidc_device()
        user = conn.describe_account().get("user_id", "unknown")
        print(f"Authenticated as: {user}")

        print(f"\nDownloading multi-temporal stacks...")
        for i, chip in enumerate(chips):
            print(f"\n   [{i + 1}/{len(chips)}] {chip['chip_id']}")
            try:
                download_chip_stack(conn, chip, OUTPUT_DIR)
            except Exception as e:
                print(f"      [ERROR] {e}")
                continue

    # Process each chip
    print(f"\n{'=' * 60}")
    print("Running SLIC temporal mosaic on each chip...")
    print(f"{'=' * 60}")

    for i, chip in enumerate(chips):
        nc_path = OUTPUT_DIR / f"{chip['chip_id']}_stack.nc"
        if not nc_path.exists():
            print(f"\n   [{i + 1}/{len(chips)}] {chip['chip_id']} — SKIPPED (no data)")
            continue

        print(f"\n   [{i + 1}/{len(chips)}]")
        try:
            process_chip(chip, nc_path, OUTPUT_DIR)
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Done! Figures saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
