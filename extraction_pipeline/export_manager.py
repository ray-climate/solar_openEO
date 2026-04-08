"""Manage GEE mosaic export jobs for Stage-1 tiles.

Each unique tile in the tile manifest gets one call to
``create_temporal_mosaic()``.  Export status is persisted in
``export_jobs.csv`` so runs are resumable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import ee
import pandas as pd

from mosaic_module import create_temporal_mosaic
from mosaic_module.io_helpers import initialize_ee

from . import config as cfg
from .tiling import chip_id_to_bounds

LOGGER = logging.getLogger(__name__)

_JOBS_COLS = [
    "chip_id_str",
    "gee_task_id",
    "gee_task_state",
    "export_name",
    "chip_center_lat",
    "chip_center_lon",
    "status",          # mirrors gee_task_state; updated by refresh_job_status()
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def chip_to_export_name(chip_id_str: str, name_prefix: str = "stage1") -> str:
    """Deterministic GEE export name for a chip.

    Example: ``stage1_cp0002504_rm0000213`` (positives)
             ``neg_cp0002504_rm0000213``    (negatives)
    Using the chip_id_str keeps it unique and traceable to the manifest.
    """
    # Replace + and - with p and m so the name is GEE-safe (alphanumeric + _ -)
    safe = chip_id_str.replace("+", "p").replace("-", "m")
    return f"{name_prefix}_{safe}"


# ---------------------------------------------------------------------------
# Single-tile submit
# ---------------------------------------------------------------------------

def submit_chip_export(
    chip_row: pd.Series,
    drive_folder: str | None = None,
    name_prefix: str = "stage1",
) -> dict:
    """Submit a GEE export for one chip row from the unique chip manifest.

    Passes exact chip grid bounds to ``create_temporal_mosaic`` so GEE exports
    a pixel-grid-aligned GeoTIFF at exactly CHIP_SIZE_PX × CHIP_SIZE_PX pixels,
    avoiding off-by-one resampling blur.

    Parameters
    ----------
    drive_folder:
        Google Drive folder for the export.  Defaults to ``cfg.DRIVE_FOLDER``
        when ``None``.  Pass an explicit value to export negatives or other
        datasets to a separate folder.
    name_prefix:
        Prefix for the GEE export task name, e.g. ``"stage1"`` for positives
        or ``"neg"`` for negatives.

    Returns a dict with job metadata.
    """
    bounds = chip_id_to_bounds(int(chip_row["chip_col"]), int(chip_row["chip_row"]))
    result = create_temporal_mosaic(
        center_lat=float(chip_row["chip_center_lat"]),
        center_lon=float(chip_row["chip_center_lon"]),
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        export_name=chip_to_export_name(chip_row["chip_id_str"], name_prefix=name_prefix),
        drive_folder=drive_folder if drive_folder is not None else cfg.DRIVE_FOLDER,
        export_rgb=False,   # RGB is generated locally from the 13-band chip
        aoi_bounds_3857=bounds,
        **cfg.MOSAIC_PARAMS,
    )

    return {
        "chip_id_str": chip_row["chip_id_str"],
        "gee_task_id": result["task_id"],
        "gee_task_state": result["task_state"],
        "export_name": result["export_name"],
        "chip_center_lat": chip_row["chip_center_lat"],
        "chip_center_lon": chip_row["chip_center_lon"],
        "status": result["task_state"],
    }


# ---------------------------------------------------------------------------
# Batch submit (with resume support)
# ---------------------------------------------------------------------------

def submit_all_exports(
    unique_chip_manifest: pd.DataFrame,
    jobs_csv_path: str | Path,
    dry_run: bool = False,
    batch_size: int = 50,
    chip_ids: list[str] | None = None,
    retry_errors: bool = False,
    drive_folder: str | None = None,
    name_prefix: str = "stage1",
) -> pd.DataFrame:
    """Submit GEE exports for chips not yet in jobs_csv.

    Parameters
    ----------
    unique_chip_manifest:
        Output of ``build_unique_chip_manifest()``.
    jobs_csv_path:
        Path to persist job records.  Loaded if it exists (resume support).
    dry_run:
        If True, print what would be submitted without calling GEE.
    batch_size:
        Max number of new jobs to submit per call.
    chip_ids:
        If provided, only submit these chip_id_str values (subset).

    Returns
    -------
    Updated jobs DataFrame (also saved to jobs_csv_path).
    """
    jobs_csv_path = Path(jobs_csv_path)

    if jobs_csv_path.exists():
        existing = pd.read_csv(jobs_csv_path, dtype=str)
        LOGGER.info("Loaded %d existing job records from %s", len(existing), jobs_csv_path)
        if retry_errors:
            n_errors = (existing["status"] == "SUBMIT_ERROR").sum()
            existing = existing[existing["status"] != "SUBMIT_ERROR"].copy()
            LOGGER.info("retry_errors=True: removed %d SUBMIT_ERROR rows", n_errors)
    else:
        existing = pd.DataFrame(columns=_JOBS_COLS)

    already_submitted = set(existing["chip_id_str"].tolist())

    pending = unique_chip_manifest[
        ~unique_chip_manifest["chip_id_str"].isin(already_submitted)
    ]
    if chip_ids is not None:
        pending = pending[pending["chip_id_str"].isin(chip_ids)]

    pending = pending.head(batch_size)
    LOGGER.info(
        "%d chips pending submission (batch_size=%d, dry_run=%s)",
        len(pending), batch_size, dry_run,
    )

    if dry_run:
        for _, row in pending.iterrows():
            print(f"  [DRY RUN] Would submit: {row['chip_id_str']}  "
                  f"lat={row['chip_center_lat']:.4f}  lon={row['chip_center_lon']:.4f}")
        return existing

    initialize_ee()
    new_jobs: list[dict] = []
    for _, row in pending.iterrows():
        LOGGER.info("Submitting chip %s ...", row["chip_id_str"])
        try:
            job = submit_chip_export(row, drive_folder=drive_folder, name_prefix=name_prefix)
            new_jobs.append(job)
            LOGGER.info(
                "  → task_id=%s  state=%s", job["gee_task_id"], job["gee_task_state"]
            )
        except Exception as exc:
            LOGGER.error("  Failed to submit %s: %s", row["chip_id_str"], exc)
            new_jobs.append({
                "chip_id_str": row["chip_id_str"],
                "gee_task_id": "",
                "gee_task_state": "SUBMIT_ERROR",
                "export_name": chip_to_export_name(row["chip_id_str"], name_prefix=name_prefix),
                "chip_center_lat": row["chip_center_lat"],
                "chip_center_lon": row["chip_center_lon"],
                "status": "SUBMIT_ERROR",
            })
        time.sleep(0.5)

    new_df = pd.DataFrame(new_jobs) if new_jobs else pd.DataFrame(columns=_JOBS_COLS)
    frames = [df for df in [existing, new_df] if not df.empty]
    all_jobs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_JOBS_COLS)
    jobs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_jobs.to_csv(jobs_csv_path, index=False)
    LOGGER.info("Saved %d job records to %s", len(all_jobs), jobs_csv_path)
    return all_jobs


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

def refresh_job_status(jobs_csv_path: str | Path) -> pd.DataFrame:
    """Query GEE for current task status of all submitted jobs.

    Updates the ``status`` column in-place and saves the CSV.

    Returns the updated DataFrame.
    """
    jobs_csv_path = Path(jobs_csv_path)
    jobs = pd.read_csv(jobs_csv_path, dtype=str)

    initialize_ee()
    task_list = ee.batch.Task.list()
    task_map = {t.id: t.status()["state"] for t in task_list}

    updated = 0
    for i, row in jobs.iterrows():
        task_id = row.get("gee_task_id", "")
        if task_id and task_id in task_map:
            new_state = task_map[task_id]
            jobs.at[i, "status"] = new_state
            jobs.at[i, "gee_task_state"] = new_state
            updated += 1

    jobs.to_csv(jobs_csv_path, index=False)
    summary = jobs["status"].value_counts().to_dict()
    LOGGER.info("Status refresh: updated %d tasks  summary=%s", updated, summary)
    return jobs


def get_completed_chip_ids(jobs_csv_path: str | Path) -> list[str]:
    """Return chip_id_str for all COMPLETED jobs."""
    jobs = pd.read_csv(jobs_csv_path, dtype=str)
    completed = jobs[jobs["status"] == "COMPLETED"]["chip_id_str"].tolist()
    return completed


def print_status_summary(jobs_csv_path: str | Path) -> None:
    """Print a human-readable status table to stdout."""
    jobs = pd.read_csv(jobs_csv_path, dtype=str)
    counts = jobs["status"].value_counts()
    total = len(jobs)
    print(f"\nExport job status  ({jobs_csv_path})")
    print(f"  Total submitted : {total}")
    for state, n in counts.items():
        print(f"  {state:20s}: {n}")
    print()
