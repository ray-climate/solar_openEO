"""Step 3: Poll GEE for current export job status.

Updates outputs/stage1/export_jobs.csv in-place and prints a summary.

Usage:
  conda run -n geospatial python scripts/03_check_export_status.py

  # Repeat until all jobs are COMPLETED:
  watch -n 30 conda run -n geospatial python scripts/03_check_export_status.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.export_manager import print_status_summary, refresh_job_status

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    if not cfg.EXPORT_JOBS_CSV.exists():
        LOGGER.error("No export jobs file found: %s", cfg.EXPORT_JOBS_CSV)
        LOGGER.error("Run scripts/02_launch_gee_exports.py first.")
        sys.exit(1)

    jobs = refresh_job_status(cfg.EXPORT_JOBS_CSV)
    print_status_summary(cfg.EXPORT_JOBS_CSV)

    completed = (jobs["status"] == "COMPLETED").sum()
    total = len(jobs)
    if completed == total:
        print(f"All {total} jobs COMPLETED.")
        print(f"Download mosaics from Google Drive folder '{cfg.DRIVE_FOLDER}'")
        print(f"  → place .tif files in {cfg.MOSAICS_DIR}")
        print(f"Next: run scripts/04_extract_chips.py")
    else:
        remaining = total - completed
        print(f"{remaining} job(s) still running. Re-run this script to check again.")


if __name__ == "__main__":
    main()
