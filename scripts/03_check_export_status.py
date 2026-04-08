"""Step 3: Poll GEE for current export job status.

Updates the export jobs CSV in-place and prints a summary.

Usage:
  conda run -n geospatial python scripts/03_check_export_status.py

  # For negatives:
  conda run -n geospatial python scripts/03_check_export_status.py \\
      --jobs-csv outputs/stage1/negatives/export_jobs.csv

  # Repeat until all jobs are COMPLETED:
  watch -n 30 conda run -n geospatial python scripts/03_check_export_status.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction_pipeline import config as cfg
from extraction_pipeline.export_manager import print_status_summary, refresh_job_status

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-csv", default=str(cfg.EXPORT_JOBS_CSV),
                   help="Export jobs CSV to poll (default: export_jobs.csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jobs_csv = Path(args.jobs_csv)

    if not jobs_csv.exists():
        LOGGER.error("No export jobs file found: %s", jobs_csv)
        LOGGER.error("Run scripts/02_launch_gee_exports.py first.")
        sys.exit(1)

    jobs = refresh_job_status(jobs_csv)
    print_status_summary(jobs_csv)

    completed = (jobs["status"] == "COMPLETED").sum()
    total = len(jobs)
    if completed == total:
        print(f"All {total} jobs COMPLETED.")
        print(f"Next: run scripts/04_extract_chips.py")
    else:
        remaining = total - completed
        print(f"{remaining} job(s) still running. Re-run this script to check again.")


if __name__ == "__main__":
    main()
