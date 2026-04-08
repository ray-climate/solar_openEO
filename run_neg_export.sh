#!/bin/bash
# Submit negatives to GEE in batches of 50, matching the positive pipeline approach.
# The script is resumable — already-submitted chips are skipped automatically.
# Run repeatedly until all 2000 chips are submitted.

PYTHON=/home/users/ruisong/mambaforge_new/bin/python
MANIFEST=outputs/stage1/negatives/unique_chip_manifest.csv
JOBS_CSV=outputs/stage1/negatives/export_jobs.csv
FOLDER=solar_openEO_negatives_v1
LOG=outputs/stage1/negatives/export_submit.log

nohup bash -c '
  while true; do
    PENDING=$('$PYTHON' -c "
import pandas as pd
manifest = pd.read_csv(\"'$MANIFEST'\")[\"chip_id_str\"].tolist()
try:
    done = pd.read_csv(\"'$JOBS_CSV'\")[\"chip_id_str\"].tolist()
except FileNotFoundError:
    done = []
print(len(set(manifest) - set(done)))
")
    echo "$(date): $PENDING chips remaining"
    if [ "$PENDING" -eq 0 ]; then
      echo "All chips submitted."
      break
    fi
    '$PYTHON' scripts/02_launch_gee_exports.py \
      --manifest '$MANIFEST' \
      --drive-folder '$FOLDER' \
      --jobs-csv '$JOBS_CSV' \
      --name-prefix neg \
      --batch-size 50
    sleep 10
  done
' >> "$LOG" 2>&1 &

echo "Submission loop running in background (PID $!)"
echo "Tail log with: tail -f $LOG"
