#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=diagnose_site
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base

CHIP_ID=${CHIP_ID:?Set CHIP_ID=<chip_id>}
python -u scripts/diagnose_site.py --chip-id "$CHIP_ID"
