#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=fpGeeExtract
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO
set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base
# GEE reduceRegions over all 26,749 polygon-windows. Resumes via skip-existing
# (chunks already written to outputs/fp_classifier/timeseries/ are skipped).
python -u scripts/run_fp_phase2_extract_gee.py
