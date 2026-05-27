#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=eu100
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO
set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base
python -u scripts/test_europe_100.py
