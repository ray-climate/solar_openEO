#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=cmp50_v2
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /gws/ssde/j25b/gbov/solar_openEO
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base
python -u scripts/compare_realworld_50.py
