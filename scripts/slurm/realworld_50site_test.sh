#!/bin/bash
# Run the 50-site real-world detection test on a SLURM compute node.
# Survives login-session/tmp cleanup. Uses the cached OpenEO refresh
# token from ~/.local/share/openeo-python-client/.
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=realworld_50site
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gws/ssde/j25b/gbov/solar_openEO}"
cd "$REPO_DIR"
mkdir -p slurm_logs

echo "============================================================"
echo "Job ID  : ${SLURM_JOB_ID:-not_assigned}"
echo "Node    : ${SLURM_NODELIST:-unknown}"
echo "Started : $(date -u +%FT%TZ)"
echo "============================================================"

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base

python -u scripts/19_test_real_world_sites.py --phase all \
    --n-sites 50 --unseen-only \
    --output-dir docs/realworld_unseen_50_2026Q1

echo "Finished: $(date -u +%FT%TZ)"
