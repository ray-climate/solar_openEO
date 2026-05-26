#!/bin/bash
# Run the 30K self-audit using the fine-tuned weights (exp_round3_r101_reviewed).
# Output: docs/audit_30k_h5_reviewed.csv  (kept separate from the baseline audit
# at docs/audit_30k_h5.csv for easy diff)
#SBATCH --partition=standard
#SBATCH --account=gbov
#SBATCH --qos=high
#SBATCH --job-name=audit_30k_reviewed
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO
set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate base

python -u scripts/audit_30k_full.py \
    --weights experiments/exp_round3_r101_reviewed/best.weights.h5 \
    --out docs/audit_30k_h5_reviewed.csv
