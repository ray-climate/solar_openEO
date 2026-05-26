#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=compare_realworld_50
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -eo pipefail
cd /gws/ssde/j25b/gbov/solar_openEO
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate tf-gpu
NVIDIA_LD_PATHS=$(find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"
python -u scripts/compare_realworld_50.py
