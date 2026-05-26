#!/bin/bash
# Fine-tune R3 R101 on the user-reviewed training set.
# Resumes from the R3 R101 dropout02 best weights at low LR.
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=exp_round3_r101_reviewed
#SBATCH -o /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.out
#SBATCH -e /gws/ssde/j25b/gbov/solar_openEO/slurm_logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -eo pipefail
REPO_DIR="${REPO_DIR:-/gws/ssde/j25b/gbov/solar_openEO}"
cd "$REPO_DIR"
mkdir -p slurm_logs

RUNTIME_LOG="slurm_logs/${SLURM_JOB_NAME:-finetune_reviewed}_${SLURM_JOB_ID:-manual}.runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1
trap 'echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO}" >&2' ERR

# JASMIN orchid uses CUDA 13 by default; pin to legacy 12.8 for our tf-gpu env.
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate tf-gpu
NVIDIA_LD_PATHS=$(find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"

EXP_ID="exp_round3_r101_reviewed"
CONFIG="configs/training/experiments/exp_round3_r101_reviewed.yaml"
RESUME_WEIGHTS="experiments/exp_round3_r101_dropout02/best.weights.h5"

echo "============================================================"
echo "Experiment : $EXP_ID"
echo "Config     : $CONFIG"
echo "Resume     : $RESUME_WEIGHTS"
echo "Job ID     : ${SLURM_JOB_ID:-not_assigned}"
echo "Node       : ${SLURM_NODELIST:-unknown}"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u scripts/train_unet_stage1.py \
    --experiment "$EXP_ID" \
    --config "$CONFIG" \
    --resume "$RESUME_WEIGHTS" \
    -v
