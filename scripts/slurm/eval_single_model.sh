#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=eval_single
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -eo pipefail
REPO_DIR="/gws/nopw/j04/gbov/solar_openEO"
cd "$REPO_DIR"
mkdir -p slurm_logs

exec > >(tee -a "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.runtime.log") 2>&1

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate tf-gpu
NVIDIA_LD_PATHS=$(find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"

EXP=${EXP:-experiments/exp_stage1_r101_dice_zscore_longer}

echo "=== No TTA ==="
python scripts/evaluate_training_run.py --exp "$EXP"

echo ""
echo "=== With TTA ==="
python scripts/evaluate_training_run.py --exp "$EXP" --tta
