#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=eval_ensemble
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -eo pipefail

REPO_DIR="/gws/nopw/j04/gbov/solar_openEO"
cd "$REPO_DIR"
mkdir -p slurm_logs

RUNTIME_LOG="slurm_logs/${SLURM_JOB_NAME:-eval_ensemble}_${SLURM_JOB_ID:-manual}.runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate tf-gpu

NVIDIA_LD_PATHS=$(
  find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true
)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"

TOP=${TOP:-15}
MIN_DICE=${MIN_DICE:-0.800}

echo "============================================================"
echo "Ensemble Evaluation"
echo "Top-N        : $TOP"
echo "Min Dice     : $MIN_DICE"
echo "Job ID       : ${SLURM_JOB_ID:-not_assigned}"
echo "Node         : ${SLURM_NODELIST:-unknown}"
echo "Python       : $(command -v python)"
python --version
echo "============================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo ""
echo "=== Step 1: TTA on best single model ==="
python scripts/evaluate_training_run.py \
  --exp experiments/exp_stage1_r101_dice_zscore_longer \
  --tta

echo ""
echo "=== Step 2: Ensemble (no TTA) ==="
python scripts/evaluate_ensemble.py \
  --min-dice "$MIN_DICE" \
  --top "$TOP"

echo ""
echo "=== Step 3: Ensemble + TTA ==="
python scripts/evaluate_ensemble.py \
  --min-dice "$MIN_DICE" \
  --top "$TOP" \
  --tta

echo ""
echo "Done."
