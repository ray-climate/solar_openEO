#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=solar_unet
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -eo pipefail

REPO_DIR="/gws/nopw/j04/gbov/solar_openEO"
cd "$REPO_DIR"
mkdir -p slurm_logs

RUNTIME_LOG="slurm_logs/${SLURM_JOB_NAME:-solar_unet}_${SLURM_JOB_ID:-manual}.runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1
trap 'echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO}" >&2' ERR

set +u
source /home/users/ruisong/mambaforge_new/etc/profile.d/conda.sh
conda activate tf-gpu

# Keep CUDA/NVIDIA library path discovery best-effort so preflight does not fail silently.
NVIDIA_LD_PATHS=$(
  find "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':' || true
)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"

EXP_ID=${EXP_ID:-manual_stage1}
CONFIG_PATH=${CONFIG_PATH:-configs/training/experiments/exp_stage1_resnet50.yaml}
EXTRA_ARGS=${EXTRA_ARGS:-"-v"}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "============================================================"
echo "Experiment ID : $EXP_ID"
echo "Config        : $CONFIG_PATH"
echo "Extra args    : $EXTRA_ARGS"
echo "Job ID        : ${SLURM_JOB_ID:-not_assigned}"
echo "Node          : ${SLURM_NODELIST:-unknown}"
echo "Conda prefix  : ${CONDA_PREFIX:-unset}"
echo "Python        : $(command -v python)"
python --version
echo "============================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found in PATH" >&2
fi

python scripts/train_unet_stage1.py --experiment "$EXP_ID" --config "$CONFIG_PATH" $EXTRA_ARGS
