#!/bin/bash
# Submit the improvement sweep:
#   - Augmentation ablation (noise / jitter / scale / full) on R101
#   - Full augmentation on R50
#   - Wide decoder on R101 and R50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_LOG="$REPO_DIR/experiments/runs.csv"
SLURM_SCRIPT="$REPO_DIR/scripts/slurm/train_stage1_gpu.sh"
EXTRA_ARGS=${EXTRA_ARGS:--v}
DRY_RUN=${DRY_RUN:-0}
RESUBMIT_EXISTING=${RESUBMIT_EXISTING:-0}

cd "$REPO_DIR"
mkdir -p slurm_logs experiments

if [[ ! -f "$RUN_LOG" ]]; then
  printf "timestamp,exp_id,config,slurm_job_id,description\n" >"$RUN_LOG"
fi

should_skip_experiment() {
  local exp_id=$1
  local exp_dir="$REPO_DIR/experiments/$exp_id"

  if [[ "$RESUBMIT_EXISTING" == "1" ]]; then
    return 1
  fi

  if timeout 15 squeue -h --name="$exp_id" -o '%j' 2>/dev/null | grep -Fxq "$exp_id"; then
    echo "Skipping $exp_id: already present in Slurm queue."
    return 0
  fi

  if [[ -f "$exp_dir/best.weights.h5" || -f "$exp_dir/latest.weights.h5" ]]; then
    echo "Skipping $exp_id: existing experiment checkpoints found in $exp_dir."
    return 0
  fi

  if awk -F, -v exp_id="$exp_id" 'NR > 1 && $2 == exp_id && $4 != "dry-run" {found=1} END {exit(found ? 0 : 1)}' "$RUN_LOG"; then
    echo "Skipping $exp_id: already recorded in $RUN_LOG."
    return 0
  fi

  return 1
}

# Augmentation ablation on R101 + R50 full; wide decoder on R101 + R50
GRID=$(cat <<'EOF'
exp_stage1_r101_dice_zscore_longer_aug_noise,configs/training/experiments/exp_stage1_r101_dice_zscore_longer_aug_noise.yaml,"R101 + Gaussian noise augmentation"
exp_stage1_r101_dice_zscore_longer_aug_jitter,configs/training/experiments/exp_stage1_r101_dice_zscore_longer_aug_jitter.yaml,"R101 + per-band brightness jitter augmentation"
exp_stage1_r101_dice_zscore_longer_aug_scale,configs/training/experiments/exp_stage1_r101_dice_zscore_longer_aug_scale.yaml,"R101 + scale-jitter (random crop+resize) augmentation"
exp_stage1_r101_dice_zscore_longer_aug_full,configs/training/experiments/exp_stage1_r101_dice_zscore_longer_aug_full.yaml,"R101 + full augmentation (noise+jitter+scale)"
exp_stage1_r50_dice_zscore_longer_aug_full,configs/training/experiments/exp_stage1_r50_dice_zscore_longer_aug_full.yaml,"R50 + full augmentation (noise+jitter+scale)"
exp_stage1_r101_dice_zscore_longer_wide,configs/training/experiments/exp_stage1_r101_dice_zscore_longer_wide.yaml,"R101 + wide decoder (512,256,128,64,32)"
exp_stage1_r50_dice_zscore_longer_wide,configs/training/experiments/exp_stage1_r50_dice_zscore_longer_wide.yaml,"R50 + wide decoder (512,256,128,64,32)"
EOF
)

while IFS=, read -r EXP_ID CONFIG DESC; do
  [[ -z "${EXP_ID// }" ]] && continue
  DESC=${DESC%\"}
  DESC=${DESC#\"}

  if should_skip_experiment "$EXP_ID"; then
    continue
  fi

  CMD=(env "EXP_ID=${EXP_ID}" "CONFIG_PATH=${CONFIG}" "EXTRA_ARGS=${EXTRA_ARGS}" sbatch "--job-name=${EXP_ID}" "$SLURM_SCRIPT")

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] ${CMD[*]}"
    JOB_ID="dry-run"
  else
    SUBMIT_OUTPUT="$("${CMD[@]}")"
    echo "$SUBMIT_OUTPUT"
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
  fi

  TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  printf "%s,%s,%s,%s,%s\n" "$TIMESTAMP" "$EXP_ID" "$CONFIG" "$JOB_ID" "$DESC" >>"$RUN_LOG"
done <<<"$GRID"

echo "Run metadata appended to $RUN_LOG"
echo ""
echo "To also run ensemble + TTA on existing weights (no GPU needed, interactive):"
echo "  python scripts/evaluate_ensemble.py --min-dice 0.800 --top 10"
echo "  python scripts/evaluate_ensemble.py --min-dice 0.800 --top 10 --tta"
