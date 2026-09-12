#!/bin/bash -l
#
# Continue the 6h pdbbind-general run (train_pdbbind_general_6h.sh, job
# 8447572) for another 6h - cumulative 12h training, second stage of the
# staged big-run plan (STATUS.md PAUSE-PUNKT #14). Resumes optimizer state,
# epoch count, and W&B run via --resume_from_checkpoint, writing into the
# SAME experiment directory as the checkpoint (see scripts/train.py's
# get_exp_dir_from_ckpt) rather than starting a new one.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/train_pdbbind_general_resume_6more.sh
#
#SBATCH --job-name=sigmaflow-pdbbind-general-resume6h
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint to resume from, e.g. experiments/sigmadock/<timestamp>/checkpoints/last.ckpt}"

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
echo "ckpt:    $CKPT_DIR"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/data \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size 8 \
    --num_workers 6 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 1000 \
    --val_check_interval 50 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --resume_from_checkpoint "${CKPT_DIR}" \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp - should match the resumed experiment dir)"
echo "Ran with: train=pdbbind-general val/test=posebusters, resumed from ${CKPT_DIR} for another 6h (cumulative 12h)"
