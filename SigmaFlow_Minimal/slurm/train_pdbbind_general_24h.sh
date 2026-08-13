#!/bin/bash -l
#
# Third stage of the staged big-run plan (STATUS.md PAUSE-PUNKT #14):
# fresh 24h run from scratch (NOT --resume_from_checkpoint) on the real
# pdbbind-general dataset, to test whether the 6h/12h trade-off result
# (SigmaFlow: fewer transition bonds; SigmaDock: smaller deviations when
# they occur) persists at 24h. Independent training run, own new
# experiment directory - not a continuation of the 6h/12h checkpoints
# (those were already overwritten by the 6h->12h resume anyway, see
# STATUS.md).
#
# Same production hyperparameters as train_pdbbind_general_6h.sh
# (trans_score_weight=2.0, rot_score_weight=0.5). 24h exceeds the `short`
# partition's 12h limit, so this uses `medium` (48h max, confirmed
# 2026-08-05).
#
# Usage: sbatch slurm/train_pdbbind_general_24h.sh
#
#SBATCH --job-name=sigmaflow-pdbbind-general-24h
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
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
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
echo "Ran with: train=pdbbind-general val/test=posebusters, trans_score_weight=2.0 rot_score_weight=0.5, fresh 24h from scratch"
