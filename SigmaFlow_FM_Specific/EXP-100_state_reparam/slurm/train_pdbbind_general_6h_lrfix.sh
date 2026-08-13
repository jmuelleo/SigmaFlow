#!/bin/bash -l
#
# 6h pdbbind-general run with a CORRECTED learning-rate schedule.
#
# THE BUG THIS FIXES (STATUS.md, "KONFIGURATIONSFEHLER GEFUNDEN"):
# max_epochs does not only cap runtime - it feeds max_steps
# (scripts/train.py:218-222), which feeds the LR scheduler
# (trainer.py:137-157: LinearLR warmup over lr_warmup_frac=1/16 of
# max_steps, then cosine annealing). With the previous --max_epochs 1000
# on 19,443 complexes at batch 8, max_steps = 2,430,375 and warmup alone
# lasts 151,898 steps - but 6h only reaches ~7,750 steps. Every pdbbind run
# so far (6h/12h/24h, SigmaFlow AND SigmaDock) therefore spent its entire
# life inside LR warmup at 5-20% of the intended peak LR (1e-4) and never
# reached annealing at all.
#
# WHY max_epochs=3: 6h reaches ~7,750 steps ~= 3.2 epochs, so 3 makes the
# schedule fit the actual compute budget (warmup 455 steps, then anneal).
# This mirrors the PAPER's own setup, which is self-consistent: Appendix
# E.3 specifies 256 epochs with warmup over "the first 16 epochs", i.e.
# exactly the lr_warmup_frac=1/16 default. The default was calibrated for
# max_epochs=256; our 1000 broke that relationship.
# DO NOT raise max_epochs "for safety" - that is precisely the bug.
#
# COMPARISON TARGET: the existing 12h broken-schedule run scored 122.2 deg
# mean per-fragment rotation error on the full 209-complex PoseBusters set
# (random baseline = 126.5 deg). If this 6h run beats that with HALF the
# compute, the LR bug is confirmed as materially significant.
#
# Usage: sbatch slurm/train_pdbbind_general_6h_lrfix.sh
#
#SBATCH --job-name=sigmaflow-pdbbind-6h-lrfix
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
    --max_epochs 3 \
    --val_check_interval 50 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 6h pdbbind-general, LR-FIXED (max_epochs=3), trans_score_weight=2.0 rot_score_weight=0.5"
