#!/bin/bash -l
#
# 9h dummy-overfit CONTROL run for the ORIGINAL SigmaDock (diffusion), to be
# placed in the separate ARC SigmaDock clone (there is no local copy of that
# tree in this repo - see STATUS.md PAUSE-PUNKT #9).
#
# WHY A CONTROL IS ESSENTIAL HERE: the diagnostic question is whether a model
# can learn to ORIENT fragments at all when merely memorising 10 complexes.
# Without SigmaDock run under identical conditions we cannot tell a
# SigmaFlow-specific failure from one inherited from the shared architecture
# / data pipeline / LR schedule. At 3h both were near the random baseline
# (SigmaFlow 102.0 deg, SigmaDock 109.8 deg vs 126.5 deg random), which is
# precisely why this must be re-checked for both at 9h.
#
# NOTE the flag difference vs. the SigmaFlow scripts: --rot_score_method and
# --rot_score_scaling are REAL, live flags in SigmaDock (conf/training/
# slurm.yaml) - unlike SigmaFlow's removed --rot_vector_field_scaling. Do not
# "harmonise" these away.
#
# max_epochs=1800 is deliberate and load-bearing - it feeds max_steps which
# feeds the LR scheduler. An unreachably large value is exactly the bug that
# left every pdbbind run stuck in LR warmup at ~10-20% of peak LR.
#
# Usage (from the SigmaDock clone root): sbatch slurm/train_dummy_overfit_9h_sigmadock.sh
#
#SBATCH --job-name=sigmadock-dummy9h-control
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock/notebooks \
    --train_exps dummy_train \
    --val_exps dummy_train \
    --test_exps dummy_train \
    --batch_size 2 \
    --num_workers 0 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 1800 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --rot_score_method space \
    --rot_score_scaling rms \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 9h dummy overfit (SigmaDock control), max_epochs=1800"
