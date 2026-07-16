#!/bin/bash -l
#
# GPU smoke test + overfitting sanity check on the tiny dummy dataset (10
# complexes), before committing to the multi-day real training run.
# Goal: (1) prove the pipeline works on a real GPU, not just CPU, and
# (2) prove the model can actually drive the loss down given enough steps
# on a tiny dataset it can memorize. NOT meant to produce a useful model.
#
#SBATCH --job-name=sigmaflow-overfit-gpu-test
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:45:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development

# Bypass PATH-ordering issues seen with conda activate in a login-shell
# batch script (see train_dummy_test.sh) - call the interpreter directly.
PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
echo "prefix:  $CONDA_PREFIX"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/notebooks \
    --train_exps dummy_train \
    --val_exps dummy_train \
    --test_exps dummy_train \
    --batch_size 2 \
    --num_workers 0 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 300 \
    --early_stopping_patience 0 \
    --offline_run \
    --debug
