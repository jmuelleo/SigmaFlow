#!/bin/bash -l
#
# Longer overfitting run on the tiny dummy dataset (10 complexes), ~2.75h on
# GPU, to push memorization further than the ~1.75h/300-epoch tests in
# PAUSE-PUNKT #3/#4 before visually checking the sampled pose in PyMOL via
# scripts/sample.py.
#
# Rate estimate from job 8177699/8182812 (PAUSE-PUNKT #3/#4): ~3.4 epochs/min
# on GPU L40S -> ~600 epochs in 2.75h. max_epochs is set a bit above that
# (700) so the SLURM --time limit is what actually cuts the job off, not an
# early max_epochs stop; ModelCheckpoint (save_last=True, top-3 by val loss,
# see scripts/train.py) keeps a usable checkpoint no matter when it's killed.
#
# rot_score_weight=2.0 kept from PAUSE-PUNKT #4 round 2 (best validated so
# far: loss_R clearly below the random-guess baseline, at the cost of a
# somewhat worse loss_trans) - change below if you want to try e.g. 1.0
# instead.
#
#SBATCH --job-name=sigmaflow-overfit-gpu-3h
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:45:00
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
    --max_epochs 700 \
    --early_stopping_patience 0 \
    --rot_score_weight 2.0 \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
