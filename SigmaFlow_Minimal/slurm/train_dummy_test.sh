#!/bin/bash -l
#
# First sanity-check training run on the tiny dummy dataset (10 complexes).
# Goal: prove the training pipeline runs end-to-end, not to train a good model.
#
#SBATCH --job-name=sigmaflow-dummy-test
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script. SLURM opens the --output/--error files before any line of this
# script runs, so `mkdir -p slurm_logs` inside the script itself is too late.
# Run `mkdir -p slurm_logs` yourself once, from this directory, first.

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development

# `conda activate` in a login-shell batch script has proven unreliable here
# (CONDA_PREFIX gets set correctly, but PATH ends up pointing at the base
# Mamba module's python instead of this environment's). Bypass PATH lookup
# entirely and call this environment's interpreter by its absolute path.
PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# Diagnostics: confirm the right environment is actually active before
# running anything real. If this prints the wrong path/version, something
# is still wrong and nothing downstream can be trusted.
echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
echo "prefix:  $CONDA_PREFIX"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/notebooks \
    --train_exps dummy_train \
    --val_exps dummy_train \
    --test_exps dummy_train \
    --batch_size 2 \
    --num_workers 0 \
    --accelerator cpu \
    --devices 1 \
    --max_steps 5 \
    --offline_run \
    --debug