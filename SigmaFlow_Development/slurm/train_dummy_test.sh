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
source activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development

python scripts/train.py \
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