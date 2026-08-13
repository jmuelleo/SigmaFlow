#!/bin/bash -l
#
# SigmaDock (original diffusion) counterpart to train_pdbbind_general_6h.sh,
# for the first staged big-run comparison (STATUS.md PAUSE-PUNKT #14). Same
# data (train: pdbbind/general-set/, val/test: posebusters benchmark), same
# 6h budget, SigmaDock's OWN production hyperparameters (verified 2026-08-05
# directly from SigmaDock/conf/training/slurm.yaml, the local read-only
# reference copy: trans_score_weight=2.0, rot_score_weight=0.5,
# rot_score_method=space, rot_score_scaling=rms - NOT copied from SigmaFlow's
# sweep, this is SigmaDock's own file).
#
# Runs in the SEPARATE ARC clone used for prior SigmaDock comparisons (see
# STATUS.md PAUSE-PUNKT #9), NOT the local read-only SigmaDock/ reference in
# this repo (which CLAUDE.md forbids running/modifying) - own git repo, own
# conda env `myenv`.
#
# NOT VERIFIED, please check before submitting:
#   - Whether conf/experiments/pdbbind-general.yaml and posebusters.yaml
#     already exist in that clone. PAUSE-PUNKT #9 had to manually add
#     dummy_train.yaml there (it was missing) - these two may or may not
#     already be present. If missing, copy the content from this repo's
#     SigmaFlow_MinimalChange/conf/experiments/{pdbbind-general,posebusters}.yaml
#     (byte-identical to SigmaDock's own, confirmed in AUDIT_SigmaDock_vs_SigmaFlow.txt).
#   - The exact python interpreter path for the `myenv` environment - unlike
#     sigmaflow_env, no absolute .../bin/python path has been confirmed for
#     myenv in this project's history, so this script relies on `python`
#     being on PATH after `source activate` (standard conda behaviour, but
#     double check the "python loaded from" sanity check line below actually
#     resolves to something in myenv, not a system Python).
#   - Same partition/time question as the SigmaFlow script: `short` allows
#     up to 12h (confirmed by user 2026-08-05), so 6h is fine.
#
#SBATCH --job-name=sigmadock-pdbbind-general-6h
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

# Fail loudly instead of silently: without this, a crashed/erroring command
# doesn't stop the script, and the final `echo` below still exits 0 - SLURM
# then reports COMPLETED even though nothing actually ran (see job 8443875,
# the SigmaFlow sibling script's first submission).
set -euo pipefail

module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

echo "python:  $(which python)"
echo "version: $(python --version)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
python -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"
# Sanity check: must print a path under this clone
# (SigmaDock_Reproduction_JulianMueller/sigmadock/...), and must be the
# DIFFUSION denoiser.py, not accidentally the flow-matching sigma_flow_generator.py.

python scripts/train.py \
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
    --rot_score_method space \
    --rot_score_scaling rms \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
echo "Ran with: train=pdbbind-general val/test=posebusters, SigmaDock's own production hyperparameters (conf/training/slurm.yaml)"
