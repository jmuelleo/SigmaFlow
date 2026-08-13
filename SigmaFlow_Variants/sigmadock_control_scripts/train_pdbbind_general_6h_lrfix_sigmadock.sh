#!/bin/bash -l
#
# 6h pdbbind-general CONTROL run for the ORIGINAL SigmaDock (diffusion),
# with the CORRECTED learning-rate schedule. To be placed in the separate
# ARC SigmaDock clone (no local copy exists in this repo, see STATUS.md
# PAUSE-PUNKT #9).
#
# THE BUG THIS FIXES (STATUS.md, "KONFIGURATIONSFEHLER GEFUNDEN"):
# max_epochs feeds max_steps, which feeds the LR scheduler (warmup over
# lr_warmup_frac=1/16 of max_steps, then cosine annealing). With the
# previous --max_epochs 1000 on 19,443 complexes at batch 8, warmup alone
# lasts 151,898 steps while 6h only reaches ~7,750 - so every pdbbind run
# so far stayed inside warmup at 5-20% of peak LR (1e-4). This affects
# SigmaDock exactly as much as SigmaFlow: both use the same train.py and
# the same config defaults. That is why the control must be re-run too -
# without it we cannot attribute any change to the method rather than to
# the schedule fix.
#
# WHY max_epochs=3: 6h reaches ~7,750 steps ~= 3.2 epochs. This restores
# the paper's own self-consistent setup (Appendix E.3: 256 epochs, warmup
# over "the first 16 epochs" = exactly lr_warmup_frac=1/16).
# DO NOT raise max_epochs "for safety" - that is precisely the bug.
#
# NOTE --rot_score_method / --rot_score_scaling are REAL, live flags in
# SigmaDock (verified against the known-good train_pdbbind_general_24h.sh
# in this clone), unlike SigmaFlow's removed --rot_vector_field_scaling.
# Do not "harmonise" these away.
#
# Usage (from the SigmaDock clone root):
#   sbatch slurm/train_pdbbind_general_6h_lrfix_sigmadock.sh
#
#SBATCH --job-name=sigmadock-pdbbind-6h-lrfix
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

# ENVIRONMENT: SigmaDock runs in myenv, NOT sigmaflow_env, activated with
# `source activate` and using the plain `python` on PATH. This is NOT
# cosmetic: sigmaflow_env has SigmaFlow_Development installed into its
# site-packages, so under that env `import sigmadock` resolves to the
# SigmaFlow package and dies with
#   ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'
# (SigmaFlow has sigma_flow_generator.py, not denoiser.py). That exact
# mistake killed job 8512799 after 5:44. This block is copied verbatim from
# the known-good train_pdbbind_general_24h.sh in this same clone - do not
# "modernise" it to match the SigmaFlow scripts.
module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

echo "python:  $(which python)"
echo "version: $(python --version)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
python -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

python scripts/train.py \
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
    --rot_score_method space \
    --rot_score_scaling rms \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 6h pdbbind-general SigmaDock CONTROL, LR-FIXED (max_epochs=3)"
