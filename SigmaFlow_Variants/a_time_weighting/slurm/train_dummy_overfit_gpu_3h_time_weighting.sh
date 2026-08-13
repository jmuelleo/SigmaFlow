#!/bin/bash -l
#
# Variant (a) of the Test-3 bond-length ablation (see STATUS.md PAUSE-PUNKT
# #14): identical 3h confirmation run to
# train_dummy_overfit_gpu_3h_prodhparams.sh (same production hyperparameters,
# same 10 dummy complexes, same time budget), but using the code in THIS
# folder (SigmaFlow_Variants/a_time_weighting/), which adds a time-dependent
# weight to compute_losses() (SigmaFlowGenerator._time_weight, clipped
# 1/(1-t)^2, cap_t=0.9) - an explicit, unproven EMPIRICAL EXPLORATION, not a
# theoretically derived fix (see STATUS.md for why: no clean support for
# this in the flow-matching literature that was checked).
#
# Compare the resulting checkpoint's sampled bond lengths (same RDKit
# methodology as Test 1 / AUDIT_SigmaDock_vs_SigmaFlow.txt Teil 5.2/9)
# against the unweighted SigmaFlow baseline (sampling_output_prodhparams)
# and SigmaDock (sampling_output_sigmadock_prodhparams).
#
#SBATCH --job-name=sigmaflow-timeweighting-gpu-3h
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

# Fail loudly instead of silently: without this, a crashed/erroring command
# (e.g. an argparse error from a stale CLI flag) doesn't stop the script,
# and the final `echo` below still exits 0 - SLURM then reports COMPLETED
# even though nothing actually ran. Exactly what happened on the first
# submission of this script (2026-08-05, job 8443875): a leftover
# --rot_vector_field_scaling flag (removed from config.py in an earlier
# cleanup, see STATUS.md PAUSE-PUNKT #12) made `scripts/train.py` exit
# immediately with an argparse error, but the job still showed
# COMPLETED/0:0 after only ~5 minutes.
set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Variants/a_time_weighting

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# IMPORTANT: sigmaflow_env has `sigmadock` pip-installed (editable) pointing
# at SigmaFlow_Development/src by default. Without this override, the job
# would silently train the UNMODIFIED baseline code instead of this
# variant's compute_losses() change. Prepending this folder's own src/ to
# PYTHONPATH makes Python resolve `import sigmadock` here first.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"
# Sanity check: the line above MUST print a path under this job's own
# SigmaFlow_Variants/a_time_weighting/src/ - if it prints a path under
# SigmaFlow_Development/ instead, stop the job and fix PYTHONPATH first.

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Variants/a_time_weighting/notebooks \
    --train_exps dummy_train \
    --val_exps dummy_train \
    --test_exps dummy_train \
    --batch_size 2 \
    --num_workers 0 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 700 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
echo "Ran with: trans_score_weight=2.0 rot_score_weight=0.5 (rot_vector_field_scaling removed - dead CLI flag, see STATUS.md PAUSE-PUNKT #12) + variant (a) time-weighted loss (_time_weight, cap_t=0.9)"
