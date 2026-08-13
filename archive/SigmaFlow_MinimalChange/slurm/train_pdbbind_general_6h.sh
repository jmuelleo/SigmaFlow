#!/bin/bash -l
#
# First step of the staged big-run plan (STATUS.md PAUSE-PUNKT #14): 6h run
# on a REAL dataset (train: pdbbind/general-set/, ~19k complexes, confirmed
# populated and matching the config regex 2026-08-05; val/test:
# posebusters_paper/posebusters_benchmark_set/, ~209 complexes, confirmed
# populated). This is the first time SigmaFlow trains/validates on anything
# beyond the 10 memorized dummy complexes.
#
# Deliberately uses the SAME validated production hyperparameters as
# train_dummy_overfit_gpu_3h_prodhparams.sh (trans_score_weight=2.0,
# rot_score_weight=0.5), NOT the time-weighted
# loss from SigmaFlow_Variants/a_time_weighting/ - this run is isolating
# "does the (already-validated) method generalize beyond 10 complexes", a
# different question from "does time-weighting fix fragment-junction bond
# lengths". Keep these two experiments independent; don't confound them.
#
# pdbbind/refined-set/ and pdbbind/core-set/ exist as directories on ARC but
# are EMPTY (never extracted from raw/P-L.tar.gz) - confirmed 2026-08-05,
# hence general-set + posebusters instead of the more conventional
# refined-set(train)/core-set(val) split.
#
# NOT VERIFIED, please check before submitting:
#   - Whether the `short` partition actually allows a 6h job (the 3h dummy
#     run worked on `short`; 6h is unconfirmed - the pending 3h job's own
#     queue wait already showed `short` can be busy/slow, so this may need a
#     different partition or a longer queue wait. Ask ARC support or check
#     `sinfo`/partition docs if unsure).
#   - --val_check_interval=50 (lowered from an initial guess of 200 on user
#     request, to bound potential lost progress on a SLURM timeout/crash -
#     see STATUS.md PAUSE-PUNKT #14, ModelCheckpoint(save_last=True) only
#     saves on each validation, and no SLURMEnvironment/signal-based
#     graceful-shutdown-on-timeout was found in scripts/train.py) and
#     --batch_size=8 are starting GUESSES, not measured - general-set
#     complexes are larger/more complex than the 10 dummy ones, so per-step
#     time and memory headroom are unknown until this actually runs. If
#     step 0 OOMs, lower --batch_size first. --resume_from_checkpoint exists
#     if a later stage (12h/24h) needs to continue from this run's last
#     checkpoint instead of restarting.
#
# CORRECTED 2026-08-05 after the first submission of the sibling script
# (train_dummy_overfit_gpu_3h_time_weighting.sh, job 8443875) failed
# immediately with an argparse error: --rot_vector_field_scaling was
# removed from config.py in an earlier cleanup (STATUS.md PAUSE-PUNKT #12,
# dead/no-op flag) but was still copied into these scripts from the older
# train_dummy_overfit_gpu_3h_prodhparams.sh template. Removed here before
# this script was ever submitted.
#
#SBATCH --job-name=sigmaflow-pdbbind-general-6h
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
# then reports COMPLETED even though nothing actually ran (see job 8443875).
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
echo "Ran with: train=pdbbind-general val/test=posebusters, trans_score_weight=2.0 rot_score_weight=0.5 (unweighted loss, same as sampling_output_prodhparams)"
