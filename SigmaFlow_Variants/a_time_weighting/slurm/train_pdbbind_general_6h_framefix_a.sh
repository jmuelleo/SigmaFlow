#!/bin/bash -l
#
# 6h pdbbind-general run: VARIANTE A (time weighting) + FRAME FIX
#
# WHAT THIS VARIANT CHANGES
# compute_losses multiplies BOTH losses by w(t) = 1/(1-min(t,0.9))^2,
# up-weighting late times (Yim et al. 2023-inspired clipping).
#
# RELATION TO THE FRAME FIX: variant A is frame-AGNOSTIC - it is a scalar
# factor on whatever the loss already is. Its earlier null result is
# therefore not explained by the frame bug directly, but it was measured on
# top of a rotation loss that compared a WORLD-frame prediction against a
# BODY-frame target, i.e. against essentially noise. Re-testing on the fixed
# base is the first meaningful test of this variant.
#
# CAUTION when reading the result: the weight spans 1..100. It rescales the
# effective learning rate over time, so a null result here may be an
# optimisation artefact rather than a statement about time weighting.
#
# NOTE: this folder has only ever been run on the 10 dummy complexes.
# Upload it to ARC before submitting, and verify `sigmadock loaded from`.
#
# LR SCHEDULE: --max_epochs 3 is calibrated to the 6h budget (~7,750 steps
# ~= 3.2 epochs). max_epochs feeds max_steps which feeds the LR scheduler;
# an unreachably large value leaves the run stuck in warmup. DO NOT raise it.
#
# CONTROL GROUP is job 8530243 (frame fix, no loss change, 6h, max_epochs=3,
# 7050 steps). Compare against THAT, not against the old 8512798 control -
# otherwise the frame fix and the loss variant are confounded.
#
# SUCCESS CRITERION: the locality gap (bonded minus non-bonded relative
# rotation error) must beat 8530243's -12.8 deg [CI -18.3, -7.2].
# Measure with SigmaFlow_Variants/posebusters_full_comparison/
# fragment_locality.py after sampling the full 209-complex set.
#
# Usage: sbatch slurm/train_pdbbind_general_6h_framefix_a.sh
#
#SBATCH --job-name=sigmaflow-a-timeweight-framefix-6h
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

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/a_time_weighting

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# --- wandb robustness (job 8525523 died here after wasting 36 min) ---
# That run reached wandb.init() only AFTER building the 19k-complex datafront,
# then failed with:
#   ServicePollForTokenError: Failed to read port info after 30.0 seconds
#     (wandb-core PID=...): /tmp/tmpXXXX/port-XXXX.txt
# i.e. wandb's helper process could not publish its port file on the node's
# shared /tmp within the default 30 s timeout. Three mitigations:
#   1) raise that exact timeout,
#   2) keep wandb off the shared /tmp entirely,
#   3) probe wandb BEFORE the expensive dataset setup (see preflight below),
#      so a bad node costs one minute instead of thirty-six.
export WANDB__SERVICE_WAIT=300
export TMPDIR="/data/stat-cadd/shug8458/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"

# Force this folder's own src/ ahead of sigmaflow_env's installed
# SigmaFlow_Development copy - without this, `import sigmadock` would
# silently load the UNMODIFIED code and this run would be a no-op.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

# Preflight: fail fast if wandb cannot start its service on THIS node.
# Costs ~1 min; without it a bad node is only discovered after the ~36 min
# datafront build, as happened to job 8525523.
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

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
echo "Ran: 6h pdbbind-general, VARIANTE A (time weighting) + FRAME FIX, LR-fixed (max_epochs=3)"
