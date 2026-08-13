#!/bin/bash -l
#
# Loss-variant experiment (STATUS.md "c_anchor_atom_distance_loss"): tests
# whether a direct anchor-atom distance loss across each cut torsional bond
# closes the PoseBusters chemical-plausibility / transition-bond-length gap
# between SigmaFlow and SigmaDock. Unlike the two prior variants
# (a_time_weighting: reweight the existing velocity loss over t;
# b_rotation_data_space_loss: rotation-only data-space term, also just a
# disguised reweighting to leading order per the BCH derivation in
# STATUS.md) - this term directly penalises the predicted vs. true distance
# between the two anchor atoms of each cut bond, the exact quantity the
# external RDKit "transition bond" diagnostic measures. See
# sigma_flow_generator.py::_compute_anchor_distance_loss for the
# construction.
#
# Fresh 12h run from scratch on the real pdbbind-general dataset, identical
# hyperparameters to the existing (unmodified) SigmaFlow 6h/12h/24h runs and
# the two prior variants, so the result is directly comparable against the
# existing SigmaFlow-12h full-PoseBusters-set numbers already recorded in
# STATUS.md/RESULTS.md - no separate unmodified-SigmaFlow control run
# needed for this comparison.
#
# IMPORTANT: this must run from THIS folder's own src/, not the unmodified
# SigmaFlow_Development install that sigmaflow_env's site-packages points
# to by default - hence the explicit PYTHONPATH override below (same fix
# used for SigmaFlow_Variants/a_time_weighting/ and b_rotation_data_space_loss/).
#
# Usage: sbatch slurm/train_pdbbind_general_12h.sh
#
#SBATCH --job-name=sigmaflow-anchordist-pdbbind-general-12h
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/c_anchor_atom_distance_loss

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# Force this folder's own src/ ahead of sigmaflow_env's installed
# SigmaFlow_Development copy - without this, `import sigmadock` would
# silently load the UNMODIFIED code and this run would be a no-op.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

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
echo "Ran with: train=pdbbind-general val/test=posebusters, trans_score_weight=2.0 rot_score_weight=0.5, fresh 12h from scratch, anchor-atom distance loss variant"
