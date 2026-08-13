#!/bin/bash -l
#
# 3h confirmation run on the 10 dummy complexes, using the hyperparameter
# combination chosen after the wide 1-2h screening sweep (PAUSE-PUNKT #11):
# trans_score_weight=2.0, rot_score_weight=0.5, rot_vector_field_scaling=rms
# - i.e. SigmaDock's own production recipe (conf/training/slurm.yaml),
# which the sweep's averaged (8x repeated validation) results gave no
# reason to deviate from for SigmaFlow either. This supersedes
# train_dummy_overfit_gpu_3h.sh, which used the ad-hoc rot_score_weight=2.0
# found in an earlier, much smaller informal test (PAUSE-PUNKT #4) - the
# sweep showed that value actually hurts loss_trans without improving
# loss_R at all.
#
#SBATCH --job-name=sigmaflow-overfit-prodhparams-gpu-3h
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

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
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
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --rot_vector_field_scaling rms \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
echo "Ran with: trans_score_weight=2.0 rot_score_weight=0.5 rot_vector_field_scaling=rms"
