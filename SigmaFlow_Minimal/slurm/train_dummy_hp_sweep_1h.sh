#!/bin/bash -l
#
# Short (1h) hyperparameter-screening run on the 10 dummy complexes, for
# comparing trans_score_weight / rot_score_weight combinations against each
# other. NOT meant to produce a final, fully-converged checkpoint (at ~3.4
# epochs/min that's only ~200 epochs, vs. ~600 in the 3h runs) - meant to
# cheaply screen for the right DIRECTION before committing a full 3h budget
# to the winning combination. See SigmaFlow_Development/STATUS.md
# PAUSE-PUNKT #10/#11 for context (SigmaDock's own production recipe uses
# trans_score_weight=2.0, rot_score_weight=0.5 - opposite ratio from what
# our own earlier small-scale test found better for SigmaFlow).
#
# Usage (submit once per combination, override job-name for clarity):
#   TRANS_WEIGHT=1.0 ROT_WEIGHT=1.0 \
#     sbatch --job-name=sigmaflow-hp-t1-r1 slurm/train_dummy_hp_sweep_1h.sh
#
#SBATCH --job-name=sigmaflow-hp-sweep-1h
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

TRANS_WEIGHT="${TRANS_WEIGHT:-1.0}"
ROT_WEIGHT="${ROT_WEIGHT:-2.0}"

echo "python:       $PYTHON"
echo "trans_weight: $TRANS_WEIGHT"
echo "rot_weight:   $ROT_WEIGHT"
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
    --trans_score_weight "${TRANS_WEIGHT}" \
    --rot_score_weight "${ROT_WEIGHT}" \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout above for the exact timestamp)"
echo "Ran with: trans_score_weight=${TRANS_WEIGHT} rot_score_weight=${ROT_WEIGHT}"
