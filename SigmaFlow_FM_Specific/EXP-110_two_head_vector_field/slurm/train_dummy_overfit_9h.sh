#!/bin/bash -l
#
# 9h dummy-overfit run (baseline) - diagnostic, NOT a benchmark.
#
# PURPOSE (STATUS.md, "Hypothese GEPRUEFT" / "Dummy-Langlauf"): separate
# "undertrained" from "structurally broken". Memorising 10 complexes is a
# far easier task than generalising to 209 unseen ones. Primary metric is
# the PER-FRAGMENT ROTATION ERROR against the 126.5 deg uniform-random
# baseline (placement_dummy.py) - NOT transition bond lengths, which
# proved misleading three times over.
#
# Measured at 3h: SigmaFlow 102.0 deg, SigmaDock 109.8 deg vs 126.5 random,
# i.e. barely better than chance even when overfitting 10 complexes.
#
# max_epochs=1800 IS DELIBERATE AND LOAD-BEARING: max_epochs feeds
# max_steps, which feeds the LR scheduler (trainer.py:137-157). The 3h run
# used max_epochs=700 and completed ~88% of its schedule. 9h reaches
# roughly 1800 epochs at the observed ~3.7 epochs/min, so the cosine
# schedule (warmup 1/16 then anneal) completes instead of being truncated.
# Do NOT raise this "for safety" - an unreachably large max_epochs is
# exactly the bug that left every pdbbind run stuck in LR warmup at
# ~10-20% of peak LR.
#
# Usage: sbatch slurm/train_dummy_overfit_9h.sh
#
#SBATCH --job-name=sigmaflow-dummy9h-baseline
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

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

$PYTHON scripts/train.py \n    --data_dir /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/notebooks \n    --train_exps dummy_train \n    --val_exps dummy_train \n    --test_exps dummy_train \n    --batch_size 2 \n    --num_workers 0 \n    --accelerator gpu \n    --devices 1 \n    --max_epochs 1800 \n    --early_stopping_patience 0 \n    --trans_score_weight 2.0 \n    --rot_score_weight 0.5 \n    --offline_run \n    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 9h dummy overfit (baseline), max_epochs=1800, trans_score_weight=2.0 rot_score_weight=0.5"
