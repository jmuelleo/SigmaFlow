#!/bin/bash -l
#
# Re-validates each hyperparameter-sweep checkpoint to extract the real
# unweighted loss_val/loss_trans and loss_val/loss_R (needed because
# loss_val/total isn't comparable across different trans/rot_score_weight
# configs, and the offline WandB summary files for these runs were never
# written since SLURM TIMEOUT killed them before a clean finish).
#
#SBATCH --job-name=sigmaflow-eval-hp-sweep
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
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
$PYTHON eval_unweighted_losses.py
