#!/bin/bash -l
#
# Diagnostic: run scripts/diagnose_step0.py, which manually replicates sampler()'s
# very first step and checks isnan/isinf SEPARATELY at each stage (raw network output,
# aggregated force/torque, predicted vector field, true vector field) to pinpoint exactly
# why diagnose_vector_field.py showed inf/nan at step 0.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/diagnose_step0.sh
#
#SBATCH --job-name=sigmaflow-diagnose-step0
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to your checkpoint path, e.g. experiments/sigmadock/<timestamp>/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/notebooks}"
T_MIN="${T_MIN:-0.01}"

echo "python:   $PYTHON"
echo "ckpt:     $CKPT_DIR"
echo "data_dir: $DATA_DIR"
echo "t_min:    $T_MIN"

$PYTHON scripts/diagnose_step0.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=dummy_train \
    graph.sample_conformer=false \
    ode.t_min="${T_MIN}" \
    hydra.run.dir="$(pwd)/sampling_output/hydra_out_diagnose_step0"
