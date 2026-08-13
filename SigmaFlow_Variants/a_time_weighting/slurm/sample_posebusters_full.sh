#!/bin/bash -l
#
# Sample the FULL PoseBusters benchmark set (209 complexes, no whitelist) -
# same methodology as sample_posebusters_10.sh (graph.sample_conformer=false,
# bound-pose fragmentation) but on all held-out complexes instead of a
# 10-complex subset, to remove small-sample noise from the bond-length
# comparison (see STATUS.md: the 10-complex result flipped between the 6h
# and 12h checkpoints, driven by a single outlier complex).
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_posebusters_full.sh
#
#SBATCH --job-name=sigmaflow-a-sample-pb-full
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

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/a_time_weighting

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to your checkpoint path, e.g. experiments/sigmadock/<timestamp>/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_posebusters_full}"

# Force this folder's own src/ ahead of sigmaflow_env's installed
# SigmaFlow_Development copy - otherwise sampling silently uses UNMODIFIED code.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "python:    $PYTHON"
echo "ckpt:      $CKPT_DIR"
echo "data_dir:  $DATA_DIR"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    output_dir="${OUTPUT_DIR}" \
    run_tag=posebusters_full_check \
    num_seeds=1 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out"

echo "Results (predictions.pt + .sdf files) under: ${OUTPUT_DIR}/results/posebusters/"
