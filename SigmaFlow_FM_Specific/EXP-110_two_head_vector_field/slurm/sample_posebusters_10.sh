#!/bin/bash -l
#
# Sample 10 held-out PoseBusters complexes with a checkpoint trained on the
# real pdbbind-general dataset (see train_pdbbind_general_6h.sh) - a genuine
# generalization test, unlike sample_dummy.sh which checks memorization of
# the 10 training complexes. graph.sample_conformer is left at its base.yaml
# default (true, sample a fresh conformer at inference) since this is meant
# to represent realistic docking, not a memorization check.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_posebusters_10.sh
#
# WHITELIST points at a text file with one PDB ID per line (folder name
# under posebusters_paper/posebusters_benchmark_set/, e.g. "5S8I_2LY") -
# scripts/sample.py's data.blacklist acts as a WHITELIST when
# experiment=posebusters (see datafronts.py::prune_pairs_with_ids).
#
#SBATCH --job-name=sigmaflow-sample-posebusters10
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to your checkpoint path, e.g. experiments/sigmadock/<timestamp>/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
WHITELIST="${WHITELIST:-/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/slurm/posebusters_10_whitelist.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_posebusters10_6h}"

echo "python:    $PYTHON"
echo "ckpt:      $CKPT_DIR"
echo "data_dir:  $DATA_DIR"
echo "whitelist: $WHITELIST"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    data.blacklist="${WHITELIST}" \
    output_dir="${OUTPUT_DIR}" \
    run_tag=posebusters_6h_check \
    num_seeds=1 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out"

echo "Results (predictions.pt + .sdf files) under: ${OUTPUT_DIR}/results/posebusters/"
