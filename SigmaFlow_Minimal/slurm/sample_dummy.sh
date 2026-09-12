#!/bin/bash -l
#
# Sample the 10 dummy complexes with a trained checkpoint (see
# train_dummy_overfit_gpu_3h.sh) and write the final docked pose of each to
# an .sdf file (scripts/sample.py -> SamplingModule.export_predictions_to_sdf)
# for a visual sanity check in PyMOL. Ported from SigmaDock/slurm/sample.sh
# (generic template, no diffusion-specific content) with this project's
# SLURM style (see train_dummy_overfit_gpu_3h.sh).
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_dummy.sh
# Find your checkpoint with: find experiments -name "*.ckpt"
#
# postprocessing.scoring/bust_config are disabled (null) - Vina/PoseBusters
# scoring isn't needed just to look at the pose, and skips extra failure
# points (gnina/posebusters setup) not required for this check.
#
# graph.sample_conformer=false: reuse each complex's bound-pose fragmentation
# instead of sampling a fresh conformer, matching how the training data was
# prepared (sample_conformer=False, see notebooks/dummy_train.yaml usage) -
# the most direct test of whether these exact 10 complexes were memorized.
#
#SBATCH --job-name=sigmaflow-sample-dummy
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output}"

echo "python:   $PYTHON"
echo "ckpt:     $CKPT_DIR"
echo "data_dir: $DATA_DIR"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=dummy_train \
    output_dir="${OUTPUT_DIR}" \
    run_tag=dummy_overfit_check \
    num_seeds=1 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out"

echo "Results (predictions.pt + .sdf files) under: ${OUTPUT_DIR}/results/dummy_train/"
