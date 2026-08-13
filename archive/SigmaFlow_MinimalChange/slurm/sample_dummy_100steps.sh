#!/bin/bash -l
#
# Test 2 (more steps): re-sample the 10 dummy complexes with the SAME
# already-trained, unmodified checkpoint used for sampling_output_prodhparams
# (PAUSE-PUNKT #11 production-hyperparameter run), but with a much finer
# ODE discretization (ode.num_steps=100 instead of the conf/sampling/base.yaml
# default of 25). No retraining, no code change - runs directly against the
# existing SigmaFlow_Development deployment on ARC.
#
# Purpose: mirror of sample_dummy_10steps.sh at the opposite end. After
# Test 1 (oracle vector field, see STATUS.md PAUSE-PUNKT #14) already showed
# that the ODE-integration/reconstruction mechanism itself is exact given a
# perfect vector field, this checks whether simply integrating the REAL
# (imperfect) network-predicted field more finely reduces the discretization
# component of the junction-bond error, or whether the gap to SigmaDock/the
# oracle is essentially independent of step count (i.e. dominated by what
# the network predicts, not by how finely that prediction is integrated).
#
# Usage: sbatch slurm/sample_dummy_100steps.sh
# (CKPT_DIR/DATA_DIR/OUTPUT_DIR can still be overridden via env vars.)
#
#SBATCH --job-name=sigmaflow-sample-100steps
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
# Default: the PAUSE-PUNKT #11 production-hyperparameter checkpoint, the
# same one behind sampling_output_prodhparams - override via env var if you
# want to test a different checkpoint.
CKPT_DIR="${CKPT_DIR:-experiments/sigmadock/0-07-25_21-40-56/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/notebooks}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_100steps}"

echo "python:   $PYTHON"
echo "ckpt:     $CKPT_DIR"
echo "data_dir: $DATA_DIR"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

$PYTHON scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=dummy_train \
    output_dir="${OUTPUT_DIR}" \
    run_tag=dummy_overfit_check_100steps \
    num_seeds=1 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    ode.num_steps=100 \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out"

echo "Results (predictions.pt + .sdf files) under: ${OUTPUT_DIR}/results/dummy_train/"
echo "Ran with: ode.num_steps=100 (baseline was 25, see sampling_output_prodhparams)"
