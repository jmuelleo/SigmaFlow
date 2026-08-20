#!/bin/bash -l
#
# SigmaFlow (frame fix): how many ODE integration steps does it actually need?
#
# THE QUESTION
# We have always sampled with ode.num_steps=25, inherited from SigmaDock's
# default - and SigmaDock's 25 is a value the PAPER tuned for a DIFFUSION
# reverse process ("diminishing returns with more than 20-30 steps",
# Prat et al. 2026, App. F). Nothing says a flow-matching ODE has the same
# optimum. A deterministic ODE with a straight conditional path can often do
# with far fewer steps; conversely, if our vector field is rough, it may need
# many more. Both outcomes are informative:
#   - flat curve from ~10 steps      -> 25 is fine, sampling is not the
#                                       bottleneck, and we could sample 2.5x
#                                       cheaper
#   - still improving at 100-200     -> we have been UNDER-integrating all
#                                       along and every past comparison
#                                       understated SigmaFlow
#
# DESIGN: 7 step counts x 3 seeds = 21 tasks. Three seeds per point, because
# a single draw cannot separate a real trend from noise - the very lesson
# that motivated sample_pb_seeds10.sh. Steps span 5x fewer to 8x more than
# the default.
#
# OUTPUT LAYOUT: one directory PER STEP COUNT. This matters - sample.py
# separates runs by seed (results/.../seed_<seed>/), NOT by step count, so
# two step counts sharing a directory and a seed would collide.
#
# COST: runtime is roughly linear in steps. At 25 steps one full 209-complex
# pass took ~6 min, i.e. ~14 s per step. The 200-step tasks are therefore
# ~48 min each, hence --time=02:00:00.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_pb_stepsweep.sh
#
#SBATCH --job-name=sigmaflow-pb-stepsweep
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-20%5
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#
# IMPORTANT: slurm_logs/ must exist before sbatch (mkdir -p slurm_logs).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint, e.g. experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
BASE_OUT="${BASE_OUT:-$(pwd)/sampling_output_pb_stepsweep}"

# 2D grid flattened into the 1D array index: 7 step counts x 3 seeds = 21.
STEP_LIST=(5 10 15 25 50 100 200)
N_STEPS_OPTS=${#STEP_LIST[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_STEPS=${STEP_LIST[$(( IDX % N_STEPS_OPTS ))]}
SEED=$(( IDX / N_STEPS_OPTS ))          # 0, 1, 2
OUTPUT_DIR="${BASE_OUT}/steps_${NUM_STEPS}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "array idx: $IDX  ->  num_steps=$NUM_STEPS  seed=$SEED"
echo "output:    $OUTPUT_DIR"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"
$PYTHON - <<'PYCHECK'
import inspect
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in inspect.getsource(
    SigmaFlowGenerator._compute_vector_field), "FRAME FIX MISSING - aborting."
print("frame fix verified in loaded module")
PYCHECK

$PYTHON scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    output_dir="${OUTPUT_DIR}" \
    run_tag=pb_stepsweep \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Results under: ${OUTPUT_DIR}/results/posebusters/<model>/seed_${SEED}/"
echo "Ran: SigmaFlow frame fix, ${NUM_STEPS} ODE steps, seed ${SEED}, full PoseBusters"
