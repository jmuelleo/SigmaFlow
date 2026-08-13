#!/bin/bash -l
#
# SigmaFlow (frame fix): 10 INDEPENDENT SAMPLING SEEDS on the full 209-complex
# PoseBusters set, at the paper's default 25 integration steps.
#
# WHY THIS RUN EXISTS
# Every comparison so far rests on ONE draw per method. We therefore have no
# idea how much of the measured difference is method and how much is noise:
# both methods start from a RANDOMLY drawn initial pose, so a single sample
# is a single draw from an unmeasured distribution. Concretely, we cannot
# currently tell whether
#   - the aligned-RMSD gap (2.03 vs 1.93 A) is real or seed noise,
#   - SigmaDock's 42 complexes beyond 20 A are systematic or bad luck.
# 10 seeds per method gives a per-complex spread and makes those questions
# answerable. Counterpart: sample_pb_seeds10_sigmadock.sh in the SigmaDock
# clone - run BOTH, they are only meaningful together.
#
# WHY A JOB ARRAY AND NOT num_seeds=10
# conf/sampling/base.yaml says it plainly: "Prefer num_seeds=1 and multiple
# runs with different `seed` for independent draws and full reproducibility."
# Each array task is one seed, fully reproducible on its own.
#
# OUTPUT LAYOUT - no collisions, one shared OUTPUT_DIR is correct here:
# sample.py writes to results/<exp>/<model>/seed_<cfg.seed>/, i.e. the global
# seed already separates the runs. The .sdf files inside are always named
# ..._seed0.sdf (that index counts num_seeds WITHIN a run, which is 1 here) -
# do not confuse the two.
#
# NOT the paper protocol: the published 79.9% is Top-1 out of N_seed samples
# RANKED BY A TRAINED CONFIDENCE MODEL (paper App. F.2, N_seed = 10 or 40).
# We keep postprocessing.scoring=null, so this measures the raw per-seed
# distribution, not a ranked Top-1. Both methods are treated identically.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_pb_seeds10.sh
#
#SBATCH --job-name=sigmaflow-pb-seeds10
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%5
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#
# --array=0-9%5: ten tasks, at most five at a time, so the GPU queue stays
# usable for others. One task ~6 min at 25 steps.
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_pb_seeds10}"
NUM_STEPS="${NUM_STEPS:-25}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "seed:      $SEED"
echo "num_steps: $NUM_STEPS"
echo "ckpt:      $CKPT_DIR"
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
    run_tag=pb_seeds10 \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Results under: ${OUTPUT_DIR}/results/posebusters/<model>/seed_${SEED}/"
echo "Ran: SigmaFlow frame fix, seed ${SEED}, ${NUM_STEPS} ODE steps, full PoseBusters"
