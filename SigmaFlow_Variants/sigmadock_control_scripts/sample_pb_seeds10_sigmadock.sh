#!/bin/bash -l
#
# SigmaDock: 10 INDEPENDENT SAMPLING SEEDS on the full 209-complex PoseBusters
# set, at the PAPER's number of discretisation steps.
#
# Counterpart to d_frame_fix/slurm/sample_pb_seeds10.sh - run BOTH, they are
# only meaningful together. Rationale for 10 seeds is spelled out there.
#
# THE PAPER'S SETTING (Prat et al. 2026, App. F "Inference Details"):
#   "By default, we use N_steps = 25, t_min = 0.002, t_max = 1, rho = 3."
#   plus: "we find diminishing returns with more than 20-30 steps".
# So 25 steps IS the paper value - and it is ALREADY this repo's default
# (conf/sampling/base.yaml: diffusion.num_steps = 25). Passing it explicitly
# below documents the choice rather than changing anything.
#
# ONE REMAINING DEVIATION, deliberately NOT changed: this config uses
# diffusion.t_min = 0.005, the paper states 0.002. Changing it would make
# these runs incomparable to the 6h/12h runs we already analysed, which all
# used 0.005. Flagged here so it is a known deviation, not a silent one.
#
# ALSO NOT the paper protocol: the published 79.9% is Top-1 out of N_seed
# samples RANKED BY A TRAINED CONFIDENCE MODEL (App. F.2, N_seed = 10 or 40).
# We keep postprocessing.scoring=null. This measures the raw per-seed
# distribution, identically for both methods.
#
# NOTE diffusion.noise_scale is 0.0 by default here ("0 = no noise, better
# fidelity"), i.e. SigmaDock's reverse process is DETERMINISTIC given its
# starting point, exactly like SigmaFlow's ODE. The only randomness on either
# side is the initial draw - which is precisely what these 10 seeds probe.
#
# Usage (from the SigmaDock clone root):
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_pb_seeds10_sigmadock.sh
#
#SBATCH --job-name=sigmadock-pb-seeds10
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%5
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#
# IMPORTANT: slurm_logs/ must exist before sbatch (mkdir -p slurm_logs).

set -euo pipefail

# ENVIRONMENT: SigmaDock runs in myenv, NOT sigmaflow_env, activated with
# `source activate` and the plain `python` on PATH. Under sigmaflow_env,
# `import sigmadock` resolves to the SigmaFlow package and dies with
# ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'.
# That mistake cost job 8512799 5:44 of runtime.
module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the checkpoint, e.g. experiments/sigmadock/0-08-11_17-57-05/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_pb_seeds10}"
NUM_STEPS="${NUM_STEPS:-25}"   # paper value
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "seed:      $SEED"
echo "num_steps: $NUM_STEPS  (paper default)"
echo "ckpt:      $CKPT_DIR"
python -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"
python - <<'PYCHECK'
import sigmadock.diff.denoiser  # noqa: F401 - presence is the assertion
import sigmadock, os
assert "SigmaDock_Reproduction" in os.path.abspath(sigmadock.__file__), \
    f"WRONG sigmadock package loaded: {sigmadock.__file__} - aborting."
print("original SigmaDock (diffusion denoiser) verified")
PYCHECK

python scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    output_dir="${OUTPUT_DIR}" \
    run_tag=pb_seeds10 \
    seed="${SEED}" \
    num_seeds=1 \
    diffusion.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Results under: ${OUTPUT_DIR}/results/posebusters/<model>/seed_${SEED}/"
echo "Ran: SigmaDock, seed ${SEED}, ${NUM_STEPS} diffusion steps (paper value), full PoseBusters"
