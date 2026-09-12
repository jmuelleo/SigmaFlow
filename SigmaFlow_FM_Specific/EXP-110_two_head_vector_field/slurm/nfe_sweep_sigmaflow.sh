#!/bin/bash -l
#
# ZIEL-1-EXPERIMENT: Qualitaet gegen Zahl der Netzauswertungen (NFE),
# SigmaFlow-Arm. Das SigmaDock-Gegenstueck ist nfe_sweep_sigmadock.sh -
# NUR GEMEINSAM AUSSAGEKRAEFTIG.
#
# WARUM NEU, obwohl es schon einen Schritt-Sweep gibt
# Der bestehende Sweep (Job 8554148) benutzte {5,10,15,25,50,100,200} und lief
# mit --array ...%5. Beides macht ihn fuer den METHODENUEBERGREIFENDEN
# Vergleich unbrauchbar:
#   1. SigmaDock wurde nie gesweept -> die Aussage "Flow Matching braucht
#      weniger Schritte" ist bisher NICHT zulaessig. Zulaessig ist nur
#      "SigmaFlow braucht die geerbten 25 nicht".
#   2. Mit %5 liefen bis zu fuenf Tasks gleichzeitig und teilten sich CPU und
#      Dateisystem. Die Wall-Clock war dadurch von Nebenlaeufigkeit dominiert:
#      Seed 0 lag ueber ALLE Schrittzahlen flach bei 12-15 min, waehrend Seed 2
#      sauber skalierte (~1.3 min fix + ~2.4 s je Schritt). Als Kostenmass war
#      das wertlos.
#
# DESHALB HIER: identische Schrittzahlen fuer beide Methoden und %1, also
# strikt seriell. Damit ist die Wall-Clock interpretierbar.
#
# BERICHTE BEIDES: NFE ist implementierungsunabhaengig und die ehrlichere
# Groesse; Wall-Clock ist das, was ein Anwender spuert. Bei euler ist
# NFE = num_steps - 1 (ein Aufruf je Schritt, siehe timesteps[:-1] in
# sampling.py); bei heun waere es das Doppelte.
#
# DESIGN: 7 Schrittzahlen x 3 Seeds = 21 Tasks, seriell.
# LAUFZEIT: grob 1.5-3 h gesamt. Passt in short (12 h Limit).
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt \
#     sbatch slurm/nfe_sweep_sigmaflow.sh
#
#SBATCH --job-name=sigmaflow-nfe
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-20%1
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR, e.g. experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
BASE_OUT="${BASE_OUT:-$(pwd)/sampling_output_nfe_sweep}"

# IDENTISCH mit dem SigmaDock-Skript zu halten - sonst ist der Vergleich hin.
STEP_LIST=(1 2 5 10 25 50 100)
N_STEPS_OPTS=${#STEP_LIST[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_STEPS=${STEP_LIST[$(( IDX % N_STEPS_OPTS ))]}
SEED=$(( IDX / N_STEPS_OPTS ))
OUTPUT_DIR="${BASE_OUT}/steps_${NUM_STEPS}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=== SigmaFlow NFE-Sweep: ${NUM_STEPS} Schritte, Seed ${SEED} ==="
echo "START_EPOCH=$(date +%s)"
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
    run_tag=nfe_sweep \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "END_EPOCH=$(date +%s)"
echo "Ran: SigmaFlow, ${NUM_STEPS} Schritte (NFE=$(( NUM_STEPS - 1 ))), Seed ${SEED}"
