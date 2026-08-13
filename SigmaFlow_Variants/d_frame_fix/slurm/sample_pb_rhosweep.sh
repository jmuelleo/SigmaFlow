#!/bin/bash -l
#
# SigmaFlow: NICHT-LINEARE Verteilung der ODE-Schritte.
#
# WARUM DIESER LAUF EXISTIERT
# Die Kosinus-ueber-t-Diagnostik (diagnostics/rotation_completion, Job 8561684)
# hat gemessen, wie gut das Modell zu jedem Zeitpunkt weiss, wohin es den
# Liganden drehen muss:
#     t < 0.1  -> Kosinus 0.025, 47.7% negativ   (Muenzwurf, das Modell ist blind)
#     t ~ 0.75 -> Kosinus 0.730                  (zuverlaessig)
# Das aktuelle Zeitgitter ist LINEAR (conf: discretization=power), also liegen
# 10 von 25 Schritten bei t<0.4 - im blinden Bereich. Diese Schritte drehen den
# Liganden trotzdem, in im Wesentlichen zufaellige Richtung, und die
# deterministische Integration zementiert das Ergebnis. Genau das erzeugt den
# dokumentierten Kopf-Schwanz-Flip (36.5% der Posen ueber 160 Grad Fehler).
#
# Der Sampler unterstuetzt eine andere Verteilung bereits ueber
# discretization=edm mit rho (Karras-Schema):
#     t_i = ( t_min^(1/rho) + i/(N-1) * (t_max^(1/rho) - t_min^(1/rho)) )^rho
# Schritte mit t<0.4 bei N=25:
#     rho=3.0 -> 16/25   (schlechteste Wahl, Default-Wert der Config)
#     linear  -> 10/25   (aktueller Zustand)
#     rho=1.0 -> 10/25   (edm mit rho=1 == linear; dient als Kontrolle,
#                         dass edm und power denselben Pfad erzeugen)
#     rho=0.5 ->  4/25
#     rho=0.333 -> 2/25
#
# DESIGN: 4 Arme x 3 Seeds = 12 Tasks. Drei Seeds, weil die Seed-Streuung eines
# einzelnen Komplexes (SD 1.55 A) ein Vielfaches des erwarteten Effekts ist -
# dieselbe Lehre wie beim Schritt-Sweep.
#
# ERFOLGSKRITERIUM: Orientierungsfehler-Median sinkt und der Anteil unter
# 20 Grad steigt ueber die aktuellen 3.8%, OHNE dass der Schwerpunktabstand
# (aktuell 1.16 A, dort ist SigmaFlow besser als SigmaDock) steigt.
# FALSIFIKATION: keine Bewegung ueber die Seed-Streuung hinaus -> die fruehe
# Blindheit ist nicht ueber das Zeitgitter kompensierbar, und der dedizierte
# Rotationskopf wird zur ersten Wahl.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt \
#     sbatch slurm/sample_pb_rhosweep.sh
#
#SBATCH --job-name=sigmaflow-pb-rhosweep
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%4
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#
# WICHTIG: slurm_logs/ muss vor dem sbatch existieren (mkdir -p slurm_logs).

set -euo pipefail
export PYTHONUNBUFFERED=1

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR, e.g. experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
BASE_OUT="${BASE_OUT:-$(pwd)/sampling_output_pb_rhosweep}"
NUM_STEPS="${NUM_STEPS:-25}"

# 4 Arme x 3 Seeds, in den 1D-Arrayindex gefaltet
ARM_DISC=(power edm      edm      edm)
ARM_RHO=( 1.0   1.0      0.5      0.333)
ARM_NAME=(linear edm_rho1 edm_rho05 edm_rho033)
N_ARMS=4
IDX="${SLURM_ARRAY_TASK_ID:-0}"
A=$(( IDX % N_ARMS ))
SEED=$(( IDX / N_ARMS ))
DISC="${ARM_DISC[$A]}"
RHO="${ARM_RHO[$A]}"
NAME="${ARM_NAME[$A]}"
OUTPUT_DIR="${BASE_OUT}/${NAME}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "array idx : $IDX -> Arm '${NAME}' (discretization=${DISC}, rho=${RHO}), Seed ${SEED}"
echo "output    : ${OUTPUT_DIR}"
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
    run_tag="rho_${NAME}" \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    ode.discretization="${DISC}" \
    ode.rho="${RHO}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Ran: ${NAME} (disc=${DISC}, rho=${RHO}), Seed ${SEED}, ${NUM_STEPS} Schritte, volles PoseBusters"
echo "Results: ${OUTPUT_DIR}/results/posebusters/<model>/seed_${SEED}/"
