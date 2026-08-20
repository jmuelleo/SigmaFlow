#!/bin/bash -l
#
# SigmaFlow — die zwei abschliessenden Diagnosetests der Minimal-Konversion:
#   TEST 1  Kosinus-ueber-t des gelernten Vektorfeldes (Translation + Rotation)
#   TEST 2  globale SE(3)-Aequivarianz des Vektorfeldes, mit drei Kontrollen
#
# Reine Diagnostik: kein Training, keine Architekturaenderung, keine Schreibzugriffe
# ausserhalb von diagnostics/rotation_completion/out/.
#
# Beide Tests brauchen nur Vorwaertspaesse -> klein dimensioniert.
#
# Usage:
#   CKPT=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/diag_rotation_completion.sh
#
#SBATCH --job-name=sigmaflow-rotdiag
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# WICHTIG: slurm_logs/ muss vor dem sbatch existieren (mkdir -p slurm_logs).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT="${CKPT:?Set CKPT to the Frame-Fix checkpoint, e.g. experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
OUT_DIR="${OUT_DIR:-$(pwd)/diagnostics/rotation_completion/out}"
N_COMPLEXES="${N_COMPLEXES:-200}"
N_T="${N_T:-20}"
N_EQ_COMPLEXES="${N_EQ_COMPLEXES:-30}"
N_ROT="${N_ROT:-5}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

echo "==================================================================="
echo "SigmaFlow Rotations-Abschlussdiagnostik"
echo "ckpt   : ${CKPT}"
echo "data   : ${DATA_DIR}"
echo "out    : ${OUT_DIR}"
echo "==================================================================="

# --- 0) Der Frame-Fix MUSS im geladenen Modul stecken, sonst ist alles wertlos
$PYTHON - <<'PYCHECK'
import inspect
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src, "FRAME FIX MISSING - aborting."
print("frame fix verified in loaded module")
PYCHECK

# --- 1) Checkpoint identifizieren, NICHT dem Pfadnamen vertrauen
echo; echo "--- Checkpoint-Verifikation ---"
$PYTHON diagnostics/rotation_completion/verify_checkpoint.py "${CKPT}" | tee "${OUT_DIR}/checkpoint_identity.txt"

# --- 2) TEST 1: Kosinus ueber t
echo; echo "--- TEST 1: Kosinus ueber t ---"
$PYTHON diagnostics/rotation_completion/cosine_by_t.py \
    ckpt="${CKPT}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    +diag.out_dir="${OUT_DIR}" \
    +diag.max_complexes="${N_COMPLEXES}" \
    +diag.n_t="${N_T}" \
    +diag.batch_size=8 \
    hydra.run.dir="${OUT_DIR}/hydra_cosine"

# --- 3) TEST 2: globale SE(3)-Aequivarianz
echo; echo "--- TEST 2: globale SE(3)-Aequivarianz ---"
$PYTHON diagnostics/rotation_completion/equivariance.py \
    ckpt="${CKPT}" \
    data_dir="${DATA_DIR}" \
    experiment=posebusters \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    +diag.out_dir="${OUT_DIR}" \
    +diag.n_complexes="${N_EQ_COMPLEXES}" \
    +diag.n_rot="${N_ROT}" \
    +diag.batch_size=4 \
    hydra.run.dir="${OUT_DIR}/hydra_equivariance"

# --- 4) Zusammenfassung
echo; echo "--- Zusammenfassung ---"
$PYTHON diagnostics/rotation_completion/summarize.py "${OUT_DIR}" | tee "${OUT_DIR}/FINAL_SUMMARY.txt"

echo
echo "Fertig. Ergebnisse unter: ${OUT_DIR}"
echo "  cosine_by_t_raw.csv        Rohbeobachtungen"
echo "  cosine_by_t_summary.txt    Tabelle je t-Bin"
echo "  equivariance_results.json  maschinenlesbar"
echo "  FINAL_SUMMARY.txt          Gesamturteil"
