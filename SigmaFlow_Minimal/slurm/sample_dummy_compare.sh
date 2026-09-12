#!/bin/bash -l
#
# SigmaFlow auf den 10 Dummy-Komplexen - fuer die visuelle PyMOL-Kontrolle.
#
# ⚠️ WICHTIGER VORBEHALT, BITTE NICHT ALS BENCHMARK VERWENDEN ⚠️
# Die 10 Dummy-Komplexe (1G9V_RQ3, 1HWI_115, 1MZC_BNE, 1OWE_675, 1R1H_BIR,
# 1S3V_TQD, 1U1C_BAU, 1V4S_MRK, 1YQY_915, 2BSM_BSM) sind klassische
# PDBBind-Eintraege. Der hier verwendete Checkpoint wurde auf pdbbind-general
# trainiert, das diese Komplexe mit hoher Wahrscheinlichkeit ENTHAELT.
# Ausserdem wurden genau diese 10 fuer die frueheren Overfit-Laeufe benutzt.
#
# => Es ist KEIN Testset. Zahlen daraus sind fuer Leistungsaussagen wertlos.
#    Der Zweck ist ausschliesslich: sich die Posen ansehen und den
#    dokumentierten Kopf-Schwanz-Flip mit eigenen Augen erkennen.
#    Fuer belastbare Metriken bleibt das PoseBusters-Set die Grundlage
#    (siehe VERGLEICH_SigmaFlow_vs_SigmaDock.txt).
#
# 3 Seeds, weil eine Einzelziehung auch hier ueberwiegend Rauschen zeigt.
#
# Usage:
#   CKPT_DIR=experiments/sigmadock/0-08-11_18-00-41/checkpoints/last.ckpt \
#     sbatch slurm/sample_dummy_compare.sh
#
#SBATCH --job-name=sigmaflow-dummy-cmp
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the Frame-Fix checkpoint}"
# Verzeichnis, das dummy_data/ enthaelt (experiment=dummy_train sucht darin)
DATA_DIR="${DATA_DIR:-$(pwd)/notebooks}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_dummy_compare}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "SigmaFlow, Dummy-Set, Seed ${SEED}"
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
    experiment=dummy_train \
    output_dir="${OUTPUT_DIR}" \
    run_tag=dummy_compare \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps=25 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Results: ${OUTPUT_DIR}/results/dummy_train/<model>/seed_${SEED}/"
echo "ERINNERUNG: Trainingsdaten-Ueberlappung - nur zum Ansehen, nicht als Benchmark."
