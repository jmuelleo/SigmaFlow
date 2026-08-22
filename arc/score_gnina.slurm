#!/bin/bash -l
#
# VINARDO-SCORING FUER DAS RANKING
#
# Erzeugt die Affinitaeten, die SigmaDocks Rankingheuristik braucht. Reine
# CPU-Arbeit: --cnn_scoring none schaltet gninas neuronalen Teil ab, uebrig
# bleibt die klassische Vinardo-Funktion.
#
# ACHTUNG ZUM MODUL: gnina laeuft NICHT auf Login-Knoten, dort scheitert das
# Mappen von libcublasLt.so.12 am Speicherlimit. Auf Rechenknoten laeuft es,
# auch auf reinen CPU-Knoten, wenn CUDA/12.6.0 mitgeladen ist. Getestet auf
# htc-c052 am 2026-08-22.
#
# Usage:
#   MODEL=exp110 sbatch arc/score_gnina.slurm
#   MODEL=sigmaflow_minimal MAX_SEEDS=40 sbatch arc/score_gnina.slurm
#
#SBATCH --job-name=gnina-score
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | exp100 | exp110 | sigmadock}"
COMPARISON="${COMPARISON:-controlled}"
RANKING="${RANKING:-raw}"
NUM_STEPS="${NUM_STEPS:-25}"
SUFFIX="${SUFFIX:-__cpu}"
MAX_SEEDS="${MAX_SEEDS:-}"

SAMPLING_ROOT="${ARC_RUNS}/sampling/${MODEL}__${COMPARISON}__${RANKING}__nfe${NUM_STEPS}${SUFFIX}"
PDB_ROOT="${ARC_DATA}/posebusters_paper/posebusters_benchmark_set"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"

module load CUDA/12.6.0
module load gnina/1.3.2

write_run_metadata "GNINA-SCORE-${MODEL}" "$REPO_ROOT" "$PYTHON"

echo "=============================================================="
echo "  node          : $(hostname)"
echo "  model         : ${MODEL}"
echo "  sampling_root : ${SAMPLING_ROOT}"
echo "  pdb_root      : ${PDB_ROOT}"
echo "  max_seeds     : ${MAX_SEEDS:-alle}"
echo "=============================================================="

# Fail-fast, bevor 209 Aufrufe ins Leere laufen.
[ -d "$SAMPLING_ROOT" ] || { echo "ABBRUCH: $SAMPLING_ROOT existiert nicht"; exit 1; }
[ -d "$PDB_ROOT" ]      || { echo "ABBRUCH: $PDB_ROOT existiert nicht"; exit 1; }
gnina --version | head -1

N=$(find "$SAMPLING_ROOT" -name "*.sdf" | wc -l)
echo "  gefundene Posen: ${N}"
[ "$N" -gt 0 ] || { echo "ABBRUCH: keine Posen"; exit 1; }

ARGS=(--sampling_root "$SAMPLING_ROOT" --pdb_root "$PDB_ROOT"
      --out "${RUN_DIR}/gnina_scores_${MODEL}.csv")
[ -n "$MAX_SEEDS" ] && ARGS+=(--max_seeds "$MAX_SEEDS")

time "$PYTHON" "${REPO_ROOT}/arc/score_gnina.py" "${ARGS[@]}"

echo
echo "Ergebnis: ${RUN_DIR}/gnina_scores_${MODEL}.csv"
