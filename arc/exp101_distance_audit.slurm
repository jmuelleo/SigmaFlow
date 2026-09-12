#!/bin/bash -l
#
# EXP-101 — Source Distance Audit. Reine Messung, kein Training, keine GPU noetig.
#
# Beantwortet das Abbruchkriterium des Conditional-Source-Strangs:
# senkt eine einfache, inferenzverfuegbare Heuristik die Rotationsdistanz
# spuerbar unter die Haar-Referenz von 132.3 Grad?
#
# Laeuft in der EXP-100-Codebasis - in SigmaFlow-Minimal ist R_1 = I und jede
# Distanzmessung waere tautologisch. Das Skript bricht dort selbst ab.
#
# Usage: sbatch arc/exp101_distance_audit.slurm
#
#SBATCH --job-name=EXP101-distance
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam"
PYTHON="${ARC_SF_ENV}/bin/python"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

write_run_metadata "EXP101-DISTANCE" "$CODE_DIR" "$PYTHON"

$PYTHON "${ARC_REPO}/arc/exp101_distance_audit.py" \
    --data_dir "${ARC_DATA}" \
    --experiment posebusters \
    --max_complexes 250 \
    --seed 0 \
  2>&1 | tee "${RUN_DIR}/exp101_distance_audit.log"

echo
echo "Ergebnis: ${RUN_DIR}/exp101_distance_audit.log"
echo "Das Abbruchkriterium steht am Ende der Ausgabe."
