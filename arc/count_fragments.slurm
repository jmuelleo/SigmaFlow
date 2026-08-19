#!/bin/bash -l
#
# FRAGMENTZAHL JE LIGAND -- CPU-only.
#
# Erzeugt die vollstaendige Liste (ein Ligand je Zeile) und die Verteilung
# ueber den Datensatz. Braucht KEINE GPU und umgeht damit die Warteschlange.
#
# Usage:
#   sbatch arc/count_fragments.slurm                     # posebusters, 209
#   EXPERIMENT=astex sbatch arc/count_fragments.slurm    # astex, 85
#   EXPERIMENT=pdbbind-general MAX_COMPLEXES=2000 TIMEOUT=60 \
#       sbatch arc/count_fragments.slurm                 # Stichprobe aus 19k
#
# LAUFZEIT
#   Die Cut-Set-Enumeration ist kombinatorisch. Auf PoseBusters brauchen die
#   meisten Liganden Sekundenbruchteile, einzelne mit >20 Torsionen laufen in
#   das Zeitlimit. Zeitueberschreitungen werden protokolliert, nicht geraten.
#   Fuer pdbbind-general (19k Liganden) unbedingt MAX_COMPLEXES setzen.
#
#SBATCH --job-name=fragcount
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

EXPERIMENT="${EXPERIMENT:-posebusters}"
TIMEOUT="${TIMEOUT:-120}"
MAX_COMPLEXES="${MAX_COMPLEXES:-0}"

# Gezaehlt wird in der SigmaFlow-Minimal-Codebasis. Die Fragmentierung ist dort
# unveraendert aus SigmaDock geerbt, die Zahlen gelten also fuer beide Arme.
CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
PYTHON="${ARC_SF_ENV}/bin/python"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR" || exit 1
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

write_run_metadata "FRAGCOUNT-${EXPERIMENT}" "$CODE_DIR" "$PYTHON"

echo "[cfg] Experiment    = ${EXPERIMENT}"
echo "[cfg] Zeitlimit/Mol = ${TIMEOUT} s"
echo "[cfg] max_complexes = ${MAX_COMPLEXES} (0 = alle)"
echo

"$PYTHON" "${ARC_REPO}/arc/count_fragments.py" \
    --data-dir "${ARC_DATA}" \
    --experiment "${EXPERIMENT}" \
    --timeout "${TIMEOUT}" \
    --max-complexes "${MAX_COMPLEXES}" \
    --out "${RUN_DIR}/fragment_counts_${EXPERIMENT}.csv" \
    --json-out "${RUN_DIR}/fragment_counts_${EXPERIMENT}.json" \
  2>&1 | tee "${RUN_DIR}/count_fragments.log"

RC="${PIPESTATUS[0]}"
echo
echo "=============================================================="
echo "rc=${RC}"
echo "  Liste     : ${RUN_DIR}/fragment_counts_${EXPERIMENT}.csv"
echo "  Verteilung: ${RUN_DIR}/fragment_counts_${EXPERIMENT}.json"
echo "  Log       : ${RUN_DIR}/count_fragments.log"
echo "=============================================================="
exit "$RC"
