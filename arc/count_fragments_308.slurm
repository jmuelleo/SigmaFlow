#!/bin/bash -l
#
# FRAGMENTZAHL-VERTEILUNG UEBER EINE EXPLIZITE ID-LISTE -- CPU-only.
#
# Standardfall: die offiziellen 308 Komplexe der PoseBusters-Zeitschriften-
# fassung, gegen die volle 428er-Kopie unter posebusters_full/.
#
# Warum das ohne GPU geht: die Fragmentzahl folgt allein aus der Konnektivitaet
# des Liganden-SDF. Sie braucht weder Pose noch Checkpoint noch Sampling. Die
# 44 GPU-h je Modell, die in STATUS.md fuer die 308 stehen, betreffen die
# ERGEBNISSE, nicht diese Verteilung.
#
# Usage:
#   sbatch arc/count_fragments_308.slurm
#   ROOT=/pfad IDS=/pfad/ids.csv LABEL=v2-308 sbatch arc/count_fragments_308.slurm
#
# Kein `set -e`: `module load` und `conda activate` liefern hier regelmaessig
# einen Rueckgabewert != 0, ohne dass etwas kaputt ist. Statt dessen wird der
# Rueckgabewert des Zaehlaufrufs ueber PIPESTATUS abgegriffen und am Ende
# explizit weitergereicht -- damit kann ein Absturz nicht als COMPLETED
# durchgehen, was der eigentliche Zweck von `set -e` hier waere.
#
#SBATCH --job-name=frag308
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

# Eine Ebene TIEFER als posebusters_full/: dort liegen benchmark_set UND
# astex_diverse_set nebeneinander (428 + 85 = 513 Komplexordner). Ohne
# diese Praezisierung wuerde Astex mitgezaehlt.
ROOT="${ROOT:-${ARC_DATA}/posebusters_full/posebusters_benchmark_set}"
IDS="${IDS:-${ARC_REPO}/SigmaFlow_Evaluation/reference/posebusters_v2_308_ids.csv}"
LABEL="${LABEL:-posebusters-v2-308}"
TIMEOUT="${TIMEOUT:-120}"

# Gezaehlt wird in der SigmaFlow-Minimal-Codebasis. Die Fragmentierung ist dort
# unveraendert aus SigmaDock geerbt, die Zahlen gelten also fuer beide Arme.
CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
PYTHON="${ARC_SF_ENV}/bin/python"

# --- Vorpruefung: lieber hier scheitern als nach zwei Stunden ---------------
FEHLER=0
[ -d "$ROOT" ] || { echo "FEHLER: ROOT existiert nicht: ${ROOT}" >&2; FEHLER=1; }
[ -f "$IDS" ]  || { echo "FEHLER: IDS existiert nicht: ${IDS}" >&2; FEHLER=1; }
if [ -d "$ROOT" ] && [ -z "$(ls -A "$ROOT" 2>/dev/null)" ]; then
    echo "FEHLER: ROOT ist leer: ${ROOT}" >&2
    FEHLER=1
fi
[ -x "$PYTHON" ] || { echo "FEHLER: Python fehlt: ${PYTHON}" >&2; FEHLER=1; }
if [ "$FEHLER" -ne 0 ]; then
    echo "Abbruch vor dem Zaehlen." >&2
    exit 2
fi

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR" || exit 1
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

write_run_metadata "FRAG-${LABEL}" "$CODE_DIR" "$PYTHON"

echo "[cfg] ROOT          = ${ROOT}"
echo "[cfg] IDS           = ${IDS}"
echo "[cfg] LABEL         = ${LABEL}"
echo "[cfg] Zeitlimit/Mol = ${TIMEOUT} s"
echo "[cfg] Komplexordner unter ROOT: $(find "$ROOT" -maxdepth 3 -type d \
        -regextype posix-extended -regex '.*/[0-9A-Za-z]{4}_[0-9A-Za-z]{1,3}$' \
        2>/dev/null | wc -l)"
echo

"$PYTHON" "${ARC_REPO}/arc/count_fragments_set.py" \
    --root "$ROOT" \
    --ids "$IDS" \
    --label "$LABEL" \
    --timeout "$TIMEOUT" \
    --out "${RUN_DIR}/fragment_counts_${LABEL}.csv" \
    --json-out "${RUN_DIR}/fragment_counts_${LABEL}.json" \
  2>&1 | tee "${RUN_DIR}/count_fragments_${LABEL}.log"

RC="${PIPESTATUS[0]}"
echo
echo "=============================================================="
echo "rc=${RC}"
echo "  Liste     : ${RUN_DIR}/fragment_counts_${LABEL}.csv"
echo "  Verteilung: ${RUN_DIR}/fragment_counts_${LABEL}.json"
echo "  Log       : ${RUN_DIR}/count_fragments_${LABEL}.log"
if [ "$RC" -ne 0 ]; then
    echo "  ACHTUNG: rc != 0. Bei rc=2 fehlen IDs oder der Pfad stimmt nicht;"
    echo "           die Meldung oben nennt die fehlenden Komplexe."
fi
echo "=============================================================="
exit "$RC"
