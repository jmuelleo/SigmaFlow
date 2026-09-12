#!/bin/bash -l
#
# SIND DIE TATSAECHLICHEN TRAININGSZIELE PB-VALIDE?
#
# WOZU
#   `crystal_pb.slurm` misst die KRISTALLPOSEN (PDBBind 83,89 %). Das echte
#   Trainingsziel ist aber zu 80 % eine neu erzeugte, torsionsoptimierte
#   Konformation (`data.py:434`), deren Gueltigkeit mit `pb_check: false` nie
#   geprueft wird. Dieses Skript bustet `mol_info["aligned"]` -- das Molekuel,
#   aus dem die Zielkoordinaten entstehen.
#
# ZWEI ARME, GEPAART
#   PB_CHECK=false  -> was unsere Laeufe gesehen haben
#   PB_CHECK=true   -> was das publizierte Rezept erzeugt
#   Beide seeden je Index gleich, sehen also dieselben Kandidaten. Der
#   Unterschied IST die Wirkung von pb_check -- gemessen statt geschaetzt.
#
# Aufruf:
#   PB_CHECK=false CHUNKS=20 sbatch --array=0-19%20 arc/train_targets_pb.slurm
#   PB_CHECK=true  CHUNKS=20 sbatch --array=0-19%20 arc/train_targets_pb.slurm
#   STRIDE=5 PB_CHECK=false CHUNKS=4 sbatch --array=0-3%4 arc/train_targets_pb.slurm
#   LIST=1 PB_CHECK=false bash arc/train_targets_pb.slurm
#
#SBATCH --job-name=train-targets-pb
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

# Kein `set -e`: `module load` und `conda activate` liefern hier regelmaessig
# einen Rueckgabewert != 0, ohne dass etwas kaputt ist. Fehler werden statt
# dessen einzeln geprueft.
set -uo pipefail

# SLURM fuehrt eine KOPIE dieses Skripts in /var/spool/slurmd/jobNNN/ aus.
# Deshalb absoluter Pfad, nie ${BASH_SOURCE[0]}.
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

PB_CHECK="${PB_CHECK:?PB_CHECK setzen: true | false}"
CHUNKS="${CHUNKS:-1}"
STRIDE="${STRIDE:-1}"
EXP="${EXP:-pdbbind-general}"
CODE_DIR="${CODE_DIR:-${ARC_REPO}/SigmaFlow_Minimal}"
CONFIG="${CONFIG:-}"
OUT_DIR="${OUT_DIR:-${ARC_RUNS}/train_targets_pb/${EXP}_pb${PB_CHECK}}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

[ -d "$CODE_DIR" ] || { echo "ABBRUCH: CODE_DIR fehlt: ${CODE_DIR}" >&2; exit 1; }
[ -d "${ARC_DATA}" ] || { echo "ABBRUCH: ${ARC_DATA} fehlt" >&2; exit 1; }

echo "=============================================================="
echo "  Knoten     : $(hostname)"
echo "  Code       : ${CODE_DIR}"
echo "  Experiment : ${EXP}   pb_check=${PB_CHECK}"
echo "  Block ${IDX} von ${CHUNKS}, Schritt ${STRIDE}"
echo "  Ausgabe    : ${OUT_DIR}"
echo "=============================================================="

if [ "${LIST:-0}" = "1" ]; then
    echo "(nur Anzeige, nichts gerechnet)"
    exit 0
fi

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
mkdir -p "$OUT_DIR"

# Repo-Wurzel explizit auf den Pfad, statt sich auf das Arbeitsverzeichnis zu
# verlassen -- dieselbe Lehre wie bei MERGE.
# Die Arbeitskopie legt das Paket unter ${CODE_DIR}/src ab (pyproject:
# where = ["src", "scripts"]). `src_sigmadock` ist die Struktur des
# READ-ONLY-Referenzrepos und hier falsch -- daran starb 8728557.
export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}:${PYTHONPATH:-}"
[ -d "${CODE_DIR}/src/sigmadock" ] || {
    echo "ABBRUCH: ${CODE_DIR}/src/sigmadock fehlt -- falscher CODE_DIR?" >&2
    exit 1
}

OUT="${OUT_DIR}/ziele_${IDX}.csv"
ARGS=(--exp $EXP --pb_check "$PB_CHECK" --chunks "$CHUNKS" --chunk "$IDX"
      --stride "$STRIDE" --data_dir "$ARC_DATA" --out "$OUT")
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")

"$PYTHON" "${REPO_ROOT}/arc/check_train_targets.py" "${ARGS[@]}"
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "[ziele] FEHLGESCHLAGEN (rc=${RC})" >&2
    exit "$RC"
fi

# SLURM COMPLETED ist kein Beweis, dass etwas entstanden ist.
N=$(($(wc -l < "$OUT" 2>/dev/null || echo 1) - 1))
echo "[ziele] fertig: ${N} Zeilen in ${OUT}"
[ "$N" -gt 0 ] || { echo "[ziele] FEHLER: leere Ausgabe." >&2; exit 1; }
