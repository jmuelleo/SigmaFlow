#!/bin/bash -l
#
# HABEN DIE KAPUTTEN KOMPLEXE HEBEL AUF DEN VERLUST?
#
# Misst den echten Trainingsverlust je Komplex mit einem vorhandenen
# Checkpoint, fuer die geometrisch fehlerhaften Komplexe und eine
# Zufallskontrolle. Entscheidet, ob ein kuratierter Trainingslauf lohnt --
# eine Stunde statt 72 GPU-Stunden.
#
# Aufruf:
#   CKPT=... LISTE=... CHUNKS=10 sbatch --array=0-9%10 arc/verlust_je_komplex.slurm
#   LIST=1 CKPT=... LISTE=... bash arc/verlust_je_komplex.slurm
#
#SBATCH --job-name=verlust-komplex
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

CKPT="${CKPT:?CKPT auf einen .ckpt-Pfad setzen}"
LISTE="${LISTE:?LISTE auf die Textdatei mit den kaputten Codes setzen}"
CODE_DIR="${CODE_DIR:-${ARC_REPO}/SigmaFlow_Minimal}"
CHUNKS="${CHUNKS:-1}"
ZIEHUNGEN="${ZIEHUNGEN:-4}"
KONTROLLE="${KONTROLLE:-750}"
CONFIG="${CONFIG:-}"
OUT_DIR="${OUT_DIR:-${ARC_RUNS}/verlust_je_komplex}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

[ -f "$CKPT" ]  || { echo "ABBRUCH: CKPT fehlt: ${CKPT}" >&2; exit 1; }
[ -f "$LISTE" ] || { echo "ABBRUCH: LISTE fehlt: ${LISTE}" >&2; exit 1; }
[ -d "${CODE_DIR}/src/sigmadock" ] || {
    echo "ABBRUCH: ${CODE_DIR}/src/sigmadock fehlt -- falscher CODE_DIR?" >&2
    exit 1
}

echo "=============================================================="
echo "  Knoten : $(hostname)"
echo "  Code   : ${CODE_DIR}"
echo "  Ckpt   : ${CKPT}"
echo "  Liste  : ${LISTE}  ($(grep -cve '^\s*$' "$LISTE") Codes)"
echo "  Block ${IDX} von ${CHUNKS}, ${ZIEHUNGEN} Ziehungen, Kontrolle ${KONTROLLE}"
echo "=============================================================="
[ "${LIST:-0}" = "1" ] && exit 0

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"

OUT="${OUT_DIR}/verlust_${IDX}.csv"
ARGS=(--ckpt "$CKPT" --liste "$LISTE" --data_dir "$ARC_DATA"
      --kontrolle "$KONTROLLE" --ziehungen "$ZIEHUNGEN"
      --chunks "$CHUNKS" --chunk "$IDX" --out "$OUT")
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")

"$PYTHON" "${REPO_ROOT}/arc/verlust_je_komplex.py" "${ARGS[@]}"
RC=$?
[ "$RC" -ne 0 ] && { echo "[verlust] FEHLGESCHLAGEN (rc=${RC})" >&2; exit "$RC"; }

N=$(($(wc -l < "$OUT" 2>/dev/null || echo 1) - 1))
echo "[verlust] fertig: ${N} Zeilen in ${OUT}"
[ "$N" -gt 0 ] || { echo "[verlust] FEHLER: leere Ausgabe." >&2; exit 1; }
