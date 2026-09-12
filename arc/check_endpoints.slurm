#!/bin/bash -l
#
# ENDPUNKTE VON CHECKPOINTS AM ARTEFAKT PRUEFEN.
#
# WOZU
#   `meta.txt` liest die Epoche aus dem laufenden stdout zum Kopierzeitpunkt
#   und hinkt dem gespeicherten Checkpoint um 3 bis 15 Epochen hinterher.
#   Nach dem Einfrieren bei 8691436 und dem Snapshot-Kreuzfehler bei 8694389
#   ist die Pruefung am Artefakt Pflicht, nicht Kuer.
#
# WARUM ALS JOB UND NICHT AUF DEM LOGIN-KNOTEN
#   Dort gilt `ulimit -v` = 2 GB. torch kann seine Bibliotheken nicht
#   abbilden: "libtorch_cpu.so: failed to map segment from shared object".
#
# Aufruf:
#   CKPTS="/pfad/a.ckpt /pfad/b.ckpt" sbatch arc/check_endpoints.slurm
#   CODE_DIR=... CKPTS="..." sbatch arc/check_endpoints.slurm
#
#SBATCH --job-name=ckpt-endpoints
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

CKPTS="${CKPTS:?CKPTS setzen: Leerzeichen-getrennte Liste von .ckpt-Pfaden}"
CODE_DIR="${CODE_DIR:-${ARC_REPO}/SigmaFlow_Minimal}"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"

# Der Checkpoint traegt Hyperparameter als Objekte des jeweiligen Arms,
# deshalb muss der passende src-Baum auf dem Pfad liegen (weights_only=False).
export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}:${PYTHONPATH:-}"
[ -d "${CODE_DIR}/src/sigmadock" ] || {
    echo "ABBRUCH: ${CODE_DIR}/src/sigmadock fehlt -- falscher CODE_DIR?" >&2
    exit 1
}

echo "Code: ${CODE_DIR}"
echo
FEHLT=0
for C in $CKPTS; do
    echo "=== ${C}"
    if [ ! -f "$C" ]; then
        echo "     FEHLT"
        FEHLT=$((FEHLT + 1))
        continue
    fi
    echo "     $(stat -c '%y  %s Bytes' "$C")"
    "$PYTHON" "${REPO_ROOT}/arc/_ckpt_epoch.py" "$C"
done

echo
if [ "$FEHLT" -gt 0 ]; then
    echo "${FEHLT} Pfad(e) fehlten -- nachsehen." >&2
    exit 1
fi
echo "alle gelesen."
