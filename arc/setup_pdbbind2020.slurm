#!/bin/bash -l
#
# ENTPACKT DEN PDBBIND-v2020-BESTAND ALS BATCH-JOB.
#
# WARUM NICHT AUF DEM LOGIN-KNOTEN
#   Der erste Versuch lief interaktiv auf htc-login02 und war nach 50 Minuten
#   noch beim ERSTEN der drei Archive; danach war der `tar`-Prozess aus der
#   Prozessliste verschwunden. Login-Knoten sind von allen Nutzern geteilt und
#   in CPU und IO gedrosselt, und lang laufende Prozesse werden dort beendet.
#   78.000 kleine Dateien auf ein zu 98 % volles paralleles Dateisystem zu
#   schreiben gehoert auf einen Rechenknoten.
#
# WAS ER TUT
#   Nichts eigenes -- er ruft `arc/setup_pdbbind2020.sh` auf. Die gesamte
#   Logik samt Pruefsummen, Anzahlpruefungen und Dateipruefung steht dort und
#   soll nicht zweimal existieren.
#
# WIEDERANLAUF
#   Das Skript bricht ab, wenn das Ziel nicht leer ist. Nach einem Fehlschlag
#   also mit FORCE=1 einreichen, damit der halb entpackte Baum ersetzt wird:
#       FORCE=1 sbatch arc/setup_pdbbind2020.slurm
#
# Aufruf:
#   sbatch arc/setup_pdbbind2020.slurm
#   FORCE=1 sbatch arc/setup_pdbbind2020.slurm
#   tail -f /data/stat-cadd/shug8458/arc_runs/slurm_logs/<jobid>.out
#
#SBATCH --job-name=pdbbind2020-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO="${REPO:-/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow}"

echo "=== $(date) ==="
echo "Knoten     : $(hostname)"
echo "Job        : ${SLURM_JOB_ID:-<keiner>}"
echo "Repo       : ${REPO}"
echo "FORCE      : ${FORCE:-0}"
echo

cd "$REPO" || { echo "FEHLER: Repo fehlt: ${REPO}" >&2; exit 2; }
[ -f arc/setup_pdbbind2020.sh ] || { echo "FEHLER: arc/setup_pdbbind2020.sh fehlt." >&2; exit 2; }

# FORCE wird aus der Umgebung durchgereicht; `sbatch` exportiert sie
# standardmaessig (--export=ALL).
# `|| RC=$?` ist noetig: mit `set -e` wuerde der Job bei einem Fehlschlag
# sofort enden und die Abschlusszeile nie schreiben. So steht der
# Rueckgabewert im Protokoll, und der Job endet trotzdem als FAILED.
RC=0
bash arc/setup_pdbbind2020.sh || RC=$?

echo
echo "=== $(date) -- Rueckgabewert ${RC} ==="
if [ "$RC" -ne 0 ]; then
    echo "FEHLGESCHLAGEN. Nach einem Abbruch bleibt ein halb entpackter Baum"
    echo "liegen; zum Neuversuch: FORCE=1 sbatch arc/setup_pdbbind2020.slurm"
fi
exit "$RC"
