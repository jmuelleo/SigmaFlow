#!/bin/bash -l
#
# PACKT EINE DATEILISTE ZU EINEM ARCHIV -- ALS BATCH-JOB.
#
# WARUM NICHT AUF DEM LOGIN-KNOTEN
#   `tar` ueber zehntausende kleine Dateien ist auf einem parallelen
#   Dateisystem metadatengebunden, und Login-Knoten sind zusaetzlich in IO
#   gedrosselt. Gemessen am 2026-09-07: 47.600 Dateien waren nach 30 Minuten
#   noch nicht durch, waehrend 18.500 Dateien in gut einer Minute liefen --
#   der Zusammenhang ist also nicht linear, sondern die Drosselung greift.
#   Dasselbe hat schon das Entpacken der PDBBind-Archive gekostet.
#
# WARUM pigz
#   gzip ist einkernig. Bei 170.000 Dateien ist die Kompression selbst der
#   zweite Flaschenhals. `pigz` verteilt sie; ist es nicht da, faellt das
#   Skript auf gzip zurueck, statt zu scheitern.
#
# WARUM tar -T UND KEIN GLOB
#   Die Liste wird VORHER mit `find` erzeugt und geprueft (`wc -l`). Ein Glob
#   im Job koennte je nach Arbeitsverzeichnis etwas anderes treffen, und der
#   Fehler waere am Archiv nicht zu sehen.
#
# Aufruf:
#   PACK_LISTE=/tmp/astex.txt PACK_ZIEL=/data/stat-cadd/shug8458/poses_astex.tgz \
#     sbatch arc/pack_posen.slurm
#
#SBATCH --job-name=pack-posen
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

LISTE="${PACK_LISTE:?PACK_LISTE auf die Dateiliste setzen}"
ZIEL="${PACK_ZIEL:?PACK_ZIEL auf den Archivpfad setzen}"
WURZEL="${PACK_WURZEL:-/data/stat-cadd/shug8458/arc_runs}"

[ -f "$LISTE" ] || { echo "FEHLER: Liste fehlt: ${LISTE}" >&2; exit 2; }
N="$(wc -l < "$LISTE")"
[ "$N" -gt 0 ] || { echo "FEHLER: Liste ist leer." >&2; exit 2; }

echo "=== $(date) ==="
echo "Knoten  : $(hostname)"
echo "Liste   : ${LISTE}  (${N} Dateien)"
echo "Ziel    : ${ZIEL}"
echo "Wurzel  : ${WURZEL}"

# Stichprobe: existieren die Eintraege ueberhaupt? Eine Liste aus einem
# anderen Arbeitsverzeichnis enthaelt relative Pfade, die hier ins Leere
# zeigen -- tar wuerde dann ein fast leeres Archiv schreiben und mit 0 enden.
FEHLT=0
while read -r f; do
    [ -e "${WURZEL}/${f}" ] || [ -e "$f" ] || FEHLT=$(( FEHLT + 1 ))
done < <(head -50 "$LISTE")
if [ "$FEHLT" -gt 0 ]; then
    echo "FEHLER: ${FEHLT} von 50 Stichproben nicht gefunden." >&2
    echo "        Wurzel falsch? Erste Zeile der Liste:" >&2
    head -1 "$LISTE" >&2
    exit 1
fi

if command -v pigz > /dev/null; then
    echo "Kompression: pigz mit ${SLURM_CPUS_PER_TASK:-8} Kernen"
    tar -C "$WURZEL" -cf - -T "$LISTE" \
      | pigz -p "${SLURM_CPUS_PER_TASK:-8}" > "$ZIEL"
else
    echo "Kompression: gzip (pigz nicht gefunden, einkernig)"
    tar -C "$WURZEL" -czf "$ZIEL" -T "$LISTE"
fi

echo
echo "=== $(date) -- fertig ==="
ls -lh "$ZIEL"
echo "Eintraege im Archiv: $(tar -tzf "$ZIEL" | wc -l)   (Liste: ${N})"
