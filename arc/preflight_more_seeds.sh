#!/bin/bash
#
# PREFLIGHT FUER ZUSAETZLICHE SAMPLING-SEEDS
#
# ZWECK
#   Weitere Seeds sind nur dann ADDITIV zu den vorhandenen, wenn sie aus
#   demselben Checkpoint und derselben Konfiguration stammen. Sonst mischt
#   man zwei Verteilungen und haelt das Ergebnis fuer Seed-Varianz.
#
#   sample_pb_seeds_cpu.slurm schreibt die Herkunft je Seed nach
#   sampling_metadata_seed<N>.txt, vergleicht sie aber NICHT mit dem, was
#   die schon vorhandenen Seeds benutzt haben. Genau das macht dieses
#   Skript -- VOR dem Abschicken, nicht danach.
#
#   Der Ausgabepfad traegt COMPARISON, RANKING und NFE bereits im Namen;
#   eine Abweichung dort landet automatisch in einem anderen Verzeichnis.
#   Ungeprueft ist allein der Checkpoint.
#
# AUFRUF
#   CKPT=<pfad> bash arc/preflight_more_seeds.sh <model> [von] [bis]
#
#   z.B.  CKPT=/data/.../last.ckpt bash arc/preflight_more_seeds.sh exp110 40 79
#
# AUSGABE
#   Bei Erfolg die fertige sbatch-Zeile. Bei Abweichung ein Abbruch mit
#   Begruendung und KEINE sbatch-Zeile.
set -euo pipefail

MODEL="${1:?Usage: CKPT=<pfad> bash arc/preflight_more_seeds.sh <model> [von] [bis]}"
VON="${2:-40}"
BIS="${3:-79}"
CKPT="${CKPT:?CKPT setzen -- derselbe Checkpoint wie bei den vorhandenen Seeds}"

NUM_STEPS="${NUM_STEPS:-25}"
COMPARISON="${COMPARISON:-controlled}"
RANKING="${RANKING:-raw}"

ARC_RUNS="/data/stat-cadd/shug8458/arc_runs"
OUT_ROOT="${ARC_RUNS}/sampling/${MODEL}__${COMPARISON}__${RANKING}__nfe${NUM_STEPS}__cpu"

echo "=============================================================="
echo " Preflight: ${MODEL}, Seeds ${VON}-${BIS}"
echo " Ziel: ${OUT_ROOT}"
echo "=============================================================="

# --- 1) Checkpoint existiert ueberhaupt ------------------------------------
[ -f "$CKPT" ] || { echo "ABBRUCH: Checkpoint nicht gefunden: $CKPT"; exit 1; }
echo "[ok] Checkpoint vorhanden: $(du -h "$CKPT" | cut -f1)  $CKPT"

# --- 2) Was liegt schon da? ------------------------------------------------
[ -d "$OUT_ROOT" ] || { echo "ABBRUCH: ${OUT_ROOT} existiert nicht."; \
                        echo "         Es gibt keinen Lauf, zu dem diese Seeds additiv waeren."; exit 1; }

VORHANDEN=$(find "$OUT_ROOT" -type d -name 'seed_*' | wc -l)
SDF=$(find "$OUT_ROOT" -name '*.sdf' | wc -l)
echo "[ok] vorhanden: ${VORHANDEN} Seeds, ${SDF} SDF"
[ "$VORHANDEN" -gt 0 ] || { echo "ABBRUCH: keine vorhandenen Seeds."; exit 1; }

# --- 3) Kollision: schreiben wir auf etwas Bestehendes? --------------------
# find statt Glob in [ -d ]: der Glob kann auf mehrere Pfade expandieren,
# und "[ -d a b c ]" ist ein Syntaxfehler. Ausserdem kein "&&" am
# Zeilenende -- unter set -e beendet ein fehlschlagender Test die Schleife.
KOLLISION=""
for s in $(seq "$VON" "$BIS"); do
    if find "${OUT_ROOT}" -type d -name "seed_${s}" -print -quit 2>/dev/null | grep -q .; then
        KOLLISION="${KOLLISION} ${s}"
    fi
done
if [ -n "$KOLLISION" ]; then
    echo "ABBRUCH: diese Seeds existieren bereits:${KOLLISION}"
    echo "         Ein erneuter Lauf wuerde sie ueberschreiben."
    exit 1
fi
echo "[ok] keine Kollision mit vorhandenen Seeds"

# --- 4) DER eigentliche Test: derselbe Checkpoint? -------------------------
# Referenz ist die Metadatendatei mit der KLEINSTEN vorhandenen Seednummer.
REF=$(ls "${OUT_ROOT}"/sampling_metadata_seed*.txt 2>/dev/null | head -1 || true)
if [ -z "$REF" ]; then
    echo "ABBRUCH: keine sampling_metadata_seed*.txt in ${OUT_ROOT}."
    echo "         Ohne dokumentierte Herkunft der vorhandenen Seeds laesst"
    echo "         sich nicht belegen, dass neue Seeds additiv waeren."
    exit 1
fi
echo "[..] Referenz: $(basename "$REF")"

pruefe() {   # pruefe <feldname> <erwarteter wert>
    local feld="$1" soll="$2"
    local ist
    ist=$(awk -F'= *' -v f="$feld" '$1 ~ "^"f" *$" {print $2}' "$REF" | head -1)
    if [ -z "$ist" ]; then
        echo "ABBRUCH: Feld '${feld}' fehlt in ${REF}"; exit 1
    fi
    if [ "$ist" != "$soll" ]; then
        echo "ABBRUCH: ${feld} weicht ab."
        echo "         vorhandene Seeds: ${ist}"
        echo "         geplanter Lauf  : ${soll}"
        echo "         Diese Seeds waeren NICHT additiv."
        exit 1
    fi
    echo "[ok] ${feld} identisch: ${ist}"
}

pruefe checkpoint "$CKPT"
pruefe model      "$MODEL"
pruefe comparison "$COMPARISON"
pruefe ranking    "$RANKING"
pruefe nfe        "$NUM_STEPS"

# --- 5) Alle vorhandenen Seeds auf denselben Checkpoint pruefen ------------
# Nicht nur die Referenz: ein frueherer Teillauf koennte einen anderen
# Checkpoint benutzt haben, ohne dass es je aufgefallen waere.
ABW=$(grep -h '^checkpoint' "${OUT_ROOT}"/sampling_metadata_seed*.txt \
      | sed 's/^checkpoint *= *//' | sort -u | wc -l)
if [ "$ABW" -ne 1 ]; then
    echo "ABBRUCH: die vorhandenen Seeds stammen aus ${ABW} verschiedenen Checkpoints:"
    grep -h '^checkpoint' "${OUT_ROOT}"/sampling_metadata_seed*.txt \
      | sed 's/^checkpoint *= *//' | sort | uniq -c
    exit 1
fi
echo "[ok] alle ${VORHANDEN} vorhandenen Seeds aus genau einem Checkpoint"

echo
echo "=============================================================="
echo " ALLE PRUEFUNGEN BESTANDEN -- diese Seeds waeren additiv."
echo "=============================================================="
echo
echo "cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
echo "CKPT=\"${CKPT}\" MODEL=${MODEL} NUM_STEPS=${NUM_STEPS} \\"
echo "  COMPARISON=${COMPARISON} RANKING=${RANKING} \\"
echo "  sbatch --array=${VON}-${BIS} arc/sample_pb_seeds_cpu.slurm"
