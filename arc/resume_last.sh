#!/bin/bash -l
#
# WIEDEREINSTIEG VOM NEUESTEN SNAPSHOT EINES ABGESTUERZTEN LAUFS
#
# WOFUER
#   Der twohead-Arm stirbt wiederholt an CUDA-OOM, zuletzt Job 8685813 nach
#   10:57 h. Statt jeden Wiedereinstieg von Hand zusammenzusetzen, liest
#   dieses Skript die Einstellungen aus der manifest.json des toten Laufs,
#   sucht den neuesten Snapshot und reicht neu ein.
#
# WARUM NICHT RESUME_COMMAND.txt
#   Die schreibt der Trap nur bei USR1, also bei drohendem Timeout. Ein
#   Absturz durch OOM oder eine Ausnahme loest kein USR1 aus, und die Datei
#   fehlt dann. Das Manifest gibt es dagegen immer.
#
# WAS DAS SKRIPT NICHT ERRAET
#   HORIZON_FILE. Der Horizont entscheidet ueber den Anneal, und ein falscher
#   Wert erzeugt einen stillen Warm Restart. Das Manifest nennt nur eine
#   Beschreibung, keinen Pfad, also muss der Aufrufer die Datei setzen. Ohne
#   sie bricht das Skript ab und zeigt, was im Manifest steht.
#
#   Ebenso wenig erraet es FINAL_TRAIN_CONFIG, FINAL_CPUS, FINAL_NUM_WORKERS
#   und FINAL_MEM. Wer mit Konfigurationsdatei gefahren ist, setzt sie wieder.
#
# Aufruf:
#   HORIZON_FILE=$PWD/arc/final_horizon_120h.env \
#     bash arc/resume_last.sh /data/stat-cadd/shug8458/arc_runs/SF_2H_72H_s0_120H_8685813
#
#   DRY_RUN=1 davor zeigt nur, was passieren wuerde.
#
set -uo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOT="${1:?Usage: resume_last.sh <RUN_DIR des abgestuerzten Laufs>}"

[ -d "$TOT" ] || { echo "FEHLER: kein Verzeichnis: $TOT" >&2; exit 1; }
MAN="${TOT}/manifest.json"
[ -f "$MAN" ] || { echo "FEHLER: keine manifest.json in $TOT" >&2; exit 1; }

lies() {  # ein Feld aus dem Manifest, ohne jq
    python - "$MAN" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
def suche(o, k):
    if isinstance(o, dict):
        if k in o:
            return o[k]
        for v in o.values():
            t = suche(v, k)
            if t is not None:
                return t
    return None
v = suche(d, sys.argv[2])
print("" if v is None else v)
PY
}

# Die Schluessel heissen im Manifest anders als die Kommandozeilenflags von
# write_manifest.py. Nachgesehen in arc/write_manifest.py:234-241, nicht
# geraten -- die erste Fassung dieses Skripts las hier ins Leere und lieferte
# leere Felder, ohne zu meckern.
MODEL="$(lies model)"
SEED="$(lies seed)"
BATCH="$(lies batch_size_per_step)"
ACCUM="$(lies accum_grad_batches)"
STEPS="$(lies max_steps_scheduler_horizon)"
EPOCHS="$(lies max_epochs)"
WALL="$(lies requested_walltime)"
QUELLE="$(lies horizon_source)"

for feld in MODEL SEED BATCH ACCUM STEPS; do
    if [ -z "${!feld}" ]; then
        echo "FEHLER: ${feld} steht nicht im Manifest ${MAN}." >&2
        echo "        Ohne diesen Wert waere der Wiedereinstieg geraten." >&2
        exit 1
    fi
done

# Der neueste Snapshot. `ls -t` sortiert nach Aenderungszeit, der
# Notfall-Snapshot beim Absturz ist damit der erste.
CKPT="$(ls -t "${TOT}/snapshots"/snap_*.ckpt 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "FEHLER: kein snap_*.ckpt in ${TOT}/snapshots" >&2; exit 1; }

# RUN_ID_EXTRA aus dem Verzeichnisnamen zurueckgewinnen, damit der neue Lauf
# wieder gleich heisst. SF_2H_72H_s0_120H_8685813 -> _120H
BASIS="$(basename "$TOT")"
OHNE_JOB="${BASIS%_*}"
EXTRA=""
case "$OHNE_JOB" in
    *_s"${SEED}"_*) EXTRA="_${OHNE_JOB##*_s${SEED}_}" ;;
esac

echo "=============================================================="
echo "  WIEDEREINSTIEG"
echo "  toter Lauf     : $BASIS"
echo "  Modell / Seed  : ${MODEL} / ${SEED}"
echo "  Batch x Accum  : ${BATCH} x ${ACCUM}"
echo "  Horizont       : ${EPOCHS} Epochen, ${STEPS} Schritte"
echo "  Snapshot       : $(basename "$CKPT")"
echo "                   $(date -r "$CKPT" '+%Y-%m-%d %H:%M' 2>/dev/null)"
echo "  RUN_ID_EXTRA   : ${EXTRA:-<leer>}"
echo "  Walltime       : ${FINAL_WALLTIME:-$WALL}"
echo "  Horizontquelle laut Manifest:"
echo "    ${QUELLE:0:120}"
echo "=============================================================="

if [ -z "${HORIZON_FILE:-}" ]; then
    echo "ABBRUCH: HORIZON_FILE ist nicht gesetzt." >&2
    echo "         Der Horizont entscheidet ueber den Anneal; ein falscher" >&2
    echo "         Wert baut die Lernrate still um. Setze ihn ausdruecklich," >&2
    echo "         passend zur Quelle oben." >&2
    exit 1
fi
[ -f "$HORIZON_FILE" ] || { echo "FEHLER: HORIZON_FILE fehlt: $HORIZON_FILE" >&2; exit 1; }

# Was der Lauf braucht, aus dem Manifest zurueckstellen. Bereits gesetzte
# Werte gewinnen, damit man beim Wiedereinstieg etwas aendern kann.
export FINAL_BATCH_SIZE="${FINAL_BATCH_SIZE:-$BATCH}"
export FINAL_ACCUM="${FINAL_ACCUM:-$ACCUM}"
export FINAL_WALLTIME="${FINAL_WALLTIME:-$WALL}"
export RUN_ID_EXTRA="${RUN_ID_EXTRA:-$EXTRA}"
export RESUME="$CKPT"

# Snapshots wieder ueber die volle Walltime verteilen.
STUNDEN="${FINAL_WALLTIME%%:*}"
export FINAL_SNAPSHOT_HOURS="${FINAL_SNAPSHOT_HOURS:-$(seq 6 6 "$STUNDEN" | tr '\n' ' ')}"

echo "[resume] RESUME=$RESUME"
echo "[resume] HORIZON_FILE=$HORIZON_FILE"
echo "[resume] FINAL_TRAIN_CONFIG=${FINAL_TRAIN_CONFIG:-<nicht gesetzt>}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 -- nicht eingereicht."
    exit 0
fi

bash "${HIER}/submit_final.sh" "$MODEL" "$SEED"
