#!/bin/bash -l
#
# KURATIERTER TRAININGSSATZ + AUSGEHALTENER VALIDIERUNGSSATZ.
#
# WAS ENTFERNT WIRD
#   Die Codes aus `kaputte_komplexe.txt`, erzeugt aus den Kristallposen-
#   Messungen (arc/crystal_pb.slurm + arc/analyse_crystal_pb.py):
#     extern  Ligandatom naeher als 1 A an einem Proteinatom -- physikalisch
#             unmoeglich, 208 davon bei Abstand exakt null, teils zwischen
#             VERSCHIEDENEN Elementen (Cl-S, I-N). Hinterlegungsartefakte.
#     intern  mindestens eine der 15 ligandinternen Pruefungen faellt durch.
#     beides  beides zugleich.
#   Zusammen 1067 von 19037, also 5,6 %.
#
#   NICHT entfernt werden kovalente Komplexe (rund 4,2 %). Ihre kurzen
#   Kontakte sind korrekte Chemie, fuer die ein vdW-Kriterium nicht gemacht
#   ist -- keine Fehler, auch wenn PoseBusters sie meldet.
#
# N_VAL > 0 HAELT ZUSAETZLICH EINEN VALIDIERUNGSSATZ ZURUECK
#   Deterministisch (Sortierung plus festes Raster), im Trainingsverzeichnis
#   NICHT enthalten.
#
#   WARUM AUSGEHALTEN UND NICHT WIE IM ORIGINAL
#     Das publizierte Rezept validiert auf `pdbbind-core`, und core ist
#     Teilmenge von refined ist Teilmenge von general -- es validiert also auf
#     Daten, auf denen es trainiert. Mit `monitor_metric = loss_val/total`
#     waehlt Early Stopping den Checkpoint dann teilweise nach
#     Auswendiglernen. Eine ausgehaltene Teilmenge ist eine Abweichung vom
#     Original, aber in die richtige Richtung -- und ohne
#     INDEX_core_data.2016 der einzige Weg.
#
# WARUM SYMLINKS
#   datafronts.py:74 laeuft mit `sorted(self.dataroot.iterdir())` ueber die
#   oberste Ebene und filtert mit `subdir.is_dir()`. Beides folgt Symlinks,
#   und `relative_to(dataroot)` greift, weil die Pfade unaufgeloest bleiben.
#   Keine Codeaenderung noetig, und die Daten liegen einmal.
#
# WARUM MENGENOPERATIONEN STATT EINER SCHLEIFE MIT grep
#   Die erste Fassung rief je Verzeichnis ein eigenes `grep` ueber die
#   Ausschlussliste auf -- rund 40.000 Prozesse, Minuten Laufzeit, und ein
#   Verbindungsabbruch riss den Lauf mit. `comm` auf sortierten Listen macht
#   dasselbe in Sekunden.
#
# WAS DER LAUF DAMIT NICHT WIRD
#   Der gemessene Hebel der entfernten Komplexe auf den Verlust liegt bei
#   1,17 bis 1,28 (arc/verlust_je_komplex.py), also rund 13 % groessere
#   Gradienten bei 5,6 % der Daten. Ein grosser Effekt ist nicht zu erwarten.
#   Die Kuratierung ist billig und begruendbar, aber sie ist kein Hebel --
#   das gehoert so in den Text.
#
# Aufruf:
#   N_VAL=285 bash arc/make_kuratiert_dir.sh
#   FORCE=1 N_VAL=285 bash arc/make_kuratiert_dir.sh
#
#   Teilmenge mitkuratieren (refined muss mit, sonst tragen die kaputten
#   Komplexe sich ueber `train_exps: [general, refined]` wieder ein):
#     QUELLE=$D/pdbbind/refined-set ZIEL=$D/pdbbind/refined-set-kuratiert \
#       N_VAL=0 MIN_TREFFER_PROZENT=0 bash arc/make_kuratiert_dir.sh
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/stat-cadd/shug8458/data}"
QUELLE="${QUELLE:-${DATA_ROOT}/pdbbind/general-set}"
ZIEL="${ZIEL:-${DATA_ROOT}/pdbbind/general-set-kuratiert}"
ZIEL_VAL="${ZIEL_VAL:-${DATA_ROOT}/pdbbind/val-kuratiert}"
LISTE="${LISTE:-/data/stat-cadd/shug8458/arc_runs/kaputte_komplexe.txt}"
N_VAL="${N_VAL:-0}"

[ -d "$QUELLE" ] || { echo "FEHLER: Quelle fehlt: ${QUELLE}" >&2; exit 2; }
[ -f "$LISTE" ]  || { echo "FEHLER: Liste fehlt: ${LISTE}" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- Mengen aufbauen ------------------------------------------------------
find "$QUELLE" -maxdepth 1 -mindepth 1 -xtype d -printf '%f\n' | sort > "$TMP/alle"
N_QUELLE="$(wc -l < "$TMP/alle")"
echo "[kuratiert] Quelle : ${QUELLE}  (${N_QUELLE} Komplexe)"
if [ "$N_QUELLE" -lt 1000 ]; then
    echo "FEHLER: nur ${N_QUELLE} Ordner -- ist das der Gesamtbestand?" >&2
    exit 2
fi

awk -F, '!/^#/ && NF { gsub(/[ \t]/,"",$1); if ($1 != "") print $1 }' "$LISTE" \
  | sort -u > "$TMP/raus"
echo "[kuratiert] Ausschlussliste: $(wc -l < "$TMP/raus") Codes"

# Gegenprobe: stehen die Codes ueberhaupt im Bestand?
#
#   TREFFER = 0 ist IMMER ein Abbruch. Es gibt keinen legitimen Fall, in dem
#   keiner der Ausschlusscodes vorkommt -- das heisst Gross-/Kleinschreibung
#   passt nicht oder es ist die falsche Liste, und wir wuerden still NICHTS
#   entfernen und trotzdem "fertig" melden.
#
#   Die Quote darueber ist eine andere Pruefung, und sie gilt nur fuer den
#   GESAMTBESTAND. Wird eine TEILMENGE kuratiert -- refined-set etwa, das
#   ueber `train_exps: [general, refined]` sonst die kaputten Komplexe wieder
#   hereintragen wuerde, weil MetaFront nicht dedupliziert -- kommen
#   naturgemaess nur wenige der Codes darin vor. Deshalb:
#       MIN_TREFFER_PROZENT=90   (Vorgabe, Gesamtbestand)
#       MIN_TREFFER_PROZENT=0    (Teilmenge, Quote bewusst abgeschaltet)
MIN_TREFFER_PROZENT="${MIN_TREFFER_PROZENT:-90}"
N_RAUS="$(wc -l < "$TMP/raus")"
TREFFER="$(comm -12 "$TMP/alle" "$TMP/raus" | wc -l)"
echo "[kuratiert] davon im Bestand gefunden: ${TREFFER} von ${N_RAUS}" \
     "(Mindestquote ${MIN_TREFFER_PROZENT} %)"
if [ "$TREFFER" -eq 0 ]; then
    echo "FEHLER: KEIN einziger Ausschlusscode kommt im Bestand vor." >&2
    echo "        Gross-/Kleinschreibung oder falsche Liste -- Abbruch." >&2
    exit 1
fi
if [ "$TREFFER" -lt "$(( N_RAUS * MIN_TREFFER_PROZENT / 100 ))" ]; then
    echo "FEHLER: nur ${TREFFER} von ${N_RAUS} Ausschlusscodes kommen im" >&2
    echo "        Bestand vor, gefordert sind ${MIN_TREFFER_PROZENT} %." >&2
    echo "        Kuratierst du eine Teilmenge? Dann MIN_TREFFER_PROZENT=0." >&2
    exit 1
fi

comm -23 "$TMP/alle" "$TMP/raus" > "$TMP/behalten"
N_BEHALTEN="$(wc -l < "$TMP/behalten")"
echo "[kuratiert] verbleiben: ${N_BEHALTEN}"

: > "$TMP/val"
if [ "$N_VAL" -gt 0 ]; then
    if [ "$N_VAL" -ge "$N_BEHALTEN" ]; then
        echo "FEHLER: N_VAL=${N_VAL} >= ${N_BEHALTEN}." >&2
        exit 2
    fi
    SCHRITT=$(( N_BEHALTEN / N_VAL ))
    awk -v s="$SCHRITT" -v n="$N_VAL" 'NR % s == 1 && c < n { print; c++ }' \
        "$TMP/behalten" > "$TMP/val"
    echo "[kuratiert] Validierung: $(wc -l < "$TMP/val") Komplexe, jeder ${SCHRITT}-te"
fi
comm -23 "$TMP/behalten" "$TMP/val" > "$TMP/train"
N_TRAIN="$(wc -l < "$TMP/train")"

# --- Verzeichnisse anlegen ------------------------------------------------
anlegen() {   # $1 = Zielpfad, $2 = Datei mit Codes
    local ziel="$1" liste="$2"
    if [ -d "$ziel" ] && [ -n "$(ls -A "$ziel" 2>/dev/null)" ]; then
        if [ "${FORCE:-0}" != "1" ]; then
            echo "[kuratiert] ${ziel} ist nicht leer. Zum Neuanlegen: FORCE=1 ..."
            return 1
        fi
        rm -rf "$ziel"
    fi
    mkdir -p "$ziel"
    while read -r c; do ln -sfn "${QUELLE}/${c}" "${ziel}/${c}"; done < "$liste"
    local n
    n="$(find "$ziel" -maxdepth 1 -mindepth 1 | wc -l)"
    if [ "$n" -ne "$(wc -l < "$liste")" ]; then
        echo "FEHLER: ${ziel} hat ${n} Eintraege, erwartet $(wc -l < "$liste")." >&2
        exit 1
    fi
    echo "[kuratiert] ${ziel}: ${n} Symlinks"
}

anlegen "$ZIEL" "$TMP/train" || exit 0
if [ "$N_VAL" -gt 0 ]; then
    anlegen "$ZIEL_VAL" "$TMP/val" || exit 0
fi

# --- Nachpruefungen -------------------------------------------------------
if [ "$(( N_TRAIN + $(wc -l < "$TMP/val") + TREFFER ))" -ne "$N_QUELLE" ]; then
    echo "FEHLER: Training + Validierung + Ausgeschlossen != Gesamtzahl." >&2
    exit 1
fi

# Greifen die Regexe aus conf/experiments/pdbbind-*.yaml? Ohne beide Dateien
# faende die Datafront kein Paar und der Komplex fiele STILL weg.
pruefe_dateien() {
    local ziel="$1" ohne_p=0 ohne_l=0 d
    for d in "$ziel"/*; do
        compgen -G "${d}/*pocket.pdb" > /dev/null || ohne_p=$(( ohne_p + 1 ))
        compgen -G "${d}/*ligand*.sdf" > /dev/null || ohne_l=$(( ohne_l + 1 ))
    done
    echo "[kuratiert] $(basename "$ziel"): ohne *pocket.pdb ${ohne_p}, ohne *ligand*.sdf ${ohne_l}"
    [ "$ohne_p" -eq 0 ] && [ "$ohne_l" -eq 0 ]
}
pruefe_dateien "$ZIEL" || { echo "FEHLER: unvollstaendige Komplexe im Training." >&2; exit 1; }
if [ "$N_VAL" -gt 0 ]; then
    pruefe_dateien "$ZIEL_VAL" || { echo "FEHLER: unvollstaendig in der Validierung." >&2; exit 1; }
fi

echo
echo "=============================================================="
echo "[kuratiert] fertig."
echo "  Gesamtbestand   : ${N_QUELLE}"
echo "  ausgeschlossen  : ${TREFFER}  ($(awk "BEGIN{printf \"%.2f\", 100*${TREFFER}/${N_QUELLE}}") %)"
echo "  Validierung     : $(wc -l < "$TMP/val")"
echo "  TRAINING        : ${N_TRAIN}   <-- als FINAL_N_TRAIN uebergeben"
echo
echo "  TRAIN_EXPS=pdbbind-general-kuratiert"
[ "$N_VAL" -gt 0 ] && echo "  VAL_EXPS=pdbbind-val-kuratiert"
echo "=============================================================="
