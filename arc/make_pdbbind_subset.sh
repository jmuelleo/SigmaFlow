#!/bin/bash -l
#
# LEGT refined-set/ ODER core-set/ ALS SYMLINKS NACH general-set/ AN.
#
# WOZU
#   Auf ARC liegt nur der flache Gesamtbestand (general-set, 19037 Komplexe,
#   Stichtag 2019). refined-set/ und core-set/ existieren als LEERE
#   Verzeichnisse. Das Rezept des Papers braucht aber beide:
#       train_exps: [pdbbind-general, pdbbind-refined]
#       val_exps:   [pdbbind-core]
#   Beide Teilmengen sind reine Auswahlen aus dem Gesamtbestand, definiert
#   durch eine Liste von PDB-Codes -- es sind KEINE zusaetzlichen Daten.
#
# WARUM SYMLINKS
#   datafronts.py:74 laeuft mit `sorted(self.dataroot.iterdir())` ueber die
#   oberste Ebene und prueft `subdir.is_dir()`. Beides folgt Symlinks. Der
#   Ladepfad merkt also keinen Unterschied, und die Daten liegen einmal.
#
# WARUM FEHLENDE CODES HIER KEIN ABBRUCH SIND
#   Der Bestand endet 2019, eine INDEX-Datei von 2020 listet ~400 Komplexe
#   mehr. Fehlende Eintraege sind deshalb erwartet. Sie werden gezaehlt und
#   protokolliert; erst oberhalb von MAX_FEHLEND_PROZENT wird abgebrochen,
#   weil das auf die falsche Liste hindeutet.
#
# LISTENFORMAT
#   Eine Zeile je Komplex, der PDB-Code im ersten Feld. Zeilen, die mit '#'
#   beginnen, werden uebersprungen. Das ist genau das Format der
#   PDBBind-Dateien INDEX_refined_data.* und INDEX_core_data.*, funktioniert
#   aber auch mit einer schlichten Liste aus einem Code je Zeile.
#
# Aufruf:
#   bash arc/make_pdbbind_subset.sh refined /pfad/INDEX_refined_data.2020
#   bash arc/make_pdbbind_subset.sh core    /pfad/INDEX_core_data.2016
#   FORCE=1 bash arc/make_pdbbind_subset.sh refined /pfad/liste
#
set -euo pipefail

WELCHE="${1:-}"
LISTE="${2:-}"
DATA_ROOT="${DATA_ROOT:-/data/stat-cadd/shug8458/data}"
QUELLE="${QUELLE:-${DATA_ROOT}/pdbbind/general-set}"
MAX_FEHLEND_PROZENT="${MAX_FEHLEND_PROZENT:-8}"

if [ -z "$WELCHE" ] || [ -z "$LISTE" ]; then
    echo "Aufruf: bash arc/make_pdbbind_subset.sh {refined|core} <listendatei>" >&2
    exit 2
fi
# ZIEL ist ueberschreibbar, weil es ZWEI Bestaende gibt: den R1-Bestand unter
# pdbbind/ und den Original-v2020-Bestand unter pdbbind2020/. Beide brauchen
# ein eigenes core-set aus DERSELBEN Codeliste, aber mit VERSCHIEDENER Quelle
# -- die Strukturdateien sind unterschiedlich aufbereitet, und jeder Arm muss
# auf den Dateien validieren, auf deren Aufbereitung er auch trainiert.
# Ohne diesen Schalter wuerde der zweite Aufruf den ersten ueberschreiben.
case "$WELCHE" in
    refined) ZIEL="${ZIEL:-${DATA_ROOT}/pdbbind/refined-set}" ;;
    core)    ZIEL="${ZIEL:-${DATA_ROOT}/pdbbind/core-set}" ;;
    *) echo "FEHLER: erstes Argument muss 'refined' oder 'core' sein." >&2; exit 2 ;;
esac
echo "[${WELCHE}] Ziel  : ${ZIEL}"

# --- Vorpruefungen ---------------------------------------------------------
[ -d "$QUELLE" ] || { echo "FEHLER: Quelle fehlt: ${QUELLE}" >&2; exit 2; }
[ -f "$LISTE" ]  || { echo "FEHLER: Liste fehlt: ${LISTE}" >&2; exit 2; }

N_QUELLE="$(find "$QUELLE" -maxdepth 1 -mindepth 1 -xtype d | wc -l)"
echo "[${WELCHE}] Quelle: ${QUELLE}  (${N_QUELLE} Komplexe)"
if [ "$N_QUELLE" -lt 1000 ]; then
    echo "FEHLER: Die Quelle hat nur ${N_QUELLE} Ordner. Ist das wirklich" >&2
    echo "        der Gesamtbestand?" >&2
    exit 2
fi

# --- Ziel vorbereiten ------------------------------------------------------
if [ -d "$ZIEL" ] && [ "$(find "$ZIEL" -maxdepth 1 -mindepth 1 | wc -l)" -gt 0 ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "[${WELCHE}] ${ZIEL} ist nicht leer:"
        echo "           $(find "$ZIEL" -maxdepth 1 -mindepth 1 | wc -l) Eintraege"
        echo "           Zum Neuanlegen: FORCE=1 bash arc/make_pdbbind_subset.sh ..."
        exit 0
    fi
    echo "[${WELCHE}] ersetze vorhandenes ${ZIEL}"
    rm -rf "$ZIEL"
fi
mkdir -p "$ZIEL"

# --- Symlinks anlegen ------------------------------------------------------
PROTOKOLL="${ZIEL}.fehlend.txt"
: > "$PROTOKOLL"
ANZ=0
FEHLEND=0
GELISTET=0

while read -r ERSTES _rest; do
    case "$ERSTES" in ''|'#'*) continue ;; esac
    CODE="$(echo "$ERSTES" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    GELISTET=$(( GELISTET + 1 ))
    SRC="${QUELLE}/${CODE}"
    if [ ! -d "$SRC" ]; then
        echo "$CODE" >> "$PROTOKOLL"
        FEHLEND=$(( FEHLEND + 1 ))
        continue
    fi
    ln -sfn "$SRC" "${ZIEL}/${CODE}"
    ANZ=$(( ANZ + 1 ))
done < "$LISTE"

echo "[${WELCHE}] in der Liste: ${GELISTET}"
echo "[${WELCHE}] verlinkt    : ${ANZ}"
echo "[${WELCHE}] nicht im Bestand: ${FEHLEND}   (protokolliert in ${PROTOKOLL})"

if [ "$GELISTET" -eq 0 ]; then
    echo "FEHLER: Die Liste enthielt keine verwertbare Zeile. Falsches Format?" >&2
    exit 1
fi

PROZENT=$(( 100 * FEHLEND / GELISTET ))
if [ "$PROZENT" -gt "$MAX_FEHLEND_PROZENT" ]; then
    echo "FEHLER: ${PROZENT}% der gelisteten Codes fehlen im Bestand," >&2
    echo "        erlaubt sind ${MAX_FEHLEND_PROZENT}%. Das deutet auf die falsche" >&2
    echo "        Liste oder ein anderes Namensschema hin -- Abbruch." >&2
    echo "        Erste fehlende Codes:" >&2
    head -10 "$PROTOKOLL" >&2
    exit 1
fi

# --- Nachpruefung: greifen die Regexe aus conf/datasets.yaml? --------------
#   pdb_regex: ".*pocket\.pdb$"      sdf_regex: ".*ligand.*\.sdf$"
#   Greift eines von beiden nicht, faende die Datafront kein Paar und der
#   Komplex fiele STILL weg -- genau der Fehler, der schwer zu bemerken ist.
OHNE_POCKET=0
OHNE_LIGAND=0
for D in "$ZIEL"/*; do
    compgen -G "${D}/*pocket.pdb" > /dev/null || OHNE_POCKET=$(( OHNE_POCKET + 1 ))
    compgen -G "${D}/*ligand*.sdf" > /dev/null || OHNE_LIGAND=$(( OHNE_LIGAND + 1 ))
done
echo "[${WELCHE}] ohne *pocket.pdb: ${OHNE_POCKET}   ohne *ligand*.sdf: ${OHNE_LIGAND}"
if [ "$OHNE_POCKET" -gt 0 ] || [ "$OHNE_LIGAND" -gt 0 ]; then
    echo "FEHLER: Nicht jeder verlinkte Komplex hat beide Dateien. Die Regexe" >&2
    echo "        aus conf/datasets.yaml wuerden ins Leere greifen." >&2
    exit 1
fi

echo
echo "=============================================================="
echo "[${WELCHE}] fertig: ${ZIEL}"
echo "          ${ANZ} Komplexe, alle mit *pocket.pdb und *ligand*.sdf"
echo
echo "          Benutzung:  --train_exps pdbbind-general pdbbind-refined"
echo "                      --val_exps pdbbind-core"
echo "=============================================================="
