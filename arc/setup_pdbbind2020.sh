#!/bin/bash -l
#
# BAUT DEN PDBBIND-v2020-BESTAND AUF ARC AUF.
#
# WARUM EIN NEUER BAUM STATT EINES UPGRADES
#   Der vorhandene Bestand unter ${DATA_ROOT}/pdbbind/ ist PDBbind v2020.R1
#   (19.037 Komplexe) -- dieselben Komplexe wie v2020, aber mit der Pipeline
#   von v2024 NEU AUFBEREITET und um die 406 bereinigt, die PDBbind selbst
#   bis v2024 aussortiert hat. R1 ist also die BESSERE Ausgabe: Buttenschoen
#   (2025), Tabelle 3.2, misst an den Kristallposen 78,2 % bestandene
#   PoseBusters-Pruefungen gegen 63,0 % beim Original.
#
#   (Eine fruehere Fassung dieses Kommentars sprach von einem "Stichtag 2019".
#   Das war falsch begruendet: v2020 beruht laut readme/PDBbind-101.txt:42 auf
#   dem PDB-Stand vom 1.1.2020 und enthaelt NULL Eintraege mit Freigabejahr
#   2020 -- ein Bestand, der 2019 endet, ist also genau das, wie v2020
#   aussieht, und unterscheidet die Ausgaben nicht.)
#
#   Alle bisherigen Messungen -- die 72-h-Laeufe, die Kristallposen-
#   Auswertung, kaputte_komplexe.txt, die Verlustmessung -- beziehen sich auf
#   R1. Wird er ueberschrieben, ist nichts davon mehr reproduzierbar. Das
#   Original kommt deshalb DANEBEN, nach ${DATA_ROOT}/pdbbind2020/.
#
# WAS v2020 LIEFERT
#   PDBbind verteilt den general set NICHT als ein Archiv, sondern als zwei:
#       refined-set        5316 Komplexe
#       v2020-other-PL    14127 Komplexe   (general MINUS refined)
#   Die Vereinigung ist der general set: 5316 + 14127 = 19443. Das ist die
#   Zahl aus dem SigmaDock-Paper (Zeile 443: "a curated set of 19,443
#   protein-ligand complexes"). Die 406 Komplexe Unterschied zu R1 sind
#   PDBbinds eigene Aussortierung, kein Verlust -- und sie treffen fast nur
#   den general-only-Teil: von den 5316 refined-Codes fehlen in R1 nur 9.
#
# DIE FALLE AUF OBERSTER EBENE
#   Beide Archive enthalten neben den Komplexordnern auch `index/` und
#   `readme/`. `datafronts.py:74` laeuft mit `sorted(dataroot.iterdir())` und
#   `subdir.is_dir()` darueber -- die beiden wuerden als Komplexe gezaehlt,
#   faenden keine Dateien und fielen STILL weg. Dieses Skript verlinkt
#   deshalb nur, was wie ein PDB-Code aussieht: vier Zeichen, [0-9a-z].
#
# WARUM SYMLINKS FUER general-set
#   Die Daten liegen einmal unter _stock/. general-set/ und refined-set/ sind
#   reine Auswahlen. datafronts.py folgt Symlinks (`iterdir` und `is_dir`
#   tun das), und `relative_to(dataroot)` greift weiterhin, weil die Pfade
#   unaufgeloest bleiben. Keine Codeaenderung noetig.
#
# PRUEFSUMMEN
#   Die drei SHA256 sind lokal auf dem Rechner der Nutzerin berechnet worden,
#   BEVOR uebertragen wurde. Eine abgebrochene scp-Uebertragung liefert eine
#   kuerzere, aber syntaktisch gueltige .gz -- tar wuerde sie teilweise
#   entpacken und wir haetten still zu wenige Komplexe. Deshalb wird hier
#   verglichen und nicht nur entpackt.
#
# Aufruf:
#   bash arc/setup_pdbbind2020.sh
#   FORCE=1 bash arc/setup_pdbbind2020.sh      # vorhandenen Baum ersetzen
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/stat-cadd/shug8458/data}"
ARCHIV="${ARCHIV:-${DATA_ROOT}/pdbbind2020_archive}"
ZIEL="${ZIEL:-${DATA_ROOT}/pdbbind2020}"
STOCK="${ZIEL}/_stock"

# Lokal berechnet am 2026-09-07, vor der Uebertragung.
SUM_OTHER="00f12ea202d59f3753368ea56314e6c9460e353ff4616781f94f7bbf3ec782cf"
SUM_INDEX="64efe7994fba1ad4a47de223dd3c0d01e0c5f09a722a7f138000ea4f74fe342e"
SUM_REFINED="403bd2612a7f40f67aa3b5c8c99285995320c67dc5f55507ff536f4832107e7e"

# Aus der Beschreibung auf der Downloadseite. Weichen die tatsaechlichen
# Zahlen ab, ist etwas beim Entpacken schiefgegangen.
N_ERWARTET_REFINED=5316
N_ERWARTET_OTHER=14127

# --- Vorpruefungen ---------------------------------------------------------
for f in PDBbind_v2020_refined.tar.gz PDBbind_v2020_other_PL.tar.gz \
         PDBbind_v2020_plain_text_index.tar.gz; do
    [ -f "${ARCHIV}/${f}" ] || { echo "FEHLER: fehlt: ${ARCHIV}/${f}" >&2; exit 2; }
done

echo "[setup] Pruefsummen ..."
pruefe() {   # $1 = Datei, $2 = erwartete Summe
    local ist
    ist="$(sha256sum "${ARCHIV}/$1" | awk '{print $1}')"
    if [ "$ist" != "$2" ]; then
        echo "FEHLER: $1 hat SHA256 ${ist}," >&2
        echo "        erwartet ${2}. Uebertragung unvollstaendig -- neu kopieren." >&2
        exit 1
    fi
    echo "  ok  $1"
}
pruefe PDBbind_v2020_refined.tar.gz          "$SUM_REFINED"
pruefe PDBbind_v2020_other_PL.tar.gz         "$SUM_OTHER"
pruefe PDBbind_v2020_plain_text_index.tar.gz "$SUM_INDEX"

# Entpackt braucht der Bestand ein Vielfaches der 2,5 GB Archivgroesse
# (PDB-Dateien sind Text und komprimieren stark). Lieber hier abbrechen als
# nach 20 Minuten mit einem halb entpackten Baum.
FREI_KB="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
FREI_GB=$(( FREI_KB / 1024 / 1024 ))
echo "[setup] frei unter ${DATA_ROOT}: ${FREI_GB} GB"
if [ "$FREI_GB" -lt 40 ]; then
    echo "FEHLER: unter 40 GB frei. Der entpackte Bestand braucht rund 25 GB." >&2
    echo "        Mit MEHR_PLATZ=1 trotzdem erzwingen." >&2
    [ "${MEHR_PLATZ:-0}" = "1" ] || exit 1
fi

# --- Ziel vorbereiten ------------------------------------------------------
if [ -d "$ZIEL" ] && [ -n "$(ls -A "$ZIEL" 2>/dev/null)" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "[setup] ${ZIEL} ist nicht leer. Zum Neuanlegen: FORCE=1 bash $0"
        exit 0
    fi
    echo "[setup] ersetze vorhandenes ${ZIEL}"
    rm -rf "$ZIEL"
fi
mkdir -p "$STOCK"

# --- Entpacken -------------------------------------------------------------
echo "[setup] entpacke (das dauert einige Minuten) ..."
for f in PDBbind_v2020_refined.tar.gz PDBbind_v2020_other_PL.tar.gz \
         PDBbind_v2020_plain_text_index.tar.gz; do
    echo "  ${f}"
    tar -xzf "${ARCHIV}/${f}" -C "$STOCK"
done

[ -d "${STOCK}/refined-set" ]    || { echo "FEHLER: refined-set fehlt nach dem Entpacken." >&2; exit 1; }
[ -d "${STOCK}/v2020-other-PL" ] || { echo "FEHLER: v2020-other-PL fehlt nach dem Entpacken." >&2; exit 1; }

# --- Codes einsammeln ------------------------------------------------------
# Nur vierstellige [0-9a-z]-Namen. Das schliesst index/ und readme/ aus,
# ohne sie namentlich zu kennen -- faellt spaeter ein weiterer Sonderordner
# dazu, greift dieselbe Regel.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

sammle() {   # $1 = Verzeichnis -> Codeliste auf stdout
    find "$1" -maxdepth 1 -mindepth 1 -xtype d -printf '%f\n' \
      | grep -E '^[0-9a-z]{4}$' | sort
}
sammle "${STOCK}/refined-set"    > "$TMP/refined"
sammle "${STOCK}/v2020-other-PL" > "$TMP/other"

N_REFINED="$(wc -l < "$TMP/refined")"
N_OTHER="$(wc -l < "$TMP/other")"
echo "[setup] refined       : ${N_REFINED}  (erwartet ${N_ERWARTET_REFINED})"
echo "[setup] other-PL      : ${N_OTHER}  (erwartet ${N_ERWARTET_OTHER})"
if [ "$N_REFINED" -ne "$N_ERWARTET_REFINED" ] || [ "$N_OTHER" -ne "$N_ERWARTET_OTHER" ]; then
    echo "FEHLER: Anzahlen weichen ab. Entpacken unvollstaendig?" >&2
    exit 1
fi

# refined und other-PL muessen DISJUNKT sein -- "general minus refined".
# Waeren sie es nicht, zaehlte die Vereinigung doppelt und alle abgeleiteten
# Zahlen (FINAL_N_TRAIN, Epochenhorizont) waeren falsch.
N_DOPPELT="$(comm -12 "$TMP/refined" "$TMP/other" | wc -l)"
if [ "$N_DOPPELT" -ne 0 ]; then
    echo "FEHLER: ${N_DOPPELT} Codes stehen in BEIDEN Archiven." >&2
    comm -12 "$TMP/refined" "$TMP/other" | head -5 >&2
    exit 1
fi
sort -u "$TMP/refined" "$TMP/other" > "$TMP/general"
N_GENERAL="$(wc -l < "$TMP/general")"
echo "[setup] general (Vereinigung): ${N_GENERAL}"
if [ "$N_GENERAL" -ne $(( N_ERWARTET_REFINED + N_ERWARTET_OTHER )) ]; then
    echo "FEHLER: Vereinigung ist ${N_GENERAL}, erwartet 19443." >&2
    exit 1
fi

# --- Symlinkbaeume ---------------------------------------------------------
verlinke() {   # $1 = Zielverzeichnis, $2 = Codeliste
    local ziel="$1" liste="$2" c src n
    mkdir -p "$ziel"
    while read -r c; do
        if [ -d "${STOCK}/refined-set/${c}" ]; then
            src="${STOCK}/refined-set/${c}"
        else
            src="${STOCK}/v2020-other-PL/${c}"
        fi
        ln -sfn "$src" "${ziel}/${c}"
    done < "$liste"
    n="$(find "$ziel" -maxdepth 1 -mindepth 1 | wc -l)"
    if [ "$n" -ne "$(wc -l < "$liste")" ]; then
        echo "FEHLER: ${ziel} hat ${n} Eintraege, erwartet $(wc -l < "$liste")." >&2
        exit 1
    fi
    echo "[setup] $(basename "$ziel"): ${n} Symlinks"
}
verlinke "${ZIEL}/general-set" "$TMP/general"
verlinke "${ZIEL}/refined-set" "$TMP/refined"

# --- Greifen die Regexe? ---------------------------------------------------
#   pdb_regex: ".*pocket\.pdb$"   sdf_regex: ".*ligand.*\.sdf$"
#   Fehlt eines von beiden, faende die Datafront kein Paar und der Komplex
#   fiele ohne Meldung weg. Das ist die Fehlerklasse, die hier am teuersten
#   ist, also wird sie vollstaendig geprueft, nicht per Stichprobe.
echo "[setup] pruefe Dateien in general-set (19443 Ordner, dauert kurz) ..."
OHNE_P=0; OHNE_L=0
for d in "${ZIEL}/general-set"/*; do
    compgen -G "${d}/*pocket.pdb"  > /dev/null || { OHNE_P=$(( OHNE_P + 1 )); echo "  ohne pocket: $(basename "$d")"; }
    compgen -G "${d}/*ligand*.sdf" > /dev/null || { OHNE_L=$(( OHNE_L + 1 )); echo "  ohne sdf   : $(basename "$d")"; }
done
echo "[setup] ohne *pocket.pdb: ${OHNE_P}   ohne *ligand*.sdf: ${OHNE_L}"
if [ "$OHNE_P" -gt 0 ] || [ "$OHNE_L" -gt 0 ]; then
    echo "FEHLER: unvollstaendige Komplexe -- sie wuerden still wegfallen." >&2
    exit 1
fi

# --- INDEX-Dateien ---------------------------------------------------------
echo
echo "[setup] verfuegbare INDEX-Dateien:"
find "$STOCK" -name "INDEX*" -printf '  %p\n' | sort

echo
echo "=============================================================="
echo "[setup] fertig."
echo "  Bestand      : ${STOCK}"
echo "  general-set  : ${N_GENERAL}   <-- die Paper-Zahl 19443"
echo "  refined-set  : ${N_REFINED}"
echo
echo "  NOCH OFFEN: core-set. Der 285er Satz ist CASF-2016 und liegt NICHT"
echo "  in diesen drei Archiven. Ohne ihn gibt es kein val_exps=pdbbind-core."
echo "=============================================================="
