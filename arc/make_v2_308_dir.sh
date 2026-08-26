#!/bin/bash -l
#
# LEGT DAS VERZEICHNIS MIT DEN OFFIZIELLEN 308 KOMPLEXEN AN.
#
# WOZU
#   Der PoseBusters-v2-Satz aus der Zeitschriftenfassung, also die Menge, auf
#   der SigmaDock berichtet (Prat et al., S. 8: "a subset containing 308
#   structures"). Die vollstaendige Kopie unter posebusters_full/ enthaelt
#   alle 428; hier entsteht daneben ein Verzeichnis mit genau 308 Symlinks.
#
# WARUM SYMLINKS UND KEIN KOPIEREN
#   Die Daten liegen einmal. Ein zweites Mal 214 MB waeren verschwendet, und
#   zwei Kopien koennten auseinanderlaufen.
#
# WARUM NICHT EINE WHITELIST BEIM SAMPLING
#   `data.blacklist` ist trotz des Namens eine Whitelist, greift aber nur,
#   wenn der Experimentname woertlich "posebusters" lautet (sample.py:458).
#   Unter jedem anderen Namen wuerde sie STILL ignoriert -- es wuerden alle
#   428 gesampelt, und niemand haette es gemerkt. Die Auswahl steckt deshalb
#   im Verzeichnis.
#
# Aufruf:
#   bash arc/make_v2_308_dir.sh                 # legt an und prueft
#   FORCE=1 bash arc/make_v2_308_dir.sh         # vorhandenes Ziel ersetzen
#
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow}"
DATA_ROOT="${DATA_ROOT:-/data/stat-cadd/shug8458/data}"
QUELLE="${QUELLE:-${DATA_ROOT}/posebusters_full/posebusters_benchmark_set}"
ZIEL="${ZIEL:-${DATA_ROOT}/posebusters_v2_308}"
IDS="${IDS:-${REPO_ROOT}/SigmaFlow_Evaluation/reference/posebusters_v2_308_ids.csv}"

# --- Vorpruefungen: lieber hier scheitern als beim Sampling ----------------
[ -d "$QUELLE" ] || { echo "FEHLER: Quelle fehlt: ${QUELLE}" >&2; exit 2; }
[ -f "$IDS" ]    || { echo "FEHLER: ID-Liste fehlt: ${IDS}" >&2; exit 2; }

N_QUELLE="$(find "$QUELLE" -maxdepth 1 -type d -regextype posix-extended \
            -regex '.*/[0-9A-Za-z]{4}_[0-9A-Za-z]{1,3}$' | wc -l)"
echo "[308] Quelle : ${QUELLE}  (${N_QUELLE} Komplexe)"
if [ "$N_QUELLE" -lt 308 ]; then
    echo "FEHLER: Quelle hat nur ${N_QUELLE} Komplexe, mindestens 308 noetig." >&2
    echo "        Ist das die vollstaendige Kopie oder die abgebrochene Entpackung?" >&2
    exit 2
fi

if [ -e "$ZIEL" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "[308] ${ZIEL} existiert bereits."
        echo "      Eintraege: $(find "$ZIEL" -maxdepth 1 -mindepth 1 | wc -l)"
        echo "      Zum Neuanlegen: FORCE=1 bash arc/make_v2_308_dir.sh"
        exit 0
    fi
    echo "[308] ersetze vorhandenes ${ZIEL}"
    rm -rf "$ZIEL"
fi
mkdir -p "$ZIEL"

# --- Symlinks anlegen ------------------------------------------------------
FEHLEND=()
ANZ=0
while IFS=, read -r PDB CCD _rest; do
    # Kopfzeile ueberspringen
    [ "$PDB" = "pdb_id" ] && continue
    [ -z "${PDB:-}" ] && continue
    CID="$(echo "${PDB}_${CCD}" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')"
    SRC="${QUELLE}/${CID}"
    if [ ! -d "$SRC" ]; then
        FEHLEND+=("$CID")
        continue
    fi
    ln -sfn "$SRC" "${ZIEL}/${CID}"
    ANZ=$(( ANZ + 1 ))
done < "$IDS"

echo "[308] verlinkt: ${ANZ}"
if [ "${#FEHLEND[@]}" -gt 0 ]; then
    echo "FEHLER: ${#FEHLEND[@]} ID(s) nicht in der Quelle gefunden:" >&2
    printf '        %s\n' "${FEHLEND[@]:0:10}" >&2
    echo "        Das Verzeichnis waere unvollstaendig -- Abbruch." >&2
    exit 1
fi

# --- Nachpruefung: Zahl UND Inhalt ----------------------------------------
N_ZIEL="$(find "$ZIEL" -maxdepth 1 -mindepth 1 | wc -l)"
if [ "$N_ZIEL" -ne 308 ]; then
    echo "FEHLER: ${N_ZIEL} Eintraege statt 308." >&2
    exit 1
fi

# Stichprobe: die Regexe aus posebusters308.yaml muessen greifen, sonst
# faende die Datafront die Paare nicht und saemtliche Posen fielen still weg.
OHNE_PDB=0
OHNE_SDF=0
for D in "$ZIEL"/*; do
    compgen -G "${D}/*.pdb" > /dev/null || OHNE_PDB=$(( OHNE_PDB + 1 ))
    compgen -G "${D}/*ligands.sdf" > /dev/null || OHNE_SDF=$(( OHNE_SDF + 1 ))
done
echo "[308] ohne .pdb: ${OHNE_PDB}   ohne *ligands.sdf: ${OHNE_SDF}"
if [ "$OHNE_PDB" -gt 0 ] || [ "$OHNE_SDF" -gt 0 ]; then
    echo "FEHLER: Nicht jeder Komplex hat beide Dateien -- die Regexe aus" >&2
    echo "        posebusters308.yaml wuerden ins Leere greifen." >&2
    exit 1
fi

echo
echo "=============================================================="
echo "[308] fertig: ${ZIEL}"
echo "      308 Komplexe, alle mit .pdb und *ligands.sdf"
echo
echo "      Benutzung beim Sampling:  experiment=posebusters308"
echo "      Referenzliganden:         --true_dir ${ZIEL}"
echo "=============================================================="
