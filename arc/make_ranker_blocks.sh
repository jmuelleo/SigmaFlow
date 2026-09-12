#!/bin/bash -l
#
# PDBBIND-GENERAL IN VIER BLOECKE TEILEN, FUER DIE RANKER-DATEN
#
# WOZU
#   Ein Sampling-Durchlauf ueber alle 19.037 Komplexe dauert bei fuenf
#   Integrationsschritten rund 7,6 h und passt damit nicht in die
#   Sechs-Stunden-Grenze von `short`. Vier Bloecke ergeben Aufgaben von rund
#   1,9 h, und zehn Seeds mal vier Bloecke sind 40 Aufgaben, die gleichzeitig
#   laufen koennen.
#
# WIE GETEILT WIRD
#   Reihum, nicht am Stueck: die sortierte Komplexliste wird auf die Bloecke
#   verteilt wie Karten auf vier Spieler. Am Stueck geteilt wuerde jeder Block
#   einen zusammenhaengenden Bereich der alphabetisch sortierten PDB-Codes
#   bekommen, und PDB-Codes sind nicht zufaellig vergeben -- benachbarte Codes
#   stammen oft aus derselben Hinterlegung, also aus derselben Proteinfamilie.
#   Reihum verteilt ist jeder Block eine Stichprobe des ganzen Satzes, und ein
#   fehlgeschlagener Block verzerrt die Daten nicht systematisch.
#
# WAS ENTSTEHT
#   ${ARC_DATA}/pdbbind/ranker-block-<n>/    Symlinks auf die Komplexordner
#   <CODE_DIR>/conf/experiments/pdbbind-ranker-<n>.yaml
#
#   Die Symlinks kosten nur Verzeichniseintraege, keine Kopien.
#
# WARUM DIE KONFIGURATION IN DEN CODEBAUM MUSS
#   config.py:889 sucht sie unter Path(config.py).parent.parent.parent/conf,
#   also im Baum des jeweiligen Arms -- nicht im Repo-Wurzelverzeichnis.
#
# Aufruf:
#   CODE_DIR=$PWD/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field \
#     bash arc/make_ranker_blocks.sh
#
#   BLOECKE=4 und QUELLE lassen sich ueberschreiben.
#
set -uo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[ -f "${HIER}/_common.sh" ] && source "${HIER}/_common.sh"

ARC_DATA="${ARC_DATA:?ARC_DATA nicht gesetzt}"
CODE_DIR="${CODE_DIR:?CODE_DIR setzen, z.B. der EXP-110-Baum}"
BLOECKE="${BLOECKE:-4}"
QUELLE="${QUELLE:-${ARC_DATA}/pdbbind/general-set}"

[ -d "$QUELLE" ] || { echo "FEHLER: Quelle fehlt: $QUELLE" >&2; exit 1; }
[ -d "${CODE_DIR}/conf/experiments" ] || {
    echo "FEHLER: ${CODE_DIR}/conf/experiments fehlt." >&2
    echo "        CODE_DIR muss der Baum des Arms sein, nicht das Repo." >&2
    exit 1
}

mapfile -t ALLE < <(find "$QUELLE" -mindepth 1 -maxdepth 1 -type d | sort)
N="${#ALLE[@]}"
[ "$N" -gt 0 ] || { echo "FEHLER: keine Komplexordner in $QUELLE" >&2; exit 1; }

echo "Quelle          : $QUELLE"
echo "Komplexe        : $N"
echo "Bloecke         : $BLOECKE  (reihum verteilt)"
echo "Konfigurationen : ${CODE_DIR}/conf/experiments/"
echo

for ((b = 0; b < BLOECKE; b++)); do
    ZIEL="${ARC_DATA}/pdbbind/ranker-block-${b}"
    rm -rf "$ZIEL"
    mkdir -p "$ZIEL"
    anzahl=0
    for ((i = b; i < N; i += BLOECKE)); do
        ln -s "${ALLE[$i]}" "${ZIEL}/$(basename "${ALLE[$i]}")"
        anzahl=$((anzahl + 1))
    done

    # Die Regexe stammen woertlich aus pdbbind-general.yaml. Sie hier neu zu
    # erfinden waere die haeufigste Art, still den falschen Satz zu sampeln.
    # Einfache Anfuehrungszeichen: in YAML hat der Backslash dort KEINE
    # Sonderbedeutung. Mit doppelten muesste \\. stehen, und jedes Heredoc
    # dazwischen kann die Verdopplung verschlucken -- genau daran ist die
    # erste Fassung gescheitert, sie erzeugte unparsbares YAML.
    {
        echo "_target_: sigmadock.experiments.ExperimentConfig"
        echo "name: 'pdbbind-ranker-${b}'"
        echo "dataset: 'pdbbind/ranker-block-${b}/'"
        printf "pdb_regex: '.*pocket[.]pdb$'\n"
        printf "sdf_regex: '.*ligand.*[.]sdf$'\n"
    } > "${CODE_DIR}/conf/experiments/pdbbind-ranker-${b}.yaml"

    echo "Block ${b}: ${anzahl} Komplexe -> ${ZIEL}"
done

echo
SUMME=0
for ((b = 0; b < BLOECKE; b++)); do
    k=$(find "${ARC_DATA}/pdbbind/ranker-block-${b}" -mindepth 1 -maxdepth 1 | wc -l)
    SUMME=$((SUMME + k))
done
echo "Summe ueber alle Bloecke: ${SUMME} (erwartet ${N})"
[ "$SUMME" -eq "$N" ] || { echo "FEHLER: Summe passt nicht." >&2; exit 1; }

# Gegenprobe, mit der Traversierung der Datenfront selbst.
# `find` ohne -L folgt keinen Symlinks und meldete hier faelschlich einen
# Fehler. datafronts.py:73-83 benutzt dagegen iterdir und is_dir, und beide
# folgen Symlinks -- die Komplexe sind also sichtbar.
python - "${ARC_DATA}/pdbbind/ranker-block-0" <<'PYPROBE'
import re, sys
from pathlib import Path

wurzel = Path(sys.argv[1])
pdb = re.compile(r".*pocket[.]pdb$")
sdf = re.compile(r".*ligand.*[.]sdf$")

ordner = paare = 0
for sub in sorted(wurzel.iterdir()):
    if not sub.is_dir():
        continue
    ordner += 1
    hat_pdb = hat_sdf = False
    for f in sub.iterdir():
        if f.suffix.lower() == ".pdb" and pdb.search(f.name):
            hat_pdb = True
        elif f.suffix.lower() == ".sdf" and sdf.search(f.name):
            hat_sdf = True
    if hat_pdb and hat_sdf:
        paare += 1

print(f"Gegenprobe Block 0: {ordner} Ordner sichtbar, {paare} mit Paar "
      f"aus pocket.pdb und ligand.sdf")
if paare == 0:
    print("FEHLER: kein einziges Paar gefunden.")
    sys.exit(1)
if paare < 0.95 * ordner:
    print(f"WARNUNG: nur {100*paare/ordner:.1f} Prozent der Ordner haben ein "
          f"vollstaendiges Paar.")
PYPROBE
