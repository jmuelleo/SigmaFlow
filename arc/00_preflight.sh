#!/bin/bash -l
#
# SCHRITT 0 — auf dem LOGIN-Knoten ausfuehren, NICHT mit sbatch.
#
# Prueft alles, was schiefgehen kann, bevor 24 GPU-Stunden verbrannt werden.
# Startet keinen Job und veraendert nichts ausser dem Anlegen von
# Log-/Laufverzeichnissen.
#
#   bash arc/00_preflight.sh
#
# Ausgabe endet mit READY oder mit einer Liste konkreter Blocker.

set -uo pipefail   # bewusst KEIN -e: wir wollen ALLE Probleme sehen, nicht nur das erste

# shellcheck source=/dev/null
source "$(dirname "$0")/_common.sh"

PROBLEMS=()
ok()   { echo "  [ok ] $*"; }
bad()  { echo "  [!!] $*"; PROBLEMS+=("$*"); }
note() { echo "  [ i] $*"; }

echo "=============================================================="
echo "ARC PREFLIGHT   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

echo
echo "1. Verzeichnisse"
for d in "$ARC_REPO" "$ARC_SIGMADOCK" "$ARC_DATA" "$ARC_SF_ENV" "$ARC_SD_ENV"; do
    [ -d "$d" ] && ok "$d" || bad "FEHLT: $d"
done
mkdir -p "${ARC_RUNS}/slurm_logs" && ok "Laufverzeichnis bereit: ${ARC_RUNS}"

echo
echo "2. Codestand im Repo"
if [ -d "${ARC_REPO}/.git" ]; then
    ok "Repo ist ein Git-Klon"
    echo "     HEAD:   $(git -C "$ARC_REPO" rev-parse --short HEAD 2>/dev/null)"
    echo "     Branch: $(git -C "$ARC_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    DIRTY="$(git -C "$ARC_REPO" status --short 2>/dev/null | head -5)"
    [ -z "$DIRTY" ] && ok "Arbeitsverzeichnis sauber" || note "lokale Aenderungen: $DIRTY"
else
    bad "${ARC_REPO} ist KEIN Git-Klon - 'git pull' wird nicht funktionieren, per rsync synchronisieren"
fi

# Die beiden Verzeichnisse, die diese Runde neu braucht.
for d in "${ARC_REPO}/SigmaFlow_Minimal" \
         "${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam"; do
    [ -d "$d" ] && ok "vorhanden: ${d#$ARC_REPO/}" \
                || bad "FEHLT: ${d#$ARC_REPO/}  -> zuerst pushen und hier pullen"
done

echo
echo "3. Der Frame-Fix muss im geladenen Code stehen, nicht nur im Ordner"
if [ -f "${ARC_REPO}/SigmaFlow_Minimal/src/sigmadock/diff/sigma_flow_generator.py" ]; then
    if grep -q 'R_t.transpose(-1, -2) @ updates\["omega"\] @ R_t' \
        "${ARC_REPO}/SigmaFlow_Minimal/src/sigmadock/diff/sigma_flow_generator.py"; then
        ok "Frame-Fix in SigmaFlow_Minimal vorhanden"
    else
        bad "FRAME-FIX FEHLT in SigmaFlow_Minimal - Upload unvollstaendig"
    fi
else
    bad "SigmaFlow_Minimal/src/.../sigma_flow_generator.py fehlt"
fi

echo
echo "4. EXP-100 muss die Reparametrisierung enthalten"
E100="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam/src/sigmadock"
if [ -f "${E100}/diff/state_reparam.py" ] && \
   grep -q "get_fragment_com_and_rot_reparam" "${E100}/diff/sigma_flow_generator.py" 2>/dev/null && \
   grep -q "ref_conf_pos" "${E100}/data.py" 2>/dev/null; then
    ok "state_reparam.py, Generator-Patch und ref_conf_pos vorhanden"
else
    bad "EXP-100 unvollstaendig hochgeladen"
fi

echo
echo "5. Datensaetze"
for d in pdbbind-general posebusters; do
    hits="$(find "${ARC_DATA}" -maxdepth 2 -iname "*${d%%-*}*" -type d 2>/dev/null | head -3)"
    [ -n "$hits" ] && ok "$d gefunden: $(echo "$hits" | head -1)" || bad "$d NICHT gefunden unter ${ARC_DATA}"
done

echo
echo "6. Conda-Umgebungen (zwei verschiedene, das ist Absicht)"
module load Mamba 2>/dev/null || note "module load Mamba fehlgeschlagen (evtl. schon geladen)"
if [ -x "${ARC_SF_ENV}/bin/python" ]; then
    ok "sigmaflow_env: $("${ARC_SF_ENV}/bin/python" --version 2>&1)"
else
    bad "sigmaflow_env/bin/python fehlt"
fi
if [ -x "${ARC_SD_ENV}/bin/python" ]; then
    ok "myenv (SigmaDock): $("${ARC_SD_ENV}/bin/python" --version 2>&1)"
else
    bad "myenv/bin/python fehlt - SigmaDock laeuft NICHT in sigmaflow_env"
fi

echo
echo "7. Partitionen und GPU-Klasse"
if command -v sinfo >/dev/null 2>&1; then
    sinfo -o "%20P %10a %12l %8D %s" 2>/dev/null | head -12
    note "Erwartet: short<=12h, medium<=48h, long<=30d, interactive<=24h"
    if sinfo -h -o "%P" 2>/dev/null | grep -q "^medium"; then
        ok "Partition 'medium' existiert (fuer 24h noetig, short reicht nicht)"
    else
        bad "Partition 'medium' nicht gefunden - Zeitlimit der 24h-Skripte pruefen"
    fi
    echo "  GPU-Typen laut sinfo:"
    sinfo -h -o "%G" 2>/dev/null | tr ',' '\n' | grep -i gpu | sort -u | sed 's/^/     /'
    note "Bisher benutzt: gpu:l40s:1 . Falls l40s nicht mehr existiert, in ALLEN"
    note "Skripten gleichzeitig aendern - beide Seiten muessen dieselbe Klasse haben."
else
    bad "sinfo nicht verfuegbar - laeuft dieses Skript wirklich auf ARC?"
fi

echo
echo "8. Eigene Jobs (nur Anzeige, es wird nichts veraendert)"
squeue -u "$USER" -o "%.12i %.10P %.28j %.8T %.10M %.12l %R" 2>/dev/null || note "squeue nicht verfuegbar"

echo
echo "=============================================================="
if [ ${#PROBLEMS[@]} -eq 0 ]; then
    echo "PREFLIGHT: READY — weiter mit arc/01_inventory_existing_runs.sh"
else
    echo "PREFLIGHT: ${#PROBLEMS[@]} BLOCKER"
    for p in "${PROBLEMS[@]}"; do echo "  - $p"; done
    echo
    echo "Haeufigster Fall: SigmaFlow_Minimal / EXP-100 fehlen, weil die Commits"
    echo "noch nicht gepusht bzw. hier nicht gepullt wurden. Siehe Runbook Schritt 1."
fi
echo "=============================================================="
