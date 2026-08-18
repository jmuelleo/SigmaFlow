#!/bin/bash
#
# FINALER PREFLIGHT FUER DIE 72h-LAEUFE
#
# Ein Befehl, der alles prueft, was sich automatisch pruefen laesst, und der
# nur drei Antworten kennt: GREEN, AMBER, RED.
#
#   bash arc/final_72h_preflight.sh sigmaflow_minimal
#   bash arc/final_72h_preflight.sh sigmadock
#
# TRENNUNG DER ZUSTAENDIGKEITEN
#   Checks zerfallen in zwei Klassen. LOKALE Checks brauchen nur das
#   Repository und laufen ueberall. RUNTIME-Checks brauchen ARC (sinfo,
#   Dateisystem, gemessener Durchsatz). Ist ARC nicht erreichbar, meldet
#   dieses Skript NICHT faelschlich GREEN, sondern:
#
#       LOCAL READINESS      : GREEN
#       ARC RUNTIME READINESS: PENDING
#
#   Erst wenn beide GREEN sind, darf submittiert werden.
#
# Exit: 0 = GREEN, 1 = AMBER, 2 = RED

set -uo pipefail

MODEL="${1:?Usage: final_72h_preflight.sh <sigmadock|sigmaflow_minimal|sigmaflow_source|sigmaflow_conf>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

LOCAL_FAIL=0; LOCAL_WARN=0
RT_FAIL=0;    RT_WARN=0;   RT_PENDING=0
SECTION="LOCAL"

ok()      { echo "  [ OK   ] $*"; }
warn()    { echo "  [ WARN ] $*"; if [ "$SECTION" = "LOCAL" ]; then LOCAL_WARN=$((LOCAL_WARN+1)); else RT_WARN=$((RT_WARN+1)); fi; }
fail()    { echo "  [ FAIL ] $*"; if [ "$SECTION" = "LOCAL" ]; then LOCAL_FAIL=$((LOCAL_FAIL+1)); else RT_FAIL=$((RT_FAIL+1)); fi; }
pending() { echo "  [PENDING] $*"; RT_PENDING=$((RT_PENDING+1)); }
head2()   { echo; echo "-- $1 ------------------------------------------------------------"; }

echo "================================================================================"
echo "FINALER 72h-PREFLIGHT   Arm: ${MODEL}"
echo "Repository: ${REPO}"
echo "================================================================================"

# =============================================================================
# TEIL A -- LOKALE CHECKS
# =============================================================================
SECTION="LOCAL"

head2 "A1  Trainingshorizont"
# shellcheck source=/dev/null
if source "${HERE}/final_config.sh" 2>/dev/null; then
    ok "final_config.sh laedt"
else
    fail "final_config.sh laesst sich nicht sourcen"
fi

if [ "${FINAL_HORIZON_SOURCE:-FEHLT}" = "FEHLT" ]; then
    fail "arc/final_horizon.env fehlt -- Horizont ist NICHT gemessen"
    echo "         -> Stufe-2-Durchsatz messen, dann arc/calculate_final_epochs.py --write-env"
else
    ok "Horizontdatei vorhanden: ${FINAL_HORIZON_SOURCE}"
    if resolve_horizon_for_model "$MODEL" 2>/dev/null; then
        ok "Horizont fuer '${MODEL}': max_epochs=${FINAL_MAX_EPOCHS}, max_steps=${FINAL_MAX_STEPS}"
        ok "gemessene Datafront-Groesse n_train=${FINAL_N_TRAIN}"
        [ -n "${FINAL_HORIZON_POLICY:-}" ] && ok "Epochenpolitik: ${FINAL_HORIZON_POLICY}"
        [ -n "${FINAL_HORIZON_THROUGHPUT_SOURCE:-}" ] && \
            ok "Durchsatzquelle: ${FINAL_HORIZON_THROUGHPUT_SOURCE}"
    else
        fail "resolve_horizon_for_model '${MODEL}' fehlgeschlagen (siehe Meldung oben)"
    fi
fi

head2 "A2  Batch- und Akkumulationskonsistenz"
EB=$(( FINAL_BATCH_SIZE * FINAL_ACCUM ))
if [ "$EB" -eq "${FINAL_EFFECTIVE_BATCH}" ]; then
    ok "effektive Batch ${FINAL_EFFECTIVE_BATCH} = ${FINAL_BATCH_SIZE} x ${FINAL_ACCUM}"
else
    fail "FINAL_EFFECTIVE_BATCH=${FINAL_EFFECTIVE_BATCH} passt nicht zu ${FINAL_BATCH_SIZE}x${FINAL_ACCUM}"
fi
if [ "$EB" -ne 32 ]; then
    warn "effektive Batch ${EB} weicht vom Paper-Wert 32 ab -- nur bewusst zulassen"
else
    ok "effektive Batch entspricht dem Paper (32)"
fi

head2 "A3  Expliziter Scheduler-Horizont im Jobskript"
JOB="${HERE}/train_final_72h.slurm"
if grep -q -- '--max_steps "\${MAX_STEPS}"' "$JOB"; then
    ok "train.py bekommt --max_steps explizit (umgeht den accum-Bug in train.py:221)"
else
    fail "--max_steps wird NICHT explizit uebergeben -- der LR-Horizont waere bei accum>1 falsch"
fi
if grep -q 'MICROBATCHES_PER_EPOCH=' "$JOB"; then
    ok "val_check_interval wird in Microbatches gerechnet (richtige Lightning-Einheit)"
else
    fail "val_check_interval wird nicht in Microbatches gerechnet"
fi
# Nur RECHNENDE Zeilen pruefen. In Kommentaren und in Fehlermeldungen darf die
# Paper-Zahl vorkommen -- dort erklaert sie ja gerade, warum sie NICHT mehr
# gerechnet wird. Gesucht ist ausschliesslich ihre Verwendung als Wert.
if grep -vE '^[[:space:]]*#' "$JOB" | grep -vE '^[[:space:]]*echo' | grep -q '19443'; then
    fail "die Paper-Zahl 19443 steht noch in der Rechenlogik des Jobskripts"
else
    ok "keine hartkodierte Paper-Zahl 19443 in der Rechenlogik (Kommentare ausgenommen)"
fi
if grep -vE '^\s*#' "${HERE}/final_config.sh" | grep -q '19443'; then
    fail "19443 steht noch in der Rechenlogik von final_config.sh"
else
    ok "keine hartkodierte Paper-Zahl 19443 in final_config.sh"
fi

head2 "A4  Rechnerische Konsistenz des Horizonts"
if [ -n "${FINAL_N_TRAIN:-}" ] && [ -n "${FINAL_MAX_EPOCHS:-}" ] && [ -n "${FINAL_MAX_STEPS:-}" ]; then
    SPE=$(( FINAL_N_TRAIN / FINAL_EFFECTIVE_BATCH ))
    EXP=$(( SPE * FINAL_MAX_EPOCHS ))
    MB=$(( (FINAL_N_TRAIN + FINAL_BATCH_SIZE - 1) / FINAL_BATCH_SIZE ))
    ACT=$(( (MB + FINAL_ACCUM - 1) / FINAL_ACCUM ))
    ACT_TOTAL=$(( ACT * FINAL_MAX_EPOCHS ))
    if [ "$FINAL_MAX_STEPS" -eq "$EXP" ]; then
        ok "max_steps=${FINAL_MAX_STEPS} = floor(${FINAL_N_TRAIN}/${FINAL_EFFECTIVE_BATCH}) x ${FINAL_MAX_EPOCHS}"
    else
        fail "max_steps=${FINAL_MAX_STEPS}, erwartet ${EXP}"
    fi
    if [ "$ACT_TOTAL" -ge "$FINAL_MAX_STEPS" ]; then
        ok "tatsaechliche Steps ${ACT_TOTAL} >= Horizont ${FINAL_MAX_STEPS} -> Anneal laeuft durch"
    else
        fail "tatsaechliche Steps ${ACT_TOTAL} < Horizont ${FINAL_MAX_STEPS} -> Anneal UNVOLLSTAENDIG"
    fi
else
    warn "Horizont unvollstaendig -- Konsistenzrechnung uebersprungen"
fi

head2 "A5  Fehlerfortpflanzung des Trainingsprozesses"
if grep -qE '^\s*2>&1 \| tee' "$JOB"; then
    fail "Trainingsaufruf benutzt noch '| tee' -- Exit-Code kann verloren gehen"
else
    ok "kein '| tee' im Trainingsaufruf; TRAIN_PID ist eindeutig der Python-Prozess"
fi
grep -q 'TRAIN_STATUS=' "$JOB" && ok "Ergebnis wird klassifiziert (COMPLETED/CRASHED/WALLTIME/KILLED)" \
                               || fail "TRAIN_STATUS-Klassifikation fehlt"

head2 "A6  Automatisierte Tests"
run_test() {
    local label="$1"; shift
    if "$@" > /tmp/pf_test.$$ 2>&1; then
        ok "${label}: bestanden"
    else
        fail "${label}: FEHLGESCHLAGEN"
        tail -5 /tmp/pf_test.$$ | sed 's/^/         /'
    fi
    rm -f /tmp/pf_test.$$
}
PY_BIN="$(command -v python3 || command -v python)"
if [ -n "$PY_BIN" ]; then
    run_test "Scheduler-Horizont-Tests" "$PY_BIN" "${HERE}/test_scheduler_horizon.py"
    run_test "Konfigurationsvergleich"  "$PY_BIN" "${HERE}/compare_final_configs.py" \
        --sigmaflow-config "${REPO}/SigmaFlow_Minimal/src/sigmadock/config.py" \
        --sigmadock-config "${REPO}/SigmaDock/src_sigmadock/config.py" --fail-on-confounder
    if [ -f "${REPO}/audits/test_rotation_gradient_reachability.py" ]; then
        run_test "Gradient-Erreichbarkeit" "$PY_BIN" "${REPO}/audits/test_rotation_gradient_reachability.py"
    else
        warn "audits/test_rotation_gradient_reachability.py nicht gefunden"
    fi
else
    warn "kein python gefunden -- Tests uebersprungen"
fi
run_test "Exit-Code-Fortpflanzung" bash "${HERE}/test_train_rc.sh"

head2 "A7  EMA-Politik"
SF_CFG="${REPO}/SigmaFlow_Minimal/src/sigmadock/config.py"
SD_CFG="${REPO}/SigmaDock/src_sigmadock/config.py"
if [ -f "$SF_CFG" ] && [ -f "$SD_CFG" ]; then
    for key in "use_ema" "ema_rampup_ratio" "ema_halflife"; do
        A="$(grep -E "^    ${key}:" "$SF_CFG" | head -1 | sed 's/.*= *//; s/ *#.*//')"
        B="$(grep -E "^    ${key}:" "$SD_CFG" | head -1 | sed 's/.*= *//; s/ *#.*//')"
        if [ -n "$A" ] && [ "$A" = "$B" ]; then
            ok "${key} in beiden Armen gleich: ${A}"
        else
            fail "${key} weicht ab: SF='${A}' SD='${B}'"
        fi
    done
else
    warn "config.py eines Arms nicht gefunden -- EMA-Vergleich uebersprungen"
fi

head2 "A8  Snapshot-Zeitplan"
MAXH=0
for H in ${FINAL_SNAPSHOT_HOURS}; do [ "$H" -gt "$MAXH" ] && MAXH="$H"; done
WT_H="${FINAL_WALLTIME%%:*}"
if [ "$MAXH" -le "$WT_H" ]; then
    ok "letzter Snapshot bei ${MAXH} h liegt innerhalb der Walltime ${FINAL_WALLTIME}"
else
    fail "letzter Snapshot bei ${MAXH} h liegt HINTER der Walltime ${FINAL_WALLTIME}"
fi
if grep -q 'SCHED_TAG=' "$JOB"; then
    ok "Snapshots tragen den Schedule im Namen (nicht mit annealten Endpunkten verwechselbar)"
else
    fail "Snapshot-Benennung unterscheidet nicht zwischen Zwischenstand und Endpunkt"
fi

head2 "A9  Code-Provenienz"
if command -v git >/dev/null 2>&1 && [ -d "${REPO}/.git" ]; then
    COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
    ok "git commit: ${COMMIT}"
    DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$DIRTY" = "0" ]; then
        ok "Arbeitsverzeichnis sauber"
    else
        warn "${DIRTY} ungetrackte/geaenderte Dateien -- Zustand muss dokumentiert sein"
    fi
else
    warn "kein git-Repository gefunden"
fi

# =============================================================================
# TEIL B -- ARC-RUNTIME-CHECKS
# =============================================================================
SECTION="RUNTIME"
echo
echo "================================================================================"
echo "TEIL B -- ARC-RUNTIME"
echo "================================================================================"

ON_ARC=0
command -v sinfo >/dev/null 2>&1 && ON_ARC=1

head2 "B1  Partition, Walltime, GPU-Klasse"
if [ "$ON_ARC" = "0" ]; then
    pending "sinfo nicht verfuegbar -- Partition '${FINAL_PARTITION}' ist NICHT geprueft"
    pending "Walltime ${FINAL_WALLTIME} gegen das Partitionslimit NICHT geprueft"
    pending "GPU-Klasse ${FINAL_GPU} NICHT geprueft"
else
    if sinfo -h -o "%P" 2>/dev/null | sed 's/\*$//' | grep -qx "${FINAL_PARTITION}"; then
        ok "Partition '${FINAL_PARTITION}' existiert"
        LIM="$(sinfo -h -p "${FINAL_PARTITION}" -o "%l" 2>/dev/null | head -1 | tr -d ' ')"
        ok "Zeitlimit: ${LIM} (angefordert ${FINAL_WALLTIME})"
        GPU_CLASS="$(echo "${FINAL_GPU}" | cut -d: -f2)"
        if sinfo -h -p "${FINAL_PARTITION}" -o "%G" 2>/dev/null | grep -qi "${GPU_CLASS}"; then
            ok "GPU-Klasse '${GPU_CLASS}' auf '${FINAL_PARTITION}' vorhanden"
        else
            fail "GPU-Klasse '${GPU_CLASS}' auf '${FINAL_PARTITION}' NICHT vorhanden"
        fi
    else
        fail "Partition '${FINAL_PARTITION}' existiert nicht"
    fi
fi

head2 "B2  Verzeichnisse und Daten"
for d in "${ARC_DATA:-}" "${ARC_RUNS:-}"; do
    [ -z "$d" ] && continue
    if [ -d "$d" ]; then ok "vorhanden: $d"; else
        if [ "$ON_ARC" = "1" ]; then fail "fehlt: $d"; else pending "nicht pruefbar ausserhalb ARC: $d"; fi
    fi
done
case "$MODEL" in
    sigmadock)         CD="${ARC_SIGMADOCK:-}" ;;
    sigmaflow_minimal) CD="${ARC_REPO:-}/SigmaFlow_Minimal" ;;
    *)                 CD="" ;;
esac
if [ -n "$CD" ] && [ -d "$CD" ]; then ok "Codeverzeichnis: $CD"
elif [ "$ON_ARC" = "1" ]; then fail "Codeverzeichnis fehlt: $CD"
else pending "Codeverzeichnis nicht pruefbar ausserhalb ARC: $CD"; fi

head2 "B3  Durchsatzmessung passt zur Konfiguration"
if [ "${FINAL_HORIZON_SOURCE:-FEHLT}" = "FEHLT" ]; then
    pending "kein Stufe-2-Ergebnis -- Durchsatz noch nicht gemessen"
elif [ -n "${FINAL_HORIZON_EFFECTIVE_BATCH:-}" ]; then
    if [ "${FINAL_HORIZON_EFFECTIVE_BATCH}" -eq "${FINAL_EFFECTIVE_BATCH}" ]; then
        ok "Messung galt fuer effektive Batch ${FINAL_HORIZON_EFFECTIVE_BATCH} = aktuelle Konfiguration"
    else
        fail "Messung galt fuer Batch ${FINAL_HORIZON_EFFECTIVE_BATCH}, konfiguriert ist ${FINAL_EFFECTIVE_BATCH}"
    fi
fi

head2 "B4  Speicherplatz"
if command -v df >/dev/null 2>&1 && [ -n "${ARC_RUNS:-}" ] && [ -d "${ARC_RUNS}" ]; then
    AVAIL_KB="$(df -Pk "${ARC_RUNS}" | awk 'NR==2{print $4}')"
    AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
    NSNAP="$(echo ${FINAL_SNAPSHOT_HOURS} | wc -w | tr -d ' ')"
    NEED_GB=$(( NSNAP * 2 ))     # grob: mehrere hundert MB je Checkpoint
    if [ "$AVAIL_GB" -gt "$NEED_GB" ]; then
        ok "${AVAIL_GB} GB frei, geschaetzter Bedarf ~${NEED_GB} GB fuer ${NSNAP} Snapshots"
    else
        fail "nur ${AVAIL_GB} GB frei, geschaetzter Bedarf ~${NEED_GB} GB"
    fi
else
    pending "Speicherplatz nicht pruefbar"
fi

# =============================================================================
echo
echo "================================================================================"
echo "ERGEBNIS"
echo "================================================================================"

if   [ "$LOCAL_FAIL" -gt 0 ]; then LOCAL_STATE="RED"
elif [ "$LOCAL_WARN" -gt 0 ]; then LOCAL_STATE="AMBER"
else                               LOCAL_STATE="GREEN"; fi

if   [ "$RT_FAIL" -gt 0 ];    then RT_STATE="RED"
elif [ "$RT_PENDING" -gt 0 ]; then RT_STATE="PENDING"
elif [ "$RT_WARN" -gt 0 ];    then RT_STATE="AMBER"
else                               RT_STATE="GREEN"; fi

echo "  LOCAL READINESS      : ${LOCAL_STATE}   (${LOCAL_FAIL} FAIL, ${LOCAL_WARN} WARN)"
echo "  ARC RUNTIME READINESS: ${RT_STATE}   (${RT_FAIL} FAIL, ${RT_WARN} WARN, ${RT_PENDING} PENDING)"
echo

if [ "$LOCAL_STATE" = "GREEN" ] && [ "$RT_STATE" = "GREEN" ]; then
    echo "  GREEN -- SAFE TO SUBMIT"
    echo "           bash arc/submit_final.sh ${MODEL}"
    exit 0
elif [ "$LOCAL_STATE" = "RED" ] || [ "$RT_STATE" = "RED" ]; then
    echo "  RED -- NICHT SUBMITTIEREN. Die FAIL-Zeilen oben zuerst schliessen."
    exit 2
else
    echo "  AMBER -- technisch startbar, aber mindestens eine Gueltigkeitsbedingung"
    echo "           ist nicht bewiesen. Offene Punkte stehen oben als WARN/PENDING."
    exit 1
fi
