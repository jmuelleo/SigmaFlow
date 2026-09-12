#!/bin/bash
#
# SCHRITTZAHL-SWEEP UEBER ALLE DREI ARME
#
# ZWECK
#   Systematisch messen, wie stark die Zahl der Integrationsschritte beim
#   aktuellen Trainingsstand ueberhaupt wirkt. Die 25 Schritte sind von
#   SigmaDock GEERBT, wo das Paper sie fuer einen Diffusions-Rueckwaertsprozess
#   optimiert hat. Fuer eine ODE muss das nicht gelten -- und fuer SigmaDock
#   selbst ist es nie nachgemessen worden.
#
#   Ein frueherer Sweep (Job 8554148, 2026-08-13) fand fuer SigmaFlow keine
#   nachweisbare Abhaengigkeit zwischen 5 und 200 Schritten, aber nur ueber
#   DREI Seeds und nur fuer EINEN Arm. Dieser hier deckt alle drei Arme mit
#   40 bzw. 80 Seeds ab.
#
# WAS ES NICHT NEU BAUT
#   arc/sample_pb_seeds_cpu.slurm kann das bereits: NUM_STEPS steuert die
#   Schrittzahl, und der Ausgabepfad traegt sie im Namen
#   (..._nfe<N>__cpu), sodass verschiedene Schrittzahlen sich nicht
#   ueberschreiben koennen. Dieses Skript ist nur der Submit-Wrapper, der
#
#     - die Walltime an die Schrittzahl anpasst (bei 200 Schritten reichen
#       die fest verdrahteten 1:30 h nicht),
#     - vorhandene Laeufe erkennt und ueberspringt statt sie zu ueberschreiben,
#     - vor dem Abschicken zusammenfasst, was das kostet.
#
#   Die Trajektorien fallen ohne Zutun an: scripts/sample.py legt sie je
#   Komplex als "trajectory" [T, n_atoms, 3] in predictions.pt ab. Rueck-
#   transformation in Angstrom: trajectory * HPARAMS.general.dimensional_scale
#   + com (der Faktor ist 2.7, numerisch gegen x0_hat geprueft).
#
# USAGE
#   CKPT_MIN=<pfad> CKPT_110=<pfad> CKPT_SD=<pfad> \
#     bash arc/submit_step_sweep.sh [von] [bis]
#
#   Vorgabe ist der Seedbereich 0-39. Die zweite Haelfte kommt mit
#     ... bash arc/submit_step_sweep.sh 40 79
#   und landet in denselben Verzeichnissen -- additiv, ohne Kollision.
#
#   DRY_RUN=1     nur anzeigen, nichts abschicken
#   STEPS="5 25"  eigene Schrittzahlen
#   MODELS="..."  eigene Armauswahl
set -euo pipefail

VON="${1:-0}"
BIS="${2:-39}"
STEPS="${STEPS:-1 3 5 10 50 100 200}"
MODELS="${MODELS:-sigmaflow_minimal exp110 sigmadock}"
COMPARISON="${COMPARISON:-controlled}"
RANKING="${RANKING:-raw}"
DRY_RUN="${DRY_RUN:-0}"

REPO="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
ARC_RUNS="/data/stat-cadd/shug8458/arc_runs"

# 25 fehlt in der Vorgabe mit Absicht: dieser Lauf existiert bereits mit 80
# Seeds und ist die Grundlage aller bisherigen Zahlen. Erneut zu sampeln
# wuerde ihn ueberschreiben.
case " $STEPS " in
    *" 25 "*) echo "WARNUNG: 25 Schritte sind in STEPS enthalten."
              echo "         Dieser Lauf existiert bereits mit 80 Seeds und wuerde"
              echo "         ueberschrieben. Nur fortfahren, wenn das gewollt ist."
              echo ;;
esac

ckpt_fuer() {
    case "$1" in
        sigmaflow_minimal) echo "${CKPT_MIN:?CKPT_MIN setzen}" ;;
        exp110)            echo "${CKPT_110:?CKPT_110 setzen}" ;;
        sigmadock)         echo "${CKPT_SD:?CKPT_SD setzen}" ;;
        *) echo "unbekanntes MODEL: $1" >&2; return 1 ;;
    esac
}

# Walltime waechst mit der Schrittzahl. Grundlage: bei 25 Schritten lagen die
# Tasks bei 20-40 min. Grundlast fuer Datenaufbau und Modellladen plus ein
# Anteil je Schritt, grosszuegig gerundet. Zu kurz gewaehlt kostet den ganzen
# Task, zu lang kostet nur Platz in der Warteschlange.
walltime_fuer() {
    local s="$1" min
    min=$(( 45 + (s * 3 + 1) / 2 ))      # 45 min Grundlast + 1.5 min je Schritt
    [ "$min" -lt 60 ] && min=60
    [ "$min" -gt 690 ] && min=690        # short-Partition endet bei 12 h
    printf '%02d:%02d:00' $(( min / 60 )) $(( min % 60 ))
}

N_SEEDS=$(( BIS - VON + 1 ))
echo "=============================================================="
echo " SCHRITTZAHL-SWEEP"
echo " Seeds       : ${VON}-${BIS}  (${N_SEEDS} je Kombination)"
echo " Schrittzahl : ${STEPS}"
echo " Arme        : ${MODELS}"
echo " Modus       : ${COMPARISON} / ${RANKING}"
[ "$DRY_RUN" = "1" ] && echo " DRY_RUN     : nichts wird abgeschickt"
echo "=============================================================="
echo

gesamt_tasks=0
gesamt_min=0
uebersprungen=0

for M in $MODELS; do
    CK="$(ckpt_fuer "$M")"
    if [ ! -f "$CK" ]; then
        echo "ABBRUCH: Checkpoint fehlt fuer ${M}: ${CK}"
        exit 1
    fi
    for S in $STEPS; do
        OUT="${ARC_RUNS}/sampling/${M}__${COMPARISON}__${RANKING}__nfe${S}__cpu"
        WT="$(walltime_fuer "$S")"

        # Kollision: existieren Seeds aus dem angeforderten Bereich schon?
        vorhanden=0
        if [ -d "$OUT" ]; then
            for s in $(seq "$VON" "$BIS"); do
                if find "$OUT" -type d -name "seed_${s}" -print -quit 2>/dev/null | grep -q .; then
                    vorhanden=$(( vorhanden + 1 ))
                fi
            done
        fi
        if [ "$vorhanden" -gt 0 ]; then
            printf '  %-18s nfe %-4s  UEBERSPRUNGEN -- %d Seeds im Bereich existieren bereits\n' \
                   "$M" "$S" "$vorhanden"
            uebersprungen=$(( uebersprungen + 1 ))
            continue
        fi

        printf '  %-18s nfe %-4s  array=%s-%s  walltime=%s\n' "$M" "$S" "$VON" "$BIS" "$WT"
        gesamt_tasks=$(( gesamt_tasks + N_SEEDS ))
        gesamt_min=$(( gesamt_min + N_SEEDS * (45 + (S * 3 + 1) / 2) ))

        if [ "$DRY_RUN" != "1" ]; then
            ( cd "$REPO" && \
              CKPT="$CK" MODEL="$M" NUM_STEPS="$S" \
              COMPARISON="$COMPARISON" RANKING="$RANKING" \
              sbatch --time="$WT" --array="${VON}-${BIS}" \
                     --job-name="sw-${M%%_*}-${S}" \
                     arc/sample_pb_seeds_cpu.slurm )
        fi
    done
done

echo
echo "=============================================================="
echo " Tasks gesamt          : ${gesamt_tasks}"
echo " Reservierte CPU-Stunden (Obergrenze, nicht Verbrauch): $(( gesamt_min / 60 ))"
echo " Uebersprungene Kombinationen: ${uebersprungen}"
echo "=============================================================="
if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "Nichts abgeschickt. Ohne DRY_RUN=1 erneut aufrufen."
fi
