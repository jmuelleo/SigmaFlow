#!/bin/bash -l
#
# Submit-Wrapper fuer die finalen Laeufe.
#
# WARUM ES DEN WRAPPER GIBT
#   #SBATCH-Zeilen werden von SLURM gelesen, bevor irgendein Shellcode laeuft;
#   sie koennen daher keine Variablen aus final_config.sh sehen. Partition,
#   Walltime und GPU-Klasse muessen deshalb als sbatch-Argumente kommen.
#   Dieser Wrapper liest final_config.sh und setzt sie -- damit bleibt
#   final_config.sh die einzige Stelle, an der Werte gepflegt werden.
#
# Usage:
#   bash arc/submit_final.sh sigmadock
#   bash arc/submit_final.sh sigmaflow_minimal
#   bash arc/submit_final.sh sigmaflow_minimal 1        # zweiter Seed
#   MODE=screen bash arc/submit_final.sh sigmaflow_source   # 6h-Screening
#   DRY_RUN=1 bash arc/submit_final.sh sigmadock            # nur anzeigen
#
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/final_config.sh"

MODEL="${1:?Usage: submit_final.sh <sigmadock|sigmaflow_minimal|sigmaflow_twohead|sigmaflow_source|sigmaflow_conf> [seed]}"
SEED="${2:-${FINAL_SEED}}"

# MODE=screen -> 6h-Screening auf der KURZEN Partition, aber mit dem
# unveraenderten langen LR-Schedule (siehe final_config.sh).
MODE="${MODE:-full}"
case "$MODE" in
  full)   USE_PART="${FINAL_PARTITION}"; USE_TIME="${FINAL_WALLTIME}"; TAG="" ;;
  screen) USE_PART="${SCREEN_PARTITION}"; USE_TIME="${SCREEN_WALLTIME}"; TAG="_scr" ;;
  *) echo "MODE unbekannt: $MODE (full|screen)"; exit 1 ;;
esac

# Horizont fuer DIESEN Arm aufloesen. Schlaegt fehl, wenn final_horizon.env
# fehlt, der Arm dort nicht steht, oder die gespeicherte effektive Batch nicht
# mehr zur aktuellen Konfiguration passt. Ohne gueltigen Horizont wird nicht
# submittiert -- ein geratener Wert kostet 72 GPU-Stunden.
if ! resolve_horizon_for_model "$MODEL"; then
    echo "[submit] ABBRUCH: kein gueltiger Trainingshorizont fuer '${MODEL}'." >&2
    exit 1
fi

print_final_config
echo "[submit] MODEL=${MODEL}  SEED=${SEED}  MODE=${MODE}"
echo "[submit] Partition=${USE_PART}  Walltime=${USE_TIME}"
if [ "$MODE" = "screen" ]; then
    echo "[submit] Screening: max_epochs bleibt ${FINAL_MAX_EPOCHS}, damit die ersten"
    echo "[submit]            6 h DERSELBEN LR-Trajektorie gemessen werden wie im langen Lauf."
fi

CMD=(sbatch
     --partition="${USE_PART}"
     --time="${USE_TIME}"
     --gres="${FINAL_GPU}"
     --cpus-per-task="${FINAL_CPUS}"
     --mem="${FINAL_MEM}"
     # RUN_ID_EXTRA gehoert in den Jobnamen, nicht nur in die RUN_ID.
     # Ohne das heissen zwei Laeufe DESSELBEN Arms in squeue gleich -- und
     # genau diese Konstellation (zwei parallele Laeufe eines Arms) hat schon
     # einmal dazu gefuehrt, dass ein Snapshot fremde Gewichte trug.
     --job-name="${MODEL}${TAG}_s${SEED}${RUN_ID_EXTRA:-}"
     --export="ALL,MODE=${MODE},MODEL=${MODEL},FINAL_SEED=${SEED},FINAL_BATCH_SIZE=${FINAL_BATCH_SIZE},FINAL_ACCUM=${FINAL_ACCUM},FINAL_PRECISION=${FINAL_PRECISION},FINAL_CUDA_PRECISION=${FINAL_CUDA_PRECISION},FINAL_MAX_EPOCHS=${FINAL_MAX_EPOCHS},FINAL_MAX_STEPS=${FINAL_MAX_STEPS},FINAL_N_TRAIN=${FINAL_N_TRAIN},RESUME=${RESUME:-}"
     "${HERE}/train_final_72h.slurm")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

# Letzte Absicherung vor vielen GPU-Stunden.
#   64 ist der Wert der ausgelieferten SigmaDock-Konfiguration. Er ist GLOBAL:
#      config.py teilt batch_size durch world_size. Rezepttreue Laeufe stehen
#      hier.
#   32 war der Wert der drei walltime-gematchten Laeufe des Hauptvergleichs.
#   Alles andere ist erklaerungsbeduerftig und muss bestaetigt werden.
#
# Frueher stand hier `-ne 32` mit einem blanken `read`. Beides war falsch: der
# rezepttreue Lauf mit effektiv 64 lief in die Rueckfrage, und `read` blockiert
# ohne Terminal an stdin, bis die Session in die Zeitgrenze laeuft. Genau so
# ging eine Einreichung verloren. Jetzt bricht es ab, statt zu haengen.
case "${FINAL_EFFECTIVE_BATCH}" in
    32|64) ;;
    *)
        echo "[submit] WARNUNG: effektive Batch ist ${FINAL_EFFECTIVE_BATCH}."
        echo "         Erwartet: 64 (Originalrezept) oder 32 (gematchte Laeufe)."
        if [ "${CONFIRM_BATCH:-}" = "y" ]; then
            echo "         CONFIRM_BATCH=y gesetzt, weiter."
        elif [ -t 0 ]; then
            echo "         Weiter? [y/N]"
            read -r ans
            [ "$ans" = "y" ] || { echo "abgebrochen"; exit 1; }
        else
            echo "         Kein Terminal an stdin. Mit CONFIRM_BATCH=y bestaetigen." >&2
            exit 1
        fi
        ;;
esac

"${CMD[@]}"
