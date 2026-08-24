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
     --job-name="${MODEL}${TAG}_s${SEED}"
     --export="ALL,MODE=${MODE},MODEL=${MODEL},FINAL_SEED=${SEED},FINAL_BATCH_SIZE=${FINAL_BATCH_SIZE},FINAL_ACCUM=${FINAL_ACCUM},FINAL_PRECISION=${FINAL_PRECISION},FINAL_CUDA_PRECISION=${FINAL_CUDA_PRECISION},FINAL_MAX_EPOCHS=${FINAL_MAX_EPOCHS},FINAL_MAX_STEPS=${FINAL_MAX_STEPS},FINAL_N_TRAIN=${FINAL_N_TRAIN},RESUME=${RESUME:-}"
     "${HERE}/train_final_72h.slurm")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

# Letzte Absicherung vor 72 GPU-Stunden.
if [ "${FINAL_EFFECTIVE_BATCH}" -ne 32 ]; then
    echo "[submit] WARNUNG: effektive Batch ist ${FINAL_EFFECTIVE_BATCH}, nicht 32."
    echo "         Alle Varianten muessen denselben Wert benutzen, sonst ist der"
    echo "         Vergleich nicht mehr kontrolliert. Weiter? [y/N]"
    read -r ans; [ "$ans" = "y" ] || { echo "abgebrochen"; exit 1; }
fi

"${CMD[@]}"
