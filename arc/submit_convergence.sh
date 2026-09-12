#!/bin/bash -l
#
# KONVERGENZLAUF MIT ABWEICHENDER WALLTIME (Vorgabe: 120 h = 5 Tage)
#
# WAS DAS IST UND WAS NICHT
#   Das ist ein EIGENSTAENDIGES EXPERIMENT, kein Teil des Hauptvergleichs.
#   Der Hauptvergleich sind die drei 72-h-Laeufe (SD_BASE/SF_MIN/SF_2H, je
#   72:00 +- 3 min). Dieser Lauf bricht die gematchte Walltime bewusst und darf
#   dessen Zahlen NIEMALS ersetzen oder ergaenzen.
#
#   Die Frage, die er beantwortet: erreicht SigmaFlow-Separate mit fuenf Tagen
#   Training das Niveau des VEROEFFENTLICHTEN SigmaDock (Top-1, RMSD < 2 A UND
#   PB-valid, N_seeds = 40: 79,9 % laut Prat et al., Abbildung 4)?
#
# WARUM EIN FRISCHER LAUF UND KEIN RESUME
#   Der Cosine-Anneal ist auf max_steps kalibriert. Ein Resume des
#   222-Epochen-Checkpoints mit verlaengertem Horizont wuerde den LR-Verlauf
#   mittendrin veraendern -- das Ergebnis waere weder ein 72-h- noch ein
#   120-h-Lauf, sondern ein Zwitter. Der `resume-guard` in
#   train_final_72h.slurm blockiert das zu Recht. Ein sauberer 120-h-Lauf
#   annealt ueber die volle Strecke.
#
# WARUM EINE EIGENE HORIZONTDATEI
#   `arc/final_horizon.env` ist die Quelle des 72-h-Hauptexperiments und wird
#   NICHT angefasst. `calculate_final_epochs.py --write-env` respektiert den
#   uebergebenen Pfad, also entsteht daneben `final_horizon_<h>h.env`.
#   `final_config.sh` liest ueber die Umgebungsvariable HORIZON_FILE.
#
# DER DURCHSATZ IST JETZT GEMESSEN, NICHT ABGELEITET
#   Die 72-h-Horizontdatei musste fuer sigmaflow_twohead auf Minimals Wert von
#   21,1 Beispielen/s raten (die Sweeps liefen in die Zeitgrenze). Ergebnis:
#   255 Epochen geplant, 223 erreicht -- 14 % zu optimistisch.
#
#   Jetzt gemessen: 223 Epochen in 72,06 h Walltime. Rechnet man wie das
#   Original mit 8 h Aufbau- und Auslaufzeit, also 64 h reiner Optimierung:
#       223 * 19037 / (64 * 3600) = 18,43 Beispiele/s
#   Das ist konservativ, denn der 72-h-Lauf hatte DREI Segmente und damit
#   dreimal den Datenfront-Aufbau; ein einzelner Lauf sollte darueber liegen.
#   Konservativ ist hier die richtige Richtung: lieber den Anneal zu Ende
#   fahren als von der Walltime abgeschnitten werden.
#
# PLATZBEDARF
#   Snapshots alle 6 h ueber 120 h sind 20 Stueck a ~273 MB = rund 5,5 GB.
#   `/data/stat-cadd` war am 2026-08-28 knapp; vorher `df -h` ansehen.
#
# Aufruf:
#   bash arc/submit_convergence.sh                          # twohead, 120 h, Seed 0
#   STUNDEN=96 bash arc/submit_convergence.sh               # kuerzer
#   bash arc/submit_convergence.sh sigmaflow_minimal 0      # anderer Arm
#   DRY_RUN=1 bash arc/submit_convergence.sh                # nur planen
#
set -uo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-sigmaflow_twohead}"
SEED="${2:-0}"
STUNDEN="${STUNDEN:-120}"

# Gemessener Durchsatz je Arm. sigmaflow_twohead aus dem 72-h-Lauf 8668713
# (223 Epochen), sigmaflow_minimal aus 8653824 (245 Epochen), sigmadock aus
# 8648493 (236 Epochen) -- jeweils gegen 64 h reine Optimierung gerechnet.
case "$MODEL" in
  sigmaflow_twohead)  SAMPLES_PER_S="${SAMPLES_PER_S:-18.43}" ;;
  sigmaflow_minimal)  SAMPLES_PER_S="${SAMPLES_PER_S:-20.25}" ;;
  sigmadock)          SAMPLES_PER_S="${SAMPLES_PER_S:-19.51}" ;;
  *) echo "MODEL unbekannt: ${MODEL}" >&2; exit 1 ;;
esac

# Aufbau, Validierung und Auslauf, die nicht in die Optimierung gehen.
# 8 h ist der Wert, mit dem auch der 72-h-Horizont gerechnet wurde.
OVERHEAD_H="${OVERHEAD_H:-8}"
BUDGET_H="$(awk -v a="$STUNDEN" -v b="$OVERHEAD_H" 'BEGIN{printf "%.1f", a-b}')"

HZ="${HIER}/final_horizon_${STUNDEN}h.env"

echo "=============================================================="
echo "  KONVERGENZLAUF -- eigenstaendiges Experiment"
echo "  Arm            : ${MODEL}   Seed ${SEED}"
echo "  Walltime       : ${STUNDEN} h"
echo "  Optimierung    : ${BUDGET_H} h  (abzueglich ${OVERHEAD_H} h Aufbau/Auslauf)"
echo "  Durchsatz      : ${SAMPLES_PER_S} Beispiele/s (GEMESSEN aus den 72-h-Laeufen)"
echo "  Horizontdatei  : ${HZ}"
echo "  Zielmarke      : 79,9 % (Top-1, RMSD<2 & PB-valid, Prat et al. N=40)"
echo "=============================================================="

python "${HIER}/calculate_final_epochs.py" \
    --arm "$MODEL" \
    --samples-per-s "$SAMPLES_PER_S" \
    --n-train 19037 \
    --physical-batch 32 \
    --accum 1 \
    --world-size 1 \
    --budget-hours "$BUDGET_H" \
    --throughput-source "GEMESSEN aus den 72-h-Laeufen (8648493/8653824/8668713): erreichte Epochen gegen 64 h reine Optimierung. Fuer sigmaflow_twohead 223 Epochen -> 18.43 Beispiele/s; die 72-h-Datei hatte hier 21.1 geraten und lag 14 Prozent zu hoch." \
    --write-env "$HZ"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "ABBRUCH: Horizont konnte nicht berechnet werden (rc=${RC})." >&2
    exit "$RC"
fi

# Snapshots alle 6 h ueber die volle Strecke.
SNAPS="$(seq 6 6 "$STUNDEN" | tr '\n' ' ')"
N_SNAPS="$(echo "$SNAPS" | wc -w)"
GB="$(awk -v n="$N_SNAPS" 'BEGIN{printf "%.1f", n*0.273}')"
echo
echo "  Snapshots (h)  : ${SNAPS}"
echo "  das sind ${N_SNAPS} Stueck, rund ${GB} GB"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 -- nicht eingereicht. Der Horizont steht in ${HZ}."
    exit 0
fi

export HORIZON_FILE="$HZ"
export FINAL_WALLTIME="${STUNDEN}:00:00"
export FINAL_SNAPSHOT_HOURS="$SNAPS"
export RUN_ID_EXTRA="_${STUNDEN}H"

# submit_final.sh ist der EINZIGE zulaessige Weg. Ein direkter sbatch fordert
# keine GPU an, weil #SBATCH-Zeilen final_config.sh nicht lesen koennen.
bash "${HIER}/submit_final.sh" "$MODEL" "$SEED"
