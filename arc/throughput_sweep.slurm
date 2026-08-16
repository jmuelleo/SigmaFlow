#!/bin/bash -l
#
# JOB 1 — THROUGHPUT / BATCH SWEEP
#
# ZWECK
#   Die Konfiguration ALLER langen Laeufe dieser Woche haengt an Zahlen, die
#   im Repository nirgends existieren: samples/s, Peak-VRAM und OOM-Grenze in
#   Abhaengigkeit von Batch-Size, Praezision und Debug-Modus. Bisher wurde
#   Batch 16/24/32 NIE getestet; die einzige bekannte Zahl ist 31.700 Schritte
#   in 24 h bei Batch 8 (Job 8465054/8465055).
#
#   Dieser Job misst sie, bevor 2 x 72 GPU-Stunden vergeben werden.
#
# DREI HYPOTHESEN, DIE HIER GEPRUEFT WERDEN
#   H-A  `--debug` kostet erheblichen Durchsatz. Es aktiviert
#        `detect_anomaly=True`, `profiler="simple"` und FullNaNCheckCallback
#        (Forward-Hooks auf JEDEM Modul). Beide 24h-Jobs setzen es.
#   H-B  `--cuda_precision highest` schaltet TF32 ab (train.py:72). Auf L40S
#        und A100 ist das potenziell der groesste einzelne Hebel.
#   H-C  `--val_check_interval 50` validiert bei ~2.430 Schritten/Epoche rund
#        49x pro Epoche auf dem vollen PoseBusters-Satz, ohne limit_val_batches.
#
#   Jede dieser Hypothesen kann sich als unwichtig erweisen. Genau deshalb
#   wird gemessen und nicht geschaetzt.
#
# DESIGN
#   Der Datafront-Aufbau dauert auf pdbbind-general 10-35 min. Bei 12 Konfigs
#   waere das die gesamte Laufzeit. Deshalb zweistufig:
#     Stufe 1  pdbbind-core (285 Komplexe, schneller Aufbau) -> RELATIVE
#              Faktoren: Batch-Skalierung, Debug, TF32, bf16, VRAM, OOM.
#     Stufe 2  pdbbind-general, EINE Konfiguration (die aus Stufe 1 gewaehlte)
#              -> ABSOLUTER Anker in samples/s auf der echten Verteilung.
#   Stufe 2 laeuft NICHT automatisch; sie wird nach Sichtung von Stufe 1
#   gestartet (STAGE=2 CONFIG=<name>).
#
# Usage:
#   MODEL=sigmaflow_minimal sbatch arc/throughput_sweep.slurm
#   MODEL=sigmadock         sbatch arc/throughput_sweep.slurm
#   MODEL=sigmaflow_minimal STAGE=2 BATCH=32 EXTRA="--cuda_precision high" \
#       sbatch arc/throughput_sweep.slurm
#
#SBATCH --job-name=throughput
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail   # bewusst KEIN -e: eine OOM-Konfiguration soll den Sweep
                   # nicht abbrechen, sondern als "OOM" protokolliert werden.

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
setup_wandb_hardening

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | sigmadock}"
STAGE="${STAGE:-1}"
STEPS="${STEPS:-120}"          # gemessen werden die letzten STEPS-WARMUP Schritte
WARMUP_STEPS="${WARMUP_STEPS:-20}"

case "$MODEL" in
  sigmaflow_minimal)
      CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      METHOD_FLAGS=()
      ;;
  sigmadock)
      CODE_DIR="${ARC_SIGMADOCK}"
      module load Mamba
      source activate "$ARC_SD_ENV"
      PYTHON="$(which python)"
      METHOD_FLAGS=(--rot_score_method space --rot_score_scaling rms)
      ;;
  *) echo "MODEL unbekannt: $MODEL"; exit 1 ;;
esac

write_run_metadata "THROUGHPUT-${MODEL}" "$CODE_DIR" "$PYTHON"
cd "$CODE_DIR" || exit 1

CSV="${RUN_DIR}/throughput_${MODEL}.csv"
echo "model,stage,config,dataset,batch,precision,cuda_precision,debug,val_interval,status,wall_s,steps,steps_per_s,samples_per_s,peak_vram_mib" > "$CSV"

# ---------------------------------------------------------------- VRAM-Poller
# nvidia-smi im Hintergrund; Maximum wird nach jedem Lauf ausgelesen.
VRAM_LOG="${RUN_DIR}/vram.log"
( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG" 2>/dev/null
      sleep 2
  done ) &
POLLER_PID=$!
trap 'kill $POLLER_PID 2>/dev/null' EXIT

run_one() {
    local cfg_name="$1" dataset="$2" batch="$3" val_int="$4"; shift 4
    local extra=("$@")

    : > "$VRAM_LOG"
    local t0 t1 status rc
    echo
    echo "──────── ${cfg_name}  (batch=${batch}, dataset=${dataset}) ────────"

    t0=$(date +%s)
    timeout 2400 "$PYTHON" scripts/train.py \
        --data_dir "${ARC_DATA}" \
        --train_exps "${dataset}" \
        --val_exps posebusters \
        --test_exps posebusters \
        --batch_size "${batch}" \
        --num_workers 6 \
        --accelerator gpu \
        --devices 1 \
        --max_steps "${STEPS}" \
        --val_check_interval "${val_int}" \
        --early_stopping_patience 0 \
        --trans_score_weight 2.0 \
        --rot_score_weight 0.5 \
        --seed 0 \
        --offline_run \
        "${METHOD_FLAGS[@]}" \
        "${extra[@]}" \
        > "${RUN_DIR}/${cfg_name}.log" 2>&1
    rc=$?
    t1=$(date +%s)

    local wall=$((t1 - t0))
    local peak
    peak="$(sort -n "$VRAM_LOG" 2>/dev/null | tail -1)"
    peak="${peak:-NA}"

    if [ $rc -eq 0 ]; then
        status="OK"
    elif grep -qiE "out of memory|CUDA error: out of memory" "${RUN_DIR}/${cfg_name}.log"; then
        status="OOM"
    elif [ $rc -eq 124 ]; then
        status="TIMEOUT"
    else
        status="FAIL_rc${rc}"
    fi

    # Nur bei OK sind Durchsatzzahlen sinnvoll. Der Datafront-Aufbau steckt in
    # wall_s mit drin -- deshalb ist NUR der Vergleich zwischen Konfigurationen
    # auf DEMSELBEN Datensatz aussagekraeftig, nicht der Absolutwert.
    local sps="NA" samp="NA"
    if [ "$status" = "OK" ] && [ "$wall" -gt 0 ]; then
        sps=$(awk -v s="$STEPS" -v w="$wall" 'BEGIN{printf "%.4f", s/w}')
        samp=$(awk -v s="$STEPS" -v b="$batch" -v w="$wall" 'BEGIN{printf "%.3f", s*b/w}')
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$MODEL" "$STAGE" "$cfg_name" "$dataset" "$batch" \
        "${PREC:-32}" "${CUDA_PREC:-highest}" "${DBG:-yes}" "$val_int" \
        "$status" "$wall" "$STEPS" "$sps" "$samp" "$peak" >> "$CSV"

    echo "  -> ${status}  wall=${wall}s  peak_vram=${peak} MiB"
}

# ============================================================================
if [ "$STAGE" = "1" ]; then
# ============================================================================
# Stufe 1 — relative Faktoren auf pdbbind-core.
# Reihenfolge bewusst: erst die Referenz, dann je EIN Hebel, dann kombiniert.
# So ist jede Differenz genau einer Aenderung zuzuordnen.

DS=pdbbind-core

# (1) Referenz: exakt die Konfiguration der aktuellen 24h-Jobs.
PREC=32 CUDA_PREC=highest DBG=yes \
  run_one "ref_b8_debug"            "$DS"  8   50    --debug

# (2) H-A: Debug aus.
PREC=32 CUDA_PREC=highest DBG=no \
  run_one "b8_nodebug"              "$DS"  8   50

# (3) H-C: Validation seltener.
PREC=32 CUDA_PREC=highest DBG=no \
  run_one "b8_nodebug_val500"       "$DS"  8   500

# (4) H-B: TF32 an.
PREC=32 CUDA_PREC=high DBG=no \
  run_one "b8_nodebug_tf32"         "$DS"  8   500  --cuda_precision high

# (5) bf16 gemischt, zusaetzlich.
PREC=bf16 CUDA_PREC=high DBG=no \
  run_one "b8_nodebug_tf32_bf16"    "$DS"  8   500  --cuda_precision high --precision bf16-mixed

# (6)-(8) Batch-Skalierung auf der besten stabilen Basis (TF32, kein Debug, fp32).
for B in 16 24 32; do
  PREC=32 CUDA_PREC=high DBG=no \
    run_one "b${B}_nodebug_tf32"    "$DS"  "$B" 500  --cuda_precision high
done

# (9)-(10) Batch-Skalierung mit bf16 — hier ist die OOM-Grenze am weitesten oben.
for B in 32 48; do
  PREC=bf16 CUDA_PREC=high DBG=no \
    run_one "b${B}_nodebug_tf32_bf16" "$DS" "$B" 500 --cuda_precision high --precision bf16-mixed
done

# ============================================================================
else
# ============================================================================
# Stufe 2 — absoluter Anker auf pdbbind-general, EINE Konfiguration.
BATCH="${BATCH:?BATCH setzen}"
EXTRA_STR="${EXTRA:-}"
read -r -a EXTRA_ARR <<< "$EXTRA_STR"
PREC="${PREC:-32}" CUDA_PREC="${CUDA_PREC:-high}" DBG=no \
  run_one "anchor_general_b${BATCH}" pdbbind-general "$BATCH" 500 "${EXTRA_ARR[@]}"
fi

kill $POLLER_PID 2>/dev/null

echo
echo "=============================================================="
echo "Ergebnis: ${CSV}"
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"
echo "=============================================================="
echo
echo "LESART"
echo "  * wall_s enthaelt den Datafront-Aufbau. Nur Konfigurationen auf"
echo "    DEMSELBEN Datensatz sind direkt vergleichbar."
echo "  * peak_vram ist die GPU-weite Belegung inkl. Allokator-Cache,"
echo "    nicht torch.max_memory_allocated. Als OOM-Abstandsmass brauchbar,"
echo "    nicht als exakter Modellbedarf."
echo "  * Fuer die Epochenrechnung der langen Laeufe zaehlt Stufe 2."
