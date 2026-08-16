#!/bin/bash -l
#
# FINALE 72h-TRAININGSLAEUFE — ein Skript fuer alle Varianten.
#
# WARUM EIN SKRIPT UND NICHT VIER
#   Der wissenschaftliche Wert des Vergleichs haengt daran, dass SigmaDock und
#   SigmaFlow exakt dieselbe Trainingskonfiguration bekommen. Vier getrennte
#   Skripte divergieren erfahrungsgemaess. Hier steht jede gemeinsame Zeile
#   genau einmal; die methodenspezifischen Abweichungen stehen in EINEM
#   `case`-Block und sind damit vollstaendig aufzaehlbar.
#
# Usage (nach dem Throughput-Sweep, Werte in arc/final_config.sh gesetzt):
#   MODEL=sigmadock          sbatch arc/train_final_72h.slurm
#   MODEL=sigmaflow_minimal  sbatch arc/train_final_72h.slurm
#   MODEL=sigmaflow_source   sbatch arc/train_final_72h.slurm   # nur nach Gate 4
#   MODEL=sigmaflow_conf     sbatch arc/train_final_72h.slurm   # Extension
#
# Zweiter Seed:
#   MODEL=sigmaflow_minimal FINAL_SEED=1 sbatch arc/train_final_72h.slurm
#
# Fortsetzen nach Knotenausfall (Checkpointordner wird vorher gesichert):
#   MODEL=sigmaflow_minimal RESUME=/pfad/zu/last.ckpt sbatch arc/train_final_72h.slurm
#
# Ressourcen kommen aus final_config.sh und werden beim Submit ueber die
# Kommandozeile gesetzt -- #SBATCH-Zeilen koennen keine Variablen lesen.
# Deshalb der Wrapper: arc/submit_final.sh
#
#SBATCH --job-name=final72
# Default-Walltime. submit_final.sh ueberschreibt sie per Kommandozeile
# (sbatch-Argumente schlagen #SBATCH-Zeilen). Sie steht hier trotzdem, damit
# ein direkter 'sbatch arc/train_final_72h.slurm' nicht stillschweigend das
# kurze Partitionsdefault erbt und nach einer Stunde abgeschossen wird.
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@900
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail   # kein -e: der Trap soll auch nach einem Fehlschlag noch
                   # den Notfall-Checkpoint sichern koennen.

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/final_config.sh"
setup_wandb_hardening

MODEL="${MODEL:?MODEL setzen: sigmadock | sigmaflow_minimal | sigmaflow_source | sigmaflow_conf}"
RESUME="${RESUME:-}"

# ---------------------------------------------------------------- Screening
# MODE=screen laesst DASSELBE Trainingsprogramm laufen wie MODE=full und
# beendet es nur frueher. max_epochs, LR-Schedule, Batch, Seed, Datensatz und
# Loss bleiben Zeichen fuer Zeichen gleich; einzig SLURM-Walltime und das
# Snapshot-Raster aendern sich.
#
# Das ist die ganze Idee des Screenings: das Ergebnis nach 6 h soll direkt mit
# dem 6h-SNAPSHOT des langen Laufs vergleichbar sein. Waere max_epochs auf 6 h
# kalibriert, liefe ein vollstaendiger Cosine-Anneal durch und man vergliche
# zwei verschiedene Trainingsprogramme miteinander.
MODE="${MODE:-full}"
case "$MODE" in
  full)   SNAP_HOURS="${FINAL_SNAPSHOT_HOURS}" ;;
  screen) SNAP_HOURS="${SCREEN_SNAPSHOT_HOURS}" ;;
  *) echo "MODE unbekannt: $MODE (full|screen)"; exit 1 ;;
esac

# ---------------------------------------------------------------- Varianten
# Jede Zeile hier ist eine bewusste Abweichung. Es gibt keine anderen.
case "$MODEL" in
  sigmadock)
      RUN_ID="SD_BASE_72H"
      CODE_DIR="${ARC_SIGMADOCK}"
      # SigmaDock laeuft in myenv. sigmaflow_env haette SigmaFlow_Development in
      # site-packages, dort loest `import sigmadock` falsch auf (Job 8512799).
      module load Mamba
      source activate "$ARC_SD_ENV"
      PYTHON="$(which python)"
      # Methodenspezifisch: SigmaDocks Score-Parametrisierung auf SO(3).
      # Ein Flow-Matching-Modell hat dafuer kein Gegenstueck.
      METHOD_FLAGS=(--rot_score_method space --rot_score_scaling rms)
      EXPECT_IN_PATH="SigmaDock_Reproduction"
      ;;
  sigmaflow_minimal)
      RUN_ID="SF_MIN_72H"
      CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      METHOD_FLAGS=()
      EXPECT_IN_PATH="SigmaFlow_Minimal"
      ;;
  sigmaflow_source)
      RUN_ID="SF_SRC_72H"
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      # Die EINE Aenderung des Experiments. Alles andere kommt aus
      # final_config.sh und ist mit den uebrigen Varianten identisch.
      METHOD_FLAGS=(--source_mode "${SOURCE_MODE}" --source_median_deg "${SOURCE_MEDIAN_DEG}")
      EXPECT_IN_PATH="EXP-102"
      echo "[cfg] EXP-102 Quelle: mode=${SOURCE_MODE} median=${SOURCE_MEDIAN_DEG} Grad"
      if [ "${SOURCE_MEDIAN_DEG%%.*}" = "132" ] && [ "${SOURCE_MODE}" != "haar" ]; then
          echo "[cfg] WARNUNG: Medianwinkel steht noch auf dem Haar-Default 132.3."
          echo "[cfg]          Er soll aus exp101_distance_audit.py kommen."
      fi
      ;;
  sigmaflow_conf)
      RUN_ID="SF_CONF_INT_72H"
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-105_confidence_ranking"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      METHOD_FLAGS=()
      EXPECT_IN_PATH="EXP-105"
      ;;
  *) echo "MODEL unbekannt: $MODEL"; exit 1 ;;
esac

RUN_ID="${RUN_ID}_s${FINAL_SEED}"
[ "$MODE" = "screen" ] && RUN_ID="${RUN_ID%_s${FINAL_SEED}}_SCREEN6H_s${FINAL_SEED}"

if [ ! -d "$CODE_DIR" ]; then
    echo "FEHLER: Codeverzeichnis fehlt: $CODE_DIR"
    echo "  -> Variante noch nicht implementiert oder nicht gepusht."
    exit 1
fi

cd "$CODE_DIR" || exit 1
print_final_config

# ---------------------------------------------------------------- Sanity-Gate
# Faellt hier etwas durch, sind 72 GPU-Stunden gerettet.
"$PYTHON" - "$EXPECT_IN_PATH" <<'PYCHECK'
import os, sys
import sigmadock
expect = sys.argv[1]
path = os.path.abspath(sigmadock.__file__)
print("sigmadock geladen aus:", path)
assert expect in path, f"FALSCHE Quelle: erwartet '{expect}' in {path}"
forbidden = {"SigmaDock_Reproduction", "SigmaFlow_Minimal", "SigmaFlow_Development",
             "SigmaFlow_Variants", "archive", "EXP-100", "EXP-102", "EXP-105"} - {expect}
for f in forbidden:
    assert f not in path, f"FALSCHE Quelle ({f}): {path}"
if expect == "SigmaDock_Reproduction":
    import sigmadock.diff.denoiser  # noqa: F401
    print("Original-SigmaDock (Diffusions-Denoiser) verifiziert")
else:
    import inspect
    from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
    src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
    assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src, "FRAME FIX FEHLT"
    print("frame fix verified")
PYCHECK
[ $? -ne 0 ] && { echo "SANITY GATE FEHLGESCHLAGEN — Abbruch."; exit 1; }

write_run_metadata "$RUN_ID" "$CODE_DIR" "$PYTHON"
SNAP_DIR="${RUN_DIR}/snapshots"; mkdir -p "$SNAP_DIR"

"$PYTHON" -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

MARKER="${TMPDIR}/start_marker"; touch "$MARKER"

# ---------------------------------------------------------------- Snapshots
# ARCHIVSNAPSHOTS OHNE CODEAENDERUNG.
#   Lightning schreibt `last.ckpt` (ueberschrieben) und save_top_k=3 nach
#   val_loss. Ein Zwischenstand bei 24 h ueberlebt das nicht zuverlaessig.
#   Statt train.py anzufassen -- was fuer SigmaDock ohnehin ausgeschlossen ist,
#   das Verzeichnis ist read-only -- kopiert ein Hintergrundprozess `last.ckpt`
#   zu definierten Stundenmarken weg. Methodenagnostisch und symmetrisch.
find_last_ckpt() {
    find "${CODE_DIR}/experiments" -name last.ckpt -newer "$MARKER" 2>/dev/null | sort | tail -1
}

JOB_START_S=$(date +%s)

# Zu jedem Snapshot gehoert seine Trainingsposition. Ohne sie ist die
# Lernkurve nur gegen Wallclock auftragbar; mit ihr auch gegen Epochen und
# gesehene Beispiele. Die Werte kommen aus dem laufenden stdout-Log, nicht aus
# dem Checkpoint - der muesste dafuer geladen werden, was Torch importiert und
# waehrend des Trainings GPU-Speicher kosten koennte.
snapshot_meta() {
    local tag="$1" ckpt="$2" now elapsed ep st lr
    now=$(date +%s); elapsed=$(( now - JOB_START_S ))
    ep="$(grep -oE 'Epoch[ ]+[0-9]+' "${RUN_DIR}/train_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || true)"
    st="$(grep -oE '(global_step|Step)[ =:]+[0-9]+' "${RUN_DIR}/train_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || true)"
    lr="$(grep -oE 'lr-AdamW[^0-9]*[0-9.eE+-]+' "${RUN_DIR}/train_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9.eE+-]+$' || true)"
    {
        echo "tag            = ${tag}"
        echo "checkpoint     = $(basename "${ckpt:-none}")"
        echo "walltime_s     = ${elapsed}"
        echo "walltime_h     = $(awk -v s="$elapsed" 'BEGIN{printf "%.3f", s/3600}')"
        echo "epoch          = ${ep:-unknown}"
        echo "global_step    = ${st:-unknown}"
        echo "effective_batch= ${FINAL_EFFECTIVE_BATCH}"
        # examples_seen bleibt leer, wenn global_step nicht lesbar war. Eine
        # geschaetzte Zahl waere hier schaedlicher als eine fehlende.
        if [ -n "${st:-}" ]; then
            echo "examples_seen  = $(( st * FINAL_EFFECTIVE_BATCH ))"
        else
            echo "examples_seen  = unknown"
        fi
        echo "lr             = ${lr:-unknown}"
        echo "utc            = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${SNAP_DIR}/snap_${tag}.meta.txt"
}

snapshot_now() {
    local tag="$1" src dst
    src="$(find_last_ckpt)"
    dst="${SNAP_DIR}/snap_${tag}.ckpt"
    if [ -n "$src" ] && [ -f "$src" ]; then
        # cp auf ein temporaeres Ziel und erst dann umbenennen: mv innerhalb
        # desselben Dateisystems ist atomar. Sonst koennte ein spaeterer
        # Auswertungsjob eine halb geschriebene Datei oeffnen.
        if cp "$src" "${dst}.part" 2>/dev/null && mv "${dst}.part" "$dst"; then
            echo "[snap] ${tag}: $(basename "$src") -> $(basename "$dst")"
            snapshot_meta "$tag" "$src"
        else
            rm -f "${dst}.part"
            echo "[snap] ${tag}: Kopie fehlgeschlagen"
        fi
    else
        echo "[snap] ${tag}: noch kein last.ckpt vorhanden"
    fi
}

# ZEITPLAN - die Stunden sind ABSOLUT ab Jobstart, nicht kumulativ.
#   Eine frueherer Fassung schlief nacheinander h*3600 Sekunden je Eintrag.
#   Fuer "6 12 24 48" haette das bei 6, 18, 42 und 90 Stunden ausgeloest --
#   der letzte Snapshot waere nie gefallen, und die Lernkurve haette falsche
#   Zeitachsen getragen, ohne dass irgendetwas fehlgeschlagen waere.
#   Deshalb wird gegen die verstrichene Zeit seit JOB_START_S geschlafen.
( prev_s=0
  for H in ${SNAP_HOURS}; do
      target_s=$(awk -v h="$H" 'BEGIN{printf "%d", h*3600}')
      delta=$(( target_s - prev_s ))
      [ "$delta" -gt 0 ] && sleep "$delta"
      snapshot_now "$(printf '%03dh' "$H")"
      prev_s="$target_s"
  done ) &
SNAP_PID=$!

# ---------------------------------------------------------------- Signal-Trap
# SLURM sendet USR1 15 min vor dem Walltime-Limit (--signal=B:USR1@900).
# Wir sichern dann NUR den letzten Checkpoint und schreiben eine
# Resume-Anleitung. Bewusst KEIN automatisches Requeue: ein still
# fortgesetzter falscher Checkpoint waere teurer als ein manueller Neustart.
on_timeout() {
    echo
    echo "[trap] USR1 empfangen — Walltime laeuft in ~15 min ab."
    snapshot_now "emergency"
    {
        echo "# Notfall-Snapshot bei drohendem Timeout"
        echo "# Fortsetzen (Checkpointordner VORHER sichern, siehe Runbook 4):"
        echo "CKPT=$(find_last_ckpt)"
        echo 'cp -r "$(dirname "$CKPT")" "$(dirname "$CKPT")_backup_$(date +%s)"'
        echo "MODEL=${MODEL} FINAL_SEED=${FINAL_SEED} RESUME=\"\$CKPT\" \\"
        echo "  sbatch arc/train_final_72h.slurm"
    } > "${RUN_DIR}/RESUME_COMMAND.txt"
    echo "[trap] Resume-Anleitung: ${RUN_DIR}/RESUME_COMMAND.txt"
}
trap on_timeout USR1

# ---------------------------------------------------------------- Resume
RESUME_FLAGS=()
if [ -n "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "FEHLER: RESUME zeigt auf keine Datei: $RESUME"; exit 1
    fi
    # Die dokumentierte save_top_k-Falle: ein Resume schreibt in DENSELBEN
    # Ordner und verdraengt die alten Checkpoints. Deshalb vorher sichern.
    BK="$(dirname "$RESUME")_backup_$(date +%s)"
    cp -r "$(dirname "$RESUME")" "$BK" && echo "[resume] Alt-Checkpoints gesichert nach $BK"
    RESUME_FLAGS=(--resume_from_checkpoint "$RESUME")
    echo "[resume] setze fort ab: $RESUME"
fi

# ---------------------------------------------------------------- Training
# Validierung 1x pro Epoche wie im Original: val_check_interval als Schrittzahl.
STEPS_PER_EPOCH=$(( 19443 / FINAL_EFFECTIVE_BATCH ))
VAL_INTERVAL=$(( STEPS_PER_EPOCH * FINAL_VAL_EVERY_N_EPOCHS ))
echo "[cfg] steps/epoch ~ ${STEPS_PER_EPOCH}, val alle ${VAL_INTERVAL} Schritte"
echo "[cfg] erwartete Gesamtschritte ~ $(( STEPS_PER_EPOCH * FINAL_MAX_EPOCHS ))"
echo "[cfg] MODE=${MODE}; Snapshots bei (h): ${SNAP_HOURS}"
if [ "$MODE" = "screen" ]; then
    echo "[cfg] SCREENING: max_epochs bleibt ${FINAL_MAX_EPOCHS} (identischer LR-Schedule);"
    echo "[cfg]            SLURM beendet den Lauf nach ${SCREEN_WALLTIME}."
fi

# KEIN --debug. Es aktiviert detect_anomaly, profiler und FullNaNCheckCallback
# (Forward-Hooks auf jedem Modul) und ist ein Diagnose-, kein Trainingsmodus.
"$PYTHON" scripts/train.py \
    --data_dir "${ARC_DATA}" \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size "${FINAL_BATCH_SIZE}" \
    --accum_grad_batches "${FINAL_ACCUM}" \
    --num_workers "${FINAL_NUM_WORKERS}" \
    --accelerator gpu \
    --devices 1 \
    --precision "${FINAL_PRECISION}" \
    --cuda_precision "${FINAL_CUDA_PRECISION}" \
    --max_epochs "${FINAL_MAX_EPOCHS}" \
    --val_check_interval "${VAL_INTERVAL}" \
    --early_stopping_patience 0 \
    --trans_score_weight "${FINAL_TRANS_WEIGHT}" \
    --rot_score_weight "${FINAL_ROT_WEIGHT}" \
    --seed "${FINAL_SEED}" \
    --offline_run \
    "${METHOD_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train_stdout.log" &

TRAIN_PID=$!
wait $TRAIN_PID          # `wait` laesst den Trap durch; ohne & und wait waere USR1 blockiert
TRAIN_RC=$?

kill $SNAP_PID 2>/dev/null
snapshot_now "final"

record_experiment_dir "$CODE_DIR" "$MARKER"

# ---------------------------------------------------------------- Manifest
"$PYTHON" "${ARC_REPO}/arc/write_manifest.py" \
    --run_dir "$RUN_DIR" \
    --run_id "$RUN_ID" \
    --model "$MODEL" \
    --code_dir "$CODE_DIR" \
    --batch_size "$FINAL_BATCH_SIZE" \
    --accum "$FINAL_ACCUM" \
    --precision "$FINAL_PRECISION" \
    --max_epochs "$FINAL_MAX_EPOCHS" \
    --seed "$FINAL_SEED" \
    --train_rc "$TRAIN_RC" \
  || echo "[manifest] Erzeugung fehlgeschlagen (Training bleibt gueltig)"

echo
echo "=============================================================="
echo "${RUN_ID} beendet (rc=${TRAIN_RC})."
echo "  Laufverzeichnis: ${RUN_DIR}"
echo "  Snapshots:       ${SNAP_DIR}"
echo "  Manifest:        ${RUN_DIR}/manifest.json"
echo "=============================================================="
