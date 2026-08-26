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
# NIEMALS DIREKT MIT sbatch AUFRUFEN.
#   Die #SBATCH-Zeilen unten werden von SLURM gelesen, BEVOR Shellcode laeuft;
#   sie koennen final_config.sh nicht sehen und fordern daher WEDER Partition
#   NOCH --gres an. Ein direkter `sbatch arc/train_final_72h.slurm` bekommt
#   deshalb einen CPU-Knoten, baut minutenlang den Datafront auf und stirbt
#   dann in Lightning an "No supported gpu backend found!".
#   Genau so verloren: 8648437, 8648438, 8648458 am 2026-08-25.
#   Der GPU-Waechter weiter unten faengt das jetzt in Sekunden ab.
#
# Usage (nach dem Throughput-Sweep, Werte in arc/final_config.sh gesetzt):
#   bash arc/submit_final.sh sigmadock
#   bash arc/submit_final.sh sigmaflow_minimal
#   bash arc/submit_final.sh sigmaflow_twohead          # EXP-110
#   bash arc/submit_final.sh sigmaflow_source           # nur nach Gate 4
#   bash arc/submit_final.sh sigmaflow_conf             # Extension
#
# Zweiter Seed:
#   bash arc/submit_final.sh sigmaflow_minimal 1
#
# Vorher ansehen, ohne einzureichen:
#   DRY_RUN=1 bash arc/submit_final.sh sigmadock
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

MODEL="${MODEL:?MODEL setzen: sigmadock | sigmaflow_minimal | sigmaflow_twohead | sigmaflow_source | sigmaflow_conf}"
RESUME="${RESUME:-}"

# Horizont aufloesen, falls submit_final.sh ihn nicht schon exportiert hat
# (direkter `sbatch arc/train_final_72h.slurm` muss ebenso abgesichert sein).
if [ -z "${FINAL_MAX_STEPS:-}" ] || [ -z "${FINAL_MAX_EPOCHS:-}" ]; then
    resolve_horizon_for_model "$MODEL" || {
        echo "ABBRUCH: kein gueltiger Trainingshorizont fuer '${MODEL}'." >&2
        exit 1
    }
fi

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
      EXPECT_IN_PATH="SigmaDock_Reproduction_JulianMueller"
      GATE_KIND="diffusion"
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
      GATE_KIND="flow"
      ;;
  sigmaflow_twohead)
      # EXP-110 Zwei-Kopf. Trainingsseitig identisch zu sigmaflow_minimal;
      # der einzige Unterschied ist der Auslesekopf (+12,8 % Parameter, je ein
      # trans_block und rot_block auf geteiltem Rumpf). Deshalb eigener
      # Durchsatz und eigener Horizont -- FINAL_MAX_EPOCHS_SIGMAFLOW_TWOHEAD.
      RUN_ID="SF_2H_72H"
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      METHOD_FLAGS=()
      EXPECT_IN_PATH="EXP-110_two_head_vector_field"
      GATE_KIND="flow_twohead"
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
      EXPECT_IN_PATH="EXP-102_heuristic_conditional_source"
      GATE_KIND="flow"
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
      EXPECT_IN_PATH="EXP-105_confidence_ranking"
      GATE_KIND="flow"
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

# ---------------------------------------------------------------- Training
#
# SCHRITT-SEMANTIK. Drei Groessen, die man nicht verwechseln darf:
#
#   MICROBATCHES_PER_EPOCH = ceil(N_train / B_phys)
#       Was der DataLoader liefert. drop_last ist NICHT gesetzt
#       (data.py:736-747), der angebrochene letzte Batch zaehlt also mit.
#       Lightning misst `val_check_interval` in DIESER Einheit.
#
#   STEPS_PER_EPOCH        = floor(N_train / B_eff)
#       Optimizer-Steps je Epoche. Der LR-Scheduler laeuft in dieser Einheit
#       ("interval": "step", trainer.py:158-161). Bewusst abgerundet: der
#       Horizont soll eher zu klein als zu gross sein.
#
#   MAX_STEPS              = STEPS_PER_EPOCH * FINAL_MAX_EPOCHS
#       Der Scheduler-Horizont, den wir EXPLIZIT uebergeben. Ohne ihn rechnet
#       train.py:221 ohne Division durch accum und der Horizont waere um genau
#       den Akkumulationsfaktor zu gross (numerisch belegt in
#       arc/test_scheduler_horizon.py).
#
if [ -z "${FINAL_N_TRAIN:-}" ]; then
    echo "FEHLER: FINAL_N_TRAIN ist nicht gesetzt."
    echo "  Die echte Laenge der Trainings-Datafront kommt aus dem Stufe-2-"
    echo "  Throughput-Lauf und steht in arc/final_horizon.env."
    echo "  Die Paper-Zahl 19443 wird hier bewusst NICHT als Ersatz benutzt."
    exit 1
fi
if [ -z "${FINAL_MAX_EPOCHS:-}" ] || [ -z "${FINAL_MAX_STEPS:-}" ]; then
    echo "FEHLER: FINAL_MAX_EPOCHS / FINAL_MAX_STEPS fehlen."
    echo "  Normalerweise setzt arc/submit_final.sh sie. Bei direktem sbatch:"
    echo "  resolve_horizon_for_model ${MODEL} muss vorher erfolgreich sein."
    exit 1
fi

MICROBATCHES_PER_EPOCH=$(( (FINAL_N_TRAIN + FINAL_BATCH_SIZE - 1) / FINAL_BATCH_SIZE ))
STEPS_PER_EPOCH=$(( FINAL_N_TRAIN / FINAL_EFFECTIVE_BATCH ))
VAL_INTERVAL=$(( MICROBATCHES_PER_EPOCH * FINAL_VAL_EVERY_N_EPOCHS ))
MAX_STEPS="${FINAL_MAX_STEPS}"

# Gegenprobe: der uebergebene Horizont muss zu den lokal berechneten
# Groessen passen. Weicht er ab, ist final_horizon.env veraltet.
EXPECTED_MAX_STEPS=$(( STEPS_PER_EPOCH * FINAL_MAX_EPOCHS ))
if [ "$MAX_STEPS" -ne "$EXPECTED_MAX_STEPS" ]; then
    echo "FEHLER: FINAL_MAX_STEPS=${MAX_STEPS} passt nicht zu"
    echo "        floor(${FINAL_N_TRAIN}/${FINAL_EFFECTIVE_BATCH}) * ${FINAL_MAX_EPOCHS} = ${EXPECTED_MAX_STEPS}."
    exit 1
fi

echo "[cfg] N_train (gemessen)   = ${FINAL_N_TRAIN}"
echo "[cfg] Microbatches/Epoche  = ${MICROBATCHES_PER_EPOCH}   (Einheit von val_check_interval)"
echo "[cfg] Optimizer-Steps/Ep.  = ${STEPS_PER_EPOCH}   (Einheit des LR-Schedulers)"
echo "[cfg] max_epochs           = ${FINAL_MAX_EPOCHS}"
echo "[cfg] max_steps (Horizont) = ${MAX_STEPS}"
echo "[cfg] Validierung alle ${VAL_INTERVAL} Microbatches"

# ------------------------------------------------------------ CUDA-Allokator
# EXP-110 ist am 2026-08-26 nach 11 h und 18.445 Schritten an
#     CUDA out of memory. Tried to allocate 686.00 MiB.
#     44.39 GiB Kapazitaet, 451 MiB frei, 873 MiB reserved but unallocated
# gestorben. Der Durchsatz-Sweep hatte fuer diesen Arm 44.143 MiB Spitze
# gemessen, also 97 Prozent der Karte -- kein Spielraum. Die Liganden sind
# unterschiedlich gross (18 bis ueber 56 Atome), irgendwann trifft eine
# Charge aus lauter grossen Komplexen und die letzten Hundert MiB fehlen.
#
# expandable_segments loest die 873 MiB Fragmentierung auf und deckt damit
# die fehlenden 686 MiB. Es aendert NUR die Speicherverwaltung, nicht die
# Arithmetik -- die Vergleichbarkeit der Arme bleibt unberuehrt. Gesetzt wird
# es fuer ALLE Arme, damit sie sich auch hierin nicht unterscheiden, und VOR
# dem GPU-Waechter, weil der bereits torch importiert und CUDA anfasst.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[cfg] PYTORCH_CUDA_ALLOC_CONF = ${PYTORCH_CUDA_ALLOC_CONF}"


# ---------------------------------------------------------------- Sanity-Gate
# Faellt hier etwas durch, sind 72 GPU-Stunden gerettet.
# Die Pruefung steht in arc/final_sanity_gate.py, nicht hier: sie ist damit
# lokal testbar (arc/test_final_sanity_gate.py, 9 Faelle). Die vorherige
# Inline-Fassung pruefte auf Teilzeichenketten und haette jeden
# SigmaFlow-Lauf getoetet, weil die Repowurzel auf ARC
# SigmaFlow_Development_JulianMueller heisst.
"$PYTHON" "${ARC_REPO}/arc/final_sanity_gate.py" "$EXPECT_IN_PATH" "$GATE_KIND"
[ $? -ne 0 ] && { echo "SANITY GATE FEHLGESCHLAGEN — Abbruch."; exit 1; }

# ------------------------------------------------------------- GPU-Waechter
# Ohne diese Pruefung laeuft ein Job ohne GPU-Zuteilung noch minutenlang
# weiter -- Datenfront aufbauen, Modell instanziieren, wandb starten -- und
# stirbt erst tief in Lightning an
#     MisconfigurationException: No supported gpu backend found!
# Am 2026-08-25 hat das drei Jobs gekostet (8648437, 8648438, 8648458), weil
# sie direkt per sbatch statt ueber submit_final.sh eingereicht wurden und
# damit ohne --gres liefen. Der Fehler gehoert in die zweite Sekunde, und die
# Meldung muss sagen, was zu tun ist -- nicht nur, dass etwas fehlt.
GPU_NAME="$("$PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
print(torch.cuda.get_device_name(0))
" 2>/dev/null)" || {
    echo "GPU-WAECHTER FEHLGESCHLAGEN — Abbruch." >&2
    echo "  torch.cuda.is_available() ist False: dieser Job hat keine GPU." >&2
    echo "  Knoten          : $(hostname)" >&2
    echo "  SLURM_JOB_GPUS  : ${SLURM_JOB_GPUS:-<keine gres/gpu-Zuteilung>}" >&2
    echo "" >&2
    echo "  Dieses Skript nicht direkt mit sbatch aufrufen. Die #SBATCH-Zeilen" >&2
    echo "  koennen final_config.sh nicht lesen und fordern keine GPU an." >&2
    echo "  Richtig:  bash arc/submit_final.sh ${MODEL}" >&2
    exit 1
}
echo "GPU-WAECHTER OK: ${GPU_NAME}"

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
        # Ohne diese drei Zeilen ist spaeter nicht mehr entscheidbar, ob ein
        # Checkpoint ein Zwischenstand oder ein ausannealter Endpunkt ist.
        echo "schedule_max_epochs = ${FINAL_MAX_EPOCHS}"
        echo "schedule_max_steps  = ${FINAL_MAX_STEPS}"
        echo "snapshot_kind       = mid_schedule_snapshot"
        echo "is_annealed_endpoint= false"
        if [ -n "${st:-}" ] && [ -n "${FINAL_MAX_STEPS:-}" ] && [ "${FINAL_MAX_STEPS}" -gt 0 ]; then
            echo "schedule_progress   = $(awk -v a="$st" -v b="$FINAL_MAX_STEPS" 'BEGIN{printf "%.4f", a/b}')"
        fi
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
# BENENNUNG — die wichtigste Zeile dieses Blocks.
#   Ein Snapshot bei 12 h aus diesem Lauf ist NICHT dasselbe wie der alte,
#   eigenstaendige 12h-Lauf. Der alte hatte einen auf 12 h kalibrierten
#   Cosine-Anneal und endete auf der minimalen Lernrate. Dieser hier sitzt
#   mitten in einem auf ${FINAL_MAX_EPOCHS} Epochen kalibrierten Schedule und
#   steht noch auf hoher Lernrate. Die beiden in eine Lernkurve zu mischen
#   waere ein Kategorienfehler.
#   Deshalb traegt jeder Snapshot den Schedule, zu dem er gehoert, im Namen:
#       sched070ep_at_012h.ckpt      (Zwischenstand eines langen Schedules)
#   gegenueber der alten Konvention
#       annealed_endpoint_12h        (eigenstaendig ausannealter Lauf)
SCHED_TAG="$(printf 'sched%03dep' "${FINAL_MAX_EPOCHS}")"
echo "[snap] Namensschema: ${SCHED_TAG}_at_<hhh>h  (Zwischenstaende, KEINE annealten Endpunkte)"

( prev_s=0
  for H in ${SNAP_HOURS}; do
      target_s=$(awk -v h="$H" 'BEGIN{printf "%d", h*3600}')
      delta=$(( target_s - prev_s ))
      [ "$delta" -gt 0 ] && sleep "$delta"
      snapshot_now "$(printf '%s_at_%03dh' "$SCHED_TAG" "$H")"
      prev_s="$target_s"
  done ) &
SNAP_PID=$!

# ---------------------------------------------------------------- Signal-Trap
# SLURM sendet USR1 15 min vor dem Walltime-Limit (--signal=B:USR1@900).
# Wir sichern dann NUR den letzten Checkpoint und schreiben eine
# Resume-Anleitung. Bewusst KEIN automatisches Requeue: ein still
# fortgesetzter falscher Checkpoint waere teurer als ein manueller Neustart.
USR1_RECEIVED=0
on_timeout() {
    USR1_RECEIVED=1
    echo
    echo "[trap] USR1 empfangen — Walltime laeuft in ~15 min ab."
    snapshot_now "${SCHED_TAG:-sched}_emergency"
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

# ---------------------------------------------------------------- Resume-Guard
# WARUM DIESER GUARD EXISTIERT
#   configure_optimizers() baut den Scheduler beim Resume NEU auf, aus dem
#   AKTUELLEN max_steps, und wendet darauf den wiederhergestellten
#   Step-Zaehler an. Ein geaenderter Horizont erzeugt damit still einen
#   ungeplanten Warm Restart -- die Lernrate springt zurueck nach oben, ohne
#   dass irgendetwas fehlschlaegt.
#
#   Die Pruefung liest den Horizont direkt aus dem Checkpoint. Der
#   LightningModule ruft save_hyperparameters() (trainer.py), damit stehen
#   max_steps und die LR-Parameter in ckpt["hyper_parameters"]. Das ist
#   belastbarer als eine Seitendatei, die verloren gehen kann.
guard_resume_horizon() {
    local ckpt="$1" want_steps="$2"
    "$PYTHON" - "$ckpt" "$want_steps" <<'PYGUARD'
import sys, torch

ckpt_path, want = sys.argv[1], int(sys.argv[2])
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hp = ck.get("hyper_parameters") or {}

got = hp.get("max_steps")
if got is None:
    print("[resume-guard] WARNUNG: max_steps steht nicht im Checkpoint.")
    print("[resume-guard] Der Horizont kann nicht verifiziert werden.")
    sys.exit(3)

print(f"[resume-guard] Checkpoint-Horizont : {got}")
print(f"[resume-guard] Aktueller Horizont  : {want}")
if int(got) != want:
    print("[resume-guard] ABBRUCH: die Horizonte weichen ab. Ein Resume wuerde")
    print("[resume-guard]          den LR-Verlauf umbauen (ungeplanter Warm Restart).")
    print("[resume-guard]          Entweder den alten Horizont wiederherstellen")
    print("[resume-guard]          oder den Lauf neu starten -- nicht fortsetzen.")
    sys.exit(1)

# Die uebrigen schedulerdefinierenden Groessen muessen ebenfalls stehen.
for key, expected_env in (("num_warmup_steps", None),):
    if key in hp:
        print(f"[resume-guard] {key} = {hp[key]}")

for key in ("max_lr_start", "max_lr_end", "min_lr_start", "min_lr_end",
            "num_lr_cycles", "cycle_warmup_frac"):
    if key in hp:
        print(f"[resume-guard] {key} = {hp[key]}")

print(f"[resume-guard] global_step im Checkpoint = {ck.get('global_step', 'unbekannt')}")
print(f"[resume-guard] epoch im Checkpoint       = {ck.get('epoch', 'unbekannt')}")
print(f"[resume-guard] EMA vorhanden             = {'ema_state_dict' in ck}")
print("[resume-guard] OK")
PYGUARD
}



if [ -n "$RESUME" ]; then
    guard_resume_horizon "$RESUME" "$MAX_STEPS"
    GUARD_RC=$?
    if [ "$GUARD_RC" -eq 1 ]; then
        echo "ABBRUCH: Resume-Guard hat den Lauf gestoppt."
        exit 1
    elif [ "$GUARD_RC" -eq 3 ]; then
        echo "ABBRUCH: Horizont im Checkpoint nicht pruefbar. Bewusst kein"
        echo "         Blindstart -- lieber neu beginnen als still zerstoeren."
        exit 1
    fi
fi
echo "[cfg] MODE=${MODE}; Snapshots bei (h): ${SNAP_HOURS}"
if [ "$MODE" = "screen" ]; then
    echo "[cfg] SCREENING: max_epochs bleibt ${FINAL_MAX_EPOCHS} (identischer LR-Schedule);"
    echo "[cfg]            SLURM beendet den Lauf nach ${SCREEN_WALLTIME}."
fi

# KEIN --debug. Es aktiviert detect_anomaly, profiler und FullNaNCheckCallback
# (Forward-Hooks auf jedem Modul) und ist ein Diagnose-, kein Trainingsmodus.
#
# KEIN --accum_grad_batches. Das Feld existiert zwar in RunConfig, ist aber in
# KEINEM der Baeume als argparse-Flag exponiert (geprueft in SigmaDock und in
# SigmaFlow_Minimal). Der Aufruf scheiterte damit sofort mit rc=2:
#   train.py: error: unrecognized arguments: --accum_grad_batches 1
# Das war die Todesursache von Job 8634117. Der Dataclass-Vorgabewert ist 1,
# das Weglassen ist bei FINAL_ACCUM=1 also verhaltensgleich. Bei jedem anderen
# Wert waere es das NICHT -- deshalb der Abbruch statt stiller Abweichung.
if [ "${FINAL_ACCUM}" != "1" ]; then
    echo "FEHLER: FINAL_ACCUM=${FINAL_ACCUM}, aber scripts/train.py kennt kein" >&2
    echo "        --accum_grad_batches. Ein Wert != 1 ist so nicht uebergebbar." >&2
    echo "        Entweder das Flag in config.py exponieren oder die effektive" >&2
    echo "        Batchgroesse ueber --batch_size allein einstellen." >&2
    exit 1
fi
"$PYTHON" scripts/train.py \
    --data_dir "${ARC_DATA}" \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size "${FINAL_BATCH_SIZE}" \
    --num_workers "${FINAL_NUM_WORKERS}" \
    --accelerator gpu \
    --devices 1 \
    --precision "${FINAL_PRECISION}" \
    --cuda_precision "${FINAL_CUDA_PRECISION}" \
    --max_epochs "${FINAL_MAX_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --val_check_interval "${VAL_INTERVAL}" \
    --early_stopping_patience 0 \
    --trans_score_weight "${FINAL_TRANS_WEIGHT}" \
    --rot_score_weight "${FINAL_ROT_WEIGHT}" \
    --seed "${FINAL_SEED}" \
    --offline_run \
    "${METHOD_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}" \
  > "${RUN_DIR}/train_stdout.log" 2>&1 &

# WARUM KEIN `| tee` MEHR
#   Frueher stand hier `python ... | tee log &`. In einer Pipeline ist `$!`
#   die PID des LETZTEN Prozesses, also von `tee` -- und `tee` gelingt fast
#   immer. Ein abgestuerztes Training waere damit als rc=0 im Manifest
#   gelandet. Dieselbe Klasse Fehler wie das fehlende `set -e`, das schon
#   einmal einen Fehlschlag als Erfolg gemeldet hat.
#   Jetzt ist TRAIN_PID eindeutig der Python-Prozess. Mitlesen geht per
#   `tail -f "${RUN_DIR}/train_stdout.log"`.
TRAIN_PID=$!
wait $TRAIN_PID          # `wait` laesst den Trap durch; ohne & und wait waere USR1 blockiert
TRAIN_RC=$?

# `wait` kann durch das USR1-Signal vorzeitig zurueckkehren, ohne dass das
# Training beendet ist. Dann ist rc > 128 und wir warten erneut, sonst
# stuende im Manifest der Signalcode statt des echten Ergebnisses.
while [ "$TRAIN_RC" -gt 128 ] && kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait $TRAIN_PID
    TRAIN_RC=$?
done

# Ergebnis klassifizieren, statt nur eine Zahl zu speichern.
if [ "$TRAIN_RC" -eq 0 ]; then
    TRAIN_STATUS="COMPLETED"
elif [ "${USR1_RECEIVED:-0}" = "1" ]; then
    TRAIN_STATUS="WALLTIME_INTERRUPTED"
elif [ "$TRAIN_RC" -gt 128 ]; then
    TRAIN_STATUS="KILLED_SIGNAL_$(( TRAIN_RC - 128 ))"
else
    TRAIN_STATUS="CRASHED_rc${TRAIN_RC}"
fi
echo "[status] TRAIN_RC=${TRAIN_RC}  TRAIN_STATUS=${TRAIN_STATUS}"

kill $SNAP_PID 2>/dev/null
snapshot_now "${SCHED_TAG:-sched}_final"

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
    --max_steps "$MAX_STEPS" \
    --n_train "$FINAL_N_TRAIN" \
    --world_size 1 \
    --cuda_precision "$FINAL_CUDA_PRECISION" \
    --partition "${SLURM_JOB_PARTITION:-unknown}" \
    --requested_walltime "$FINAL_WALLTIME" \
    --horizon_source "${FINAL_HORIZON_SOURCE:-unknown}" \
    --seed "$FINAL_SEED" \
    --train_rc "$TRAIN_RC" \
    --train_status "$TRAIN_STATUS" \
  || echo "[manifest] Erzeugung fehlgeschlagen (Training bleibt gueltig)"

echo
echo "=============================================================="
echo "${RUN_ID} beendet (rc=${TRAIN_RC})."
echo "  Laufverzeichnis: ${RUN_DIR}"
echo "  Snapshots:       ${SNAP_DIR}"
echo "  Manifest:        ${RUN_DIR}/manifest.json"
echo "=============================================================="

# Der Rueckgabewert des Trainings MUSS nach aussen. Ohne diese Zeile endet das
# Skript mit dem Status des letzten echo, also 0, und sacct meldet COMPLETED --
# so geschehen bei Job 8634117: rc=2 nach 46 Sekunden, in der Warteschlange als
# Erfolg gefuehrt. Dieselbe Klasse Fehler wie das fehlende `set -e`.
exit "$TRAIN_RC"
