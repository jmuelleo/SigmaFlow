#!/bin/bash -l
#
# LERNKURVE UEBER DIE SNAPSHOTS EINES LAUFS -- CPU-only, Array ueber
# (Snapshot x Schrittzahl x Seed).
#
# WARUM ES DIESE FASSUNG GIBT
#   `eval_snapshots.slurm` fordert `gpu:l40s:1` an und rechnet EINEN Seed je
#   Snapshot nacheinander. GPU-Jobs auf ARC starten nur nach manueller
#   Priorisierung, und ein Seed sagt ueber die Erfolgsquote fast nichts.
#   Sampling braucht aber keine GPU: die vollstaendige 40-Seed-Auswertung ueber
#   209 Komplexe lief bereits auf CPU (arc/sample_pb_seeds_cpu.slurm), eine
#   Aufgabe dort in unter 90 Minuten.
#
# DER ZUSCHNITT
#   Eine Array-Aufgabe = ein (Snapshot, Schrittzahl, Seed). Bei 4 Snapshots,
#   zwei Schrittzahlen und 10 Seeds sind das 80 Aufgaben, jede rund 20 Minuten
#   bei NFE 25 und rund 5 Minuten bei NFE 5. Alle laufen gleichzeitig.
#
#   Die Seedzahl ist ueber N_SEEDS einstellbar, ohne dass sich sonst etwas
#   aendert: mit 40 werden es 320 Aufgaben, mit 80 dann 640. Ab etwa 300
#   empfiehlt sich eine Drosselung, `--array=0-319%50` laesst hoechstens 50
#   gleichzeitig laufen.
#
# WARUM ALLE SEEDS IN DASSELBE VERZEICHNIS SCHREIBEN
#   `evaluate_run` wertet einen Sampling-Baum als Ganzes aus. Nur wenn alle
#   Seeds eines (Snapshot, Schrittzahl) unter EINEM output_dir liegen, kann es
#   Oracle-Groessen ueber die Seeds bilden. sample.py legt darunter je Seed ein
#   `seed_<n>` an -- genau wie in arc/sample_pb_seeds_cpu.slurm.
#
# Usage:
#   # 1. Was waere zu tun?
#   LIST=1 RUN_DIR=/pfad/zum/lauf N_SEEDS=10 bash arc/eval_snapshots_cpu.slurm
#
#   # 2. Sampling einreichen (Indizes ab 0)
#   RUN_DIR=/pfad/zum/lauf MODEL=sigmaflow_minimal N_SEEDS=10 \
#       sbatch --array=0-79 arc/eval_snapshots_cpu.slurm
#
#   # 3. Auswerten und zur Kurve zusammenfuegen, wenn alle durch sind
#   MERGE=1 RUN_DIR=/pfad/zum/lauf MODEL=sigmaflow_minimal N_SEEDS=10 \
#       bash arc/eval_snapshots_cpu.slurm
#
# Kein `set -e`: `module load` und `conda activate` liefern hier regelmaessig
# einen Rueckgabewert != 0, ohne dass etwas kaputt ist. Fehler werden statt
# dessen an jeder Stelle einzeln geprueft.
#
#SBATCH --job-name=evalsnap-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/final_config.sh"

SRC_RUN_DIR="${RUN_DIR:?RUN_DIR auf das Laufverzeichnis mit snapshots/ setzen}"
SNAP_DIR="${SRC_RUN_DIR}/snapshots"
OUT_DIR="${SRC_RUN_DIR}/learning_curve_cpu"
ROW_DIR="${OUT_DIR}/rows"

N_SEEDS="${N_SEEDS:-10}"
NFE_LIST="${NFE_LIST:-25 5}"
read -r -a NFES <<< "$NFE_LIST"
N_NFE="${#NFES[@]}"

[ -d "$SNAP_DIR" ] || { echo "FEHLER: keine snapshots/ unter ${SRC_RUN_DIR}" >&2; exit 1; }

# Stabile, sortierte Liste. Die Array-Indizes beziehen sich darauf; sie darf
# sich zwischen LIST und Einreichung nicht aendern, sonst zeigt ein Index auf
# einen anderen Snapshot.
mapfile -t CKPTS < <(find "$SNAP_DIR" -maxdepth 1 -name 'snap_*.ckpt' | sort)
N_SNAP="${#CKPTS[@]}"
[ "$N_SNAP" -gt 0 ] || { echo "FEHLER: keine snap_*.ckpt in ${SNAP_DIR}" >&2; exit 1; }
N_TASKS=$(( N_SNAP * N_NFE * N_SEEDS ))

# Index -> (Snapshot, Schrittzahl, Seed). Seed laeuft am schnellsten, damit
# benachbarte Aufgaben denselben Snapshot bearbeiten und die Ausgabe eines
# (Snapshot, NFE) zusammen fertig wird.
decode() {
    local i="$1"
    TASK_SEED=$(( i % N_SEEDS ))
    local rest=$(( i / N_SEEDS ))
    TASK_NFE="${NFES[$(( rest % N_NFE ))]}"
    TASK_SNAP=$(( rest / N_NFE ))
}

if [ "${LIST:-0}" = "1" ]; then
    echo "Snapshots     : ${N_SNAP}"
    for i in "${!CKPTS[@]}"; do
        printf '  %2d  %s\n' "$i" "$(basename "${CKPTS[$i]}")"
    done
    echo "Schrittzahlen : ${NFES[*]}"
    echo "Seeds         : ${N_SEEDS}  (0 .. $(( N_SEEDS - 1 )))"
    echo "Aufgaben      : ${N_SNAP} x ${N_NFE} x ${N_SEEDS} = ${N_TASKS}"
    echo
    echo "Einreichen mit:  --array=0-$(( N_TASKS - 1 ))"
    [ "$N_TASKS" -gt 300 ] && echo "  (viele Aufgaben -- drosseln mit  --array=0-$(( N_TASKS - 1 ))%50 )"
    exit 0
fi

MODEL="${MODEL:?MODEL setzen: sigmadock | sigmaflow_minimal | sigmaflow_twohead | sigmaflow_source | sigmaflow_conf}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-0}"
SUBSET_FILE="${SUBSET_FILE:-${REPO_ROOT}/arc/eval_subset.txt}"

if [ ! -f "$SUBSET_FILE" ]; then
    echo "FEHLER: Subsetdatei fehlt: ${SUBSET_FILE}" >&2
    echo "  -> einmalig erzeugen mit:" >&2
    echo "     python arc/make_eval_subset.py --data_dir ${ARC_DATA}" >&2
    exit 1
fi

# sigmaflow_twohead fehlte in der GPU-Fassung; EXP-110 waere dort mit
# "MODEL unbekannt" abgebrochen. KIND steuert ausserdem den Namen der
# Schrittzahl: SigmaDock integriert keine ODE und hat keinen ode-Block.
case "$MODEL" in
  sigmadock)
      KIND="dock"
      CODE_DIR="${ARC_SIGMADOCK}"
      module load Mamba
      source activate "$ARC_SD_ENV"
      PYTHON="$(which python)" ;;
  sigmaflow_minimal|sigmaflow_twohead|sigmaflow_source|sigmaflow_conf)
      KIND="flow"
      case "$MODEL" in
        sigmaflow_minimal) CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal" ;;
        sigmaflow_twohead) CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field" ;;
        sigmaflow_source)  CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source" ;;
        sigmaflow_conf)    CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-105_confidence_ranking" ;;
      esac
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}" ;;
  *) echo "MODEL unbekannt: $MODEL" >&2; exit 1 ;;
esac
[ -d "$CODE_DIR" ] || { echo "FEHLER: Codeverzeichnis fehlt: ${CODE_DIR}" >&2; exit 1; }

# ------------------------------------------------ Auswerten und zusammenfuegen
if [ "${MERGE:-0}" = "1" ]; then
    mkdir -p "$ROW_DIR"
    cd "$CODE_DIR" || exit 1
    shopt -s nullglob
    for D in "${OUT_DIR}"/*__nfe*; do
        [ -d "$D" ] || continue
        BASE="$(basename "$D")"
        TAG="${BASE%%__nfe*}"
        NFE="${BASE##*__nfe}"
        NPOSE="$(find "$D" -name '*.sdf' 2>/dev/null | wc -l)"
        if [ "$NPOSE" -eq 0 ]; then
            echo "[merge] ${BASE}: keine Posen -- uebersprungen"
            continue
        fi
        echo "[merge] ${BASE}: ${NPOSE} SDF"
        "$PYTHON" -m SigmaFlow_Evaluation.evaluate_run \
            --sampling_root "$D" \
            --true_dir "${ARC_TRUE_DIR}" \
            --label "${MODEL}_${BASE}" \
            --out_json "${D}/evaluation.json" \
          > "${D}/evaluate.log" 2>&1 || echo "[merge] ${BASE}: Auswertung fehlgeschlagen"

        "$PYTHON" - "${ROW_DIR}/row_${BASE}.csv" "$TAG" "$NFE" "$N_SEEDS" \
                    "${SNAP_DIR}/snap_${TAG}.meta.txt" "${D}/evaluation.json" <<'PYEOF'
import json, sys, pathlib
row_path, tag, nfe, nseeds, meta_path, ev_path = sys.argv[1:7]

def meta(key, default=""):
    try:
        for line in pathlib.Path(meta_path).read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    except OSError:
        pass
    return default

base = [tag, nfe, nseeds, meta("walltime_h"), meta("epoch"),
        meta("global_step"), meta("examples_seen"), meta("lr")]
try:
    ev = json.loads(pathlib.Path(ev_path).read_text(encoding="utf-8"))
    b = ev["basics"]
    row = base + [b.get("n_poses", ""), b.get("rmsd_median_all", ""),
                  b.get("single_sample", {}).get("success_below_2.0A", ""),
                  b.get("single_sample", {}).get("success_below_5.0A", "")]
except Exception as exc:                      # noqa: BLE001
    # Fehlende Zahlen bleiben leer. Eine geschaetzte Zeile in einer Lernkurve
    # waere schaedlicher als eine fehlende.
    row = base + ["", "", "", ""]
    print(f"[merge] {tag} nfe={nfe}: keine Metriken ({type(exc).__name__})")
pathlib.Path(row_path).write_text(
    ",".join("" if v is None else str(v) for v in row) + "\n", encoding="utf-8")
PYEOF
    done

    OUT="${OUT_DIR}/curve_${MODEL}.csv"
    echo "snapshot,nfe,n_seeds,walltime_h,epoch,global_step,examples_seen,lr,n_poses,rmsd_median,success_2A,success_5A" > "$OUT"
    cat "${ROW_DIR}"/row_*.csv 2>/dev/null | sort >> "$OUT"
    echo
    echo "geschrieben: ${OUT}"
    cat "$OUT"
    exit 0
fi

# ---------------------------------------------------------------- Sampling
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "$N_TASKS" ]; then
    echo "FEHLER: Array-Index ${IDX} >= ${N_TASKS} Aufgaben. Mit LIST=1 nachsehen." >&2
    exit 1
fi
decode "$IDX"
CKPT="${CKPTS[$TASK_SNAP]}"
TAG="$(basename "$CKPT" .ckpt)"; TAG="${TAG#snap_}"
SEED=$(( EVAL_SEED_BASE + TASK_SEED ))

OUT_ROOT="${OUT_DIR}/${TAG}__nfe${TASK_NFE}"
mkdir -p "$OUT_ROOT"

N_SUBSET="$(grep -cve '^[[:space:]]*$' "$SUBSET_FILE")"
echo "[curve] Aufgabe   : ${IDX} von ${N_TASKS}"
echo "[curve] Snapshot  : ${TAG}  (Index ${TASK_SNAP})"
echo "[curve] Schritte  : ${TASK_NFE}"
echo "[curve] Seed      : ${SEED}"
echo "[curve] Modell    : ${MODEL}  (${KIND})"
echo "[curve] Subset    : ${N_SUBSET} Komplexe aus $(basename "$SUBSET_FILE")"
echo "[curve] Ausgabe   : ${OUT_ROOT}"

cd "$CODE_DIR" || exit 1

# Die Schrittzahl heisst in beiden Welten anders. SigmaDock integriert KEINE
# ODE und hat gar keinen ode-Block; ein unbedingtes ode.num_steps laesst Hydra
# mit "Key 'ode' is not in struct" abbrechen -- so geschehen in Array 8629911,
# alle 30 Tasks nach rund 35 s.
if [ "$KIND" = "dock" ]; then
    STEP_ARG=(diffusion.num_steps="${TASK_NFE}")
else
    STEP_ARG=(ode.num_steps="${TASK_NFE}")
fi

# `data.blacklist` ist trotz des Namens eine WHITELIST (sample.py:457-464).
# Scoring und PB-Checks sind AUS: die billige Kurve misst reine
# Generationsqualitaet. Mit Ranker mischten sich Generator und Auswahl in
# einer Kurve, und ein Anstieg waere nicht mehr zuzuordnen.
"$PYTHON" scripts/sample.py \
    ckpt="${CKPT}" \
    data_dir="${ARC_DATA}" \
    experiment=posebusters \
    output_dir="${OUT_ROOT}" \
    run_tag="${MODEL}_${TAG}_nfe${TASK_NFE}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    "${STEP_ARG[@]}" \
    graph.sample_conformer=false \
    data.blacklist="${SUBSET_FILE}" \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hardware.accelerator=cpu \
    hardware.devices=1 \
    hydra.run.dir="${OUT_ROOT}/hydra_out_s${SEED}" \
  > "${OUT_ROOT}/sample_s${SEED}.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "[curve] ${TAG} nfe=${TASK_NFE} seed=${SEED}: Sampling fehlgeschlagen (rc=${RC})" >&2
    tail -20 "${OUT_ROOT}/sample_s${SEED}.log" >&2
    exit "$RC"
fi

echo
echo "[curve] fertig. Wenn ALLE Aufgaben durch sind, auswerten mit:"
echo "  MERGE=1 RUN_DIR=${SRC_RUN_DIR} MODEL=${MODEL} N_SEEDS=${N_SEEDS} \\"
echo "      bash ${REPO_ROOT}/arc/eval_snapshots_cpu.slurm"
