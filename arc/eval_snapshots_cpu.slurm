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
#SBATCH --time=06:00:00
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

# WELCHER SATZ AUSGEWERTET WIRD
#   posebusters     -> die 209 entpackten Komplexe, mit Whitelist auf einen
#                      festen Teilsatz eingeschraenkt (billige Diagnosekurve)
#   posebusters308  -> die offiziellen 308 der Zeitschriftenfassung, also die
#                      Menge, auf der SigmaDock berichtet. Die Auswahl steckt
#                      dort im Verzeichnis (arc/make_v2_308_dir.sh); eine
#                      Whitelist waere wirkungslos, weil sample.py:458 sie nur
#                      bei woertlich "posebusters" anwendet -- und zwar STILL.
EXPERIMENT="${EXPERIMENT:-posebusters}"
TRUE_DIR="${TRUE_DIR:-${ARC_TRUE_DIR}}"

# KONFORMER: sampled ODER bound. Das ist keine Kleinigkeit.
#   sampled -> graph.sample_conformer=true, der Vorgabewert der Konfiguration.
#              Der Eingangskonformer wird erzeugt; das Modell kennt die
#              Kristallkonformation nicht.
#   bound   -> graph.sample_conformer=false, Fragmente aus der Kristallpose.
#              Verschafft SigmaDock einen Rotationsprior-Vorteil und verzerrt
#              damit den Vergleich zwischen den Armen.
#
#   eval_snapshots.slurm (GPU) uebergibt fest `false`, also bound. Fuer eine
#   Kurve, die Arme gegeneinander stellt, ist das die falsche Wahl; Vorgabe
#   hier ist deshalb sampled.
#
#   Die Einstellung steht im Verzeichnisnamen, damit bound- und sampled-Posen
#   nicht im selben Ordner landen und von MERGE zu einer Zahl vermengt werden.
CONFORMER="${CONFORMER:-sampled}"
case "$CONFORMER" in
  sampled) CONF_FLAG=true ;;
  bound)   CONF_FLAG=false ;;
  *) echo "CONFORMER unbekannt: ${CONFORMER} (sampled|bound)" >&2; exit 1 ;;
esac
SUBSET_FILE="${SUBSET_FILE:-${REPO_ROOT}/arc/eval_subset.txt}"

USE_WHITELIST=0
if [ "$EXPERIMENT" = "posebusters" ]; then
    USE_WHITELIST=1
    if [ ! -f "$SUBSET_FILE" ]; then
        echo "FEHLER: Subsetdatei fehlt: ${SUBSET_FILE}" >&2
        echo "  -> einmalig erzeugen mit:" >&2
        echo "     python arc/make_eval_subset.py --data_dir ${ARC_DATA}" >&2
        exit 1
    fi
else
    echo "[curve] Experiment '${EXPERIMENT}': Auswahl kommt aus dem Verzeichnis,"
    echo "        keine Whitelist. SUBSET_FILE wird ignoriert."
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
    # NICHT ins Arm-Verzeichnis wechseln. `SigmaFlow_Evaluation` liegt in der
    # Repo-Wurzel; von SigmaFlow_Minimal aus scheitert `python -m` mit
    # "No module named 'SigmaFlow_Evaluation'". Das Modul importiert nur
    # stdlib, numpy, rdkit und sich selbst -- der Arm-Baum wird nicht
    # gebraucht, die Umgebung ist ohnehin schon aktiviert.
    cd "$ARC_REPO" || exit 1

    # Zwei Felder fehlen in den .meta.txt der Snapshots. Sie werden aus der
    # Epoche abgeleitet und als solche gekennzeichnet -- geraten wird nichts.
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-594}"
    N_TRAIN="${N_TRAIN:-19037}"
    shopt -s nullglob
    for D in "${OUT_DIR}"/*__nfe*; do
        [ -d "$D" ] || continue
        BASE="$(basename "$D")"
        TAG="${BASE%%__nfe*}"
        REST="${BASE#*__nfe}"        # z.B. "25__sampled"
        NFE="${REST%%__*}"
        CONF="${REST#*__}"
        NPOSE="$(find "$D" -name '*.sdf' 2>/dev/null | wc -l)"
        if [ "$NPOSE" -eq 0 ]; then
            echo "[merge] ${BASE}: keine Posen -- uebersprungen"
            continue
        fi
        echo "[merge] ${BASE}: ${NPOSE} SDF"
        # --per_complex_csv ist NICHT optional-schmueckend: die Oracle-Groessen
        # ueber die Seeds stehen NUR dort (evaluate_run.py:373-394). Die JSON
        # kennt bloss `basics.single_sample`, also Seed 0. Ohne diese Tabelle
        # haetten wir zehn Seeds gesampelt und einen berichtet.
        # PYTHONPATH ausdruecklich um die Repo-Wurzel erweitern.
        #   `python -m` legt normalerweise das Arbeitsverzeichnis auf sys.path,
        #   aber darauf ist kein Verlass: auf ARC scheiterte der Aufruf mit
        #   "No module named 'SigmaFlow_Evaluation'", obwohl das Verzeichnis
        #   stimmte. Ursache ist vermutlich PYTHONSAFEPATH in der Umgebung;
        #   mit explizitem Suchpfad ist es egal.
        #   Dieselbe Schwachstelle steckt in eval_snapshots.slurm und
        #   compare_rankers.slurm.
        PYTHONPATH="${ARC_REPO}:${PYTHONPATH:-}" \
        "$PYTHON" -m SigmaFlow_Evaluation.evaluate_run \
            --sampling_root "$D" \
            --true_dir "${TRUE_DIR}" \
            --label "${MODEL}_${BASE}" \
            --out_json "${D}/evaluation.json" \
            --per_complex_csv "${D}/per_complex.csv" \
          > "${D}/evaluate.log" 2>&1 || echo "[merge] ${BASE}: Auswertung fehlgeschlagen"

        "$PYTHON" - "${ROW_DIR}/row_${BASE}.csv" "$TAG" "$NFE" "$CONF" "$N_SEEDS" \
                    "${SNAP_DIR}/snap_${TAG}.meta.txt" "${D}/evaluation.json" \
                    "${D}/per_complex.csv" "$STEPS_PER_EPOCH" "$N_TRAIN" <<'PYEOF'
import csv, json, statistics, sys, pathlib
(row_path, tag, nfe, conf, nseeds, meta_path, ev_path, pc_path,
 steps_per_epoch, n_train) = sys.argv[1:11]

def meta(key, default=""):
    try:
        for line in pathlib.Path(meta_path).read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    except OSError:
        pass
    return default

# global_step und examples_seen fehlen in den Metadateien ("unknown").
# Sie folgen aus der Epoche; die Herkunft wird in einer eigenen Spalte
# festgehalten, damit abgeleitete nicht als gemessene Werte durchgehen.
epoch_s = meta("epoch")
step_s, seen_s = meta("global_step"), meta("examples_seen")
quelle = "meta"
if step_s in ("", "unknown") or seen_s in ("", "unknown"):
    try:
        e = int(float(epoch_s))
        if step_s in ("", "unknown"):
            step_s = str(e * int(steps_per_epoch))
        if seen_s in ("", "unknown"):
            seen_s = str(e * int(n_train))
        quelle = f"epoch x {steps_per_epoch}"
    except ValueError:
        quelle = "missing"

base = [tag, nfe, conf, nseeds, meta("walltime_h"), epoch_s,
        step_s, seen_s, quelle, meta("lr")]

# Aus der JSON: was ueber ALLE Posen geht.
n_poses = rmsd_all = ""
try:
    b = json.loads(pathlib.Path(ev_path).read_text(encoding="utf-8"))["basics"]
    n_poses, rmsd_all = b.get("n_poses", ""), b.get("rmsd_median_all", "")
except Exception as exc:                      # noqa: BLE001
    print(f"[merge] {tag} nfe={nfe}: keine JSON-Metriken ({type(exc).__name__})")

# Aus der Per-Komplex-Tabelle: Seed 0 UND Oracle ueber die Seeds. Nur hier
# stehen die Oracle-Groessen; die JSON kennt sie nicht.
n_cx = rmsd_best = s2_seed0 = s2_oracle = s5_oracle = ""
try:
    rows = list(csv.DictReader(pathlib.Path(pc_path).open(encoding="utf-8")))
    if rows:
        n_cx = len(rows)
        rmsd_best = round(statistics.median(float(r["rmsd_best"]) for r in rows), 4)
        s2_seed0 = round(100 * statistics.fmean(int(r["success_2A_seed0"]) for r in rows), 2)
        s2_oracle = round(100 * statistics.fmean(int(r["success_2A_oracle"]) for r in rows), 2)
        s5_oracle = round(100 * statistics.fmean(int(r["success_5A_oracle"]) for r in rows), 2)
except Exception as exc:                      # noqa: BLE001
    # Fehlende Zahlen bleiben leer. Eine geschaetzte Zeile in einer Lernkurve
    # waere schaedlicher als eine fehlende.
    print(f"[merge] {tag} nfe={nfe}: keine Per-Komplex-Tabelle ({type(exc).__name__})")

row = base + [n_cx, n_poses, rmsd_all, rmsd_best, s2_seed0, s2_oracle, s5_oracle]
pathlib.Path(row_path).write_text(
    ",".join("" if v is None else str(v) for v in row) + "\n", encoding="utf-8")
PYEOF
    done

    OUT="${OUT_DIR}/curve_${MODEL}.csv"
    echo "snapshot,nfe,conformer,n_seeds,walltime_h,epoch,global_step,examples_seen,step_source,lr,n_complexes,n_poses,rmsd_median_all,rmsd_best_median,success_2A_seed0,success_2A_oracle,success_5A_oracle" > "$OUT"
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

OUT_ROOT="${OUT_DIR}/${TAG}__nfe${TASK_NFE}__${CONFORMER}"
mkdir -p "$OUT_ROOT"

echo "[curve] Aufgabe   : ${IDX} von ${N_TASKS}"
echo "[curve] Snapshot  : ${TAG}  (Index ${TASK_SNAP})"
echo "[curve] Schritte  : ${TASK_NFE}"
echo "[curve] Seed      : ${SEED}"
echo "[curve] Modell    : ${MODEL}  (${KIND})"
echo "[curve] Experiment: ${EXPERIMENT}"
echo "[curve] Konformer : ${CONFORMER}  (sample_conformer=${CONF_FLAG})"
if [ "$USE_WHITELIST" = "1" ]; then
    echo "[curve] Subset    : $(grep -cve '^[[:space:]]*$' "$SUBSET_FILE") Komplexe aus $(basename "$SUBSET_FILE")"
    WHITE_ARG=(data.blacklist="${SUBSET_FILE}")
else
    echo "[curve] Subset    : ganzes Verzeichnis des Experiments"
    WHITE_ARG=()
fi
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
    experiment=${EXPERIMENT} \
    output_dir="${OUT_ROOT}" \
    run_tag="${MODEL}_${TAG}_nfe${TASK_NFE}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    "${STEP_ARG[@]}" \
    "${WHITE_ARG[@]}" \
    graph.sample_conformer=${CONF_FLAG} \
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
