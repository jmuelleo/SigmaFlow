#!/bin/bash -l
#
# LERNKURVE — billige Auswertung aller 6h-Snapshots eines Laufs.
#
# ZWEI EBENEN, UND WARUM
#   Volles Protokoll heisst 209 Komplexe x 10 Seeds = 2090 Posen. Bei 12
#   Snapshots und 2 Baselines waeren das 50.160 Posen — das ist die Groessen-
#   ordnung des gesamten bisherigen Samplings des Projekts, nur fuer die
#   Zwischenstaende.
#
#   Deshalb:
#     BILLIG (dieses Skript)  jeder Snapshot, fester Subset, 1 Seed
#     VOLL   (sample_pb_seeds) nur FINAL_FULL_EVAL_HOURS, 10 Seeds, 209 Komplexe
#
#   Gespeichert wird haeufiger als evaluiert. Der Checkpoint ist billig, das
#   Sampling nicht.
#
# DER FESTE SUBSET
#   Damit metric(t) ueber die Zeit vergleichbar ist, MUSS ueber alle Snapshots
#   und beide Methoden derselbe Komplexsatz laufen. Er wird deshalb einmal
#   deterministisch gezogen (arc/eval_subset.txt) und danach nie geaendert.
#
# LEAKAGE
#   Der Subset stammt aus PoseBusters, dem Testsatz. Er wird ausschliesslich
#   BEOBACHTET — es wird kein Hyperparameter daran angepasst. Alle
#   Trainingsgroessen stehen vor dem ersten Snapshot in final_config.sh fest.
#   Das ist der einzige Umgang, der hier sauber ist: mehrfaches Ablesen ist
#   erlaubt, Tuning nicht.
#
# Usage:
#   RUN_DIR=/data/.../arc_runs/SF_MIN_72H_s0_8xxxxx MODEL=sigmaflow_minimal \
#       sbatch arc/eval_snapshots.slurm
#
#SBATCH --job-name=evalsnap
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/final_config.sh"
setup_wandb_hardening

SRC_RUN_DIR="${RUN_DIR:?RUN_DIR auf das Laufverzeichnis mit snapshots/ setzen}"
MODEL="${MODEL:?MODEL setzen: sigmadock | sigmaflow_minimal | sigmaflow_source | sigmaflow_conf}"
NFE="${NFE:-25}"
EVAL_SEED="${EVAL_SEED:-0}"
SUBSET_FILE="${SUBSET_FILE:-${REPO_ROOT}/arc/eval_subset.txt}"

SNAP_DIR="${SRC_RUN_DIR}/snapshots"
OUT_DIR="${SRC_RUN_DIR}/learning_curve"
mkdir -p "$OUT_DIR"

if [ ! -d "$SNAP_DIR" ]; then
    echo "FEHLER: keine snapshots/ unter ${SRC_RUN_DIR}"; exit 1
fi
if [ ! -f "$SUBSET_FILE" ]; then
    echo "FEHLER: Subsetdatei fehlt: ${SUBSET_FILE}"
    echo "  -> einmalig erzeugen mit: python arc/make_eval_subset.py"
    exit 1
fi

case "$MODEL" in
  sigmadock)
      CODE_DIR="${ARC_SIGMADOCK}"
      module load Mamba; source activate "$ARC_SD_ENV"; PYTHON="$(which python)" ;;
  sigmaflow_minimal) CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal" ;&
  sigmaflow_source|sigmaflow_conf)
      [ "$MODEL" = "sigmaflow_source" ] && CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source"
      [ "$MODEL" = "sigmaflow_conf" ]   && CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-105_confidence_ranking"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}" ;;
  *) echo "MODEL unbekannt: $MODEL"; exit 1 ;;
esac

N_SUBSET="$(grep -cve '^\s*$' "$SUBSET_FILE")"
echo "[curve] Modell   : ${MODEL}"
echo "[curve] Code     : ${CODE_DIR}"
echo "[curve] Subset   : ${N_SUBSET} Komplexe aus $(basename "$SUBSET_FILE")"
echo "[curve] NFE      : ${NFE}, Seed ${EVAL_SEED}"

cd "$CODE_DIR" || exit 1
SUMMARY="${OUT_DIR}/curve_${MODEL}.csv"
echo "snapshot,walltime_h,epoch,global_step,examples_seen,lr,n_poses,rmsd_median,success_2A,success_5A" > "$SUMMARY"

shopt -s nullglob
for CKPT in "${SNAP_DIR}"/snap_*.ckpt; do
    TAG="$(basename "$CKPT" .ckpt)"; TAG="${TAG#snap_}"
    META="${SNAP_DIR}/snap_${TAG}.meta.txt"
    echo
    echo "======== ${TAG} ========"

    SAMPLE_OUT="${OUT_DIR}/${TAG}"
    mkdir -p "$SAMPLE_OUT"

    # Schluessel exakt wie in arc/sample_pb_seeds.slurm, das nachweislich laeuft.
    # `data.blacklist` ist trotz des Namens eine WHITELIST: sample.py:457-464
    # laedt daraus PDB-IDs und ruft `prune_pairs_with_ids`. Sie wirkt nur bei
    # experiment=posebusters -- deshalb ist der feste Subset an PoseBusters
    # gebunden und kann nicht versehentlich auf einen anderen Satz zeigen.
    #
    # Scoring und PB-Checks sind hier AUS (raw). Die billige Kurve misst reine
    # Generationsqualitaet je Snapshot. Ranking kommt erst in der vollen
    # Auswertung dazu -- sonst mischten sich zwei Effekte in einer Kurve, und
    # ein Anstieg waere nicht mehr dem Generator zuzuordnen.
    "$PYTHON" scripts/sample.py \
        ckpt="${CKPT}" \
        data_dir="${ARC_DATA}" \
        experiment=posebusters \
        output_dir="${SAMPLE_OUT}" \
        run_tag="${MODEL}_${TAG}" \
        seed="${EVAL_SEED}" \
        num_seeds=1 \
        ode.num_steps="${NFE}" \
        graph.sample_conformer=false \
        data.blacklist="${SUBSET_FILE}" \
        postprocessing.scoring=null \
        postprocessing.bust_config=null \
        hydra.run.dir="${SAMPLE_OUT}/hydra_out" \
      > "${SAMPLE_OUT}/sample.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[curve] ${TAG}: Sampling fehlgeschlagen (rc=${rc}), siehe ${SAMPLE_OUT}/sample.log"
        # Ein kaputter Snapshot darf die restliche Kurve nicht verhindern.
        continue
    fi

    "$PYTHON" -m SigmaFlow_Evaluation.evaluate_run \
        --sampling_root "${SAMPLE_OUT}" \
        --true_dir "${ARC_DATA}/posebusters" \
        --label "${MODEL}_${TAG}" \
        --out_json "${SAMPLE_OUT}/evaluation.json" \
      > "${SAMPLE_OUT}/evaluate.log" 2>&1 || \
      echo "[curve] ${TAG}: Auswertung fehlgeschlagen"

    "$PYTHON" - "$SUMMARY" "$TAG" "$META" "${SAMPLE_OUT}/evaluation.json" <<'PYEOF'
import json, sys, pathlib
summary, tag, meta_path, ev_path = sys.argv[1:5]

def meta(key, default=""):
    try:
        for line in pathlib.Path(meta_path).read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    except OSError:
        pass
    return default

try:
    ev = json.loads(pathlib.Path(ev_path).read_text(encoding="utf-8"))
    b = ev["basics"]
    row = [tag, meta("walltime_h"), meta("epoch"), meta("global_step"),
           meta("examples_seen"), meta("lr"), b["n_poses"],
           b["rmsd_median_all"], b["single_sample"].get("success_below_2.0A"),
           b["single_sample"].get("success_below_5.0A")]
except Exception as exc:                      # noqa: BLE001
    # Fehlende Zahlen bleiben leer. Eine geschaetzte Zeile in einer Lernkurve
    # waere schaedlicher als eine fehlende.
    row = [tag, meta("walltime_h"), meta("epoch"), meta("global_step"),
           meta("examples_seen"), meta("lr"), "", "", "", ""]
    print(f"[curve] {tag}: keine Metriken ({type(exc).__name__})")
with open(summary, "a", encoding="utf-8") as fh:
    fh.write(",".join("" if v is None else str(v) for v in row) + "\n")
print(f"[curve] {tag}: Zeile geschrieben")
PYEOF
done

echo
echo "[curve] fertig: ${SUMMARY}"
cat "$SUMMARY"
