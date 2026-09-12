#!/bin/bash -l
#
# POSEN AUF PDBBIND-GENERAL FUER DAS RANKER-TRAINING -- CPU-only
#
# WOZU
#   Ein gelernter Ranker braucht Posen auf dem TRAININGSSATZ. Auf PB308 oder
#   Astex trainiert waere er auf dem Testsatz trainiert, und keine Kennzahl
#   der Arbeit waere danach noch etwas wert.
#
# DER ZUSCHNITT
#   Eine Array-Aufgabe = ein (Block, Seed). Bei vier Bloecken und zehn Seeds
#   sind das 40 Aufgaben, jede rund 1,9 h bei fuenf Integrationsschritten --
#   damit passen sie in die Sechs-Stunden-Grenze von `short`.
#
#   Index i  ->  BLOCK = i / N_SEEDS,  SEED = i % N_SEEDS
#
# WARUM DIE TRAJEKTORIEN GELOESCHT WERDEN
#   predictions.pt macht 79 Prozent des Speichers aus, rund 3,7 GB ueber die
#   ganze Kampagne. ranker_data.py liest Posen, per_pose.csv, die
#   Redock-Tabellen und gnina -- keine Trajektorien. Auf einem Volume, das
#   seit Tagen bei 100 Prozent steht, ist das der Unterschied zwischen 1 und
#   5 GB. Wer sie doch braucht: KEEP_PRED=1.
#
# WAS DIESES SKRIPT NICHT TUT
#   Redocken und gnina. Das sind eigene Arrays, die auf den erzeugten Posen
#   aufsetzen; sie kommen danach.
#
# Aufruf:
#   # 1. Bloecke und Experimentkonfigurationen anlegen (einmalig)
#   CODE_DIR=$PWD/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field \
#     bash arc/make_ranker_blocks.sh
#
#   # 2. Was waere zu tun?
#   LIST=1 MODEL=sigmaflow_twohead CKPT=<pfad> bash arc/sample_ranker.slurm
#
#   # 3. Einreichen
#   MODEL=sigmaflow_twohead CKPT=<pfad> sbatch --array=0-39 arc/sample_ranker.slurm
#
#SBATCH --job-name=rankdata
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -uo pipefail

# SLURM kopiert das Batch-Skript nach /var/spool/slurmd/jobNNN/ und fuehrt die
# Kopie aus. ${BASH_SOURCE[0]} zeigt deshalb NICHT ins Repo, und ein relativer
# source-Pfad greift ins Leere -- genau daran starb Array 8691800 komplett.
# Absoluter Pfad, wie in allen anderen arc/*.slurm.
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:?MODEL setzen, z.B. sigmaflow_twohead}"
CKPT="${CKPT:?CKPT auf den Endpunkt-Snapshot setzen}"
BLOECKE="${BLOECKE:-4}"
N_SEEDS="${N_SEEDS:-10}"
NFE="${NFE:-5}"
CONF_FLAG="${CONF_FLAG:-true}"          # sampled, nicht bound
OUT_ROOT="${OUT_ROOT:-${ARC_RUNS}/ranker_data_${MODEL}}"

case "$MODEL" in
  sigmaflow_minimal) CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal" ;;
  sigmaflow_twohead) CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field" ;;
  *) echo "MODEL fuer diese Kampagne nicht vorgesehen: $MODEL" >&2; exit 1 ;;
esac

N_TASKS=$((BLOECKE * N_SEEDS))

if [ "${LIST:-0}" = "1" ]; then
    echo "Modell        : $MODEL"
    echo "Checkpoint    : $CKPT"
    echo "Codebaum      : $CODE_DIR"
    echo "Bloecke       : $BLOECKE   Seeds: $N_SEEDS   NFE: $NFE"
    echo "Aufgaben      : $N_TASKS   -> --array=0-$((N_TASKS - 1))"
    echo "Ausgabe       : $OUT_ROOT"
    echo
    for ((b = 0; b < BLOECKE; b++)); do
        y="${CODE_DIR}/conf/experiments/pdbbind-ranker-${b}.yaml"
        d="${ARC_DATA}/pdbbind/ranker-block-${b}"
        n=$(find "$d" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
        printf "  Block %d: %6d Komplexe   Konfiguration %s\n" \
               "$b" "$n" "$([ -f "$y" ] && echo vorhanden || echo FEHLT)"
    done
    exit 0
fi

[ -f "$CKPT" ] || { echo "FEHLER: CKPT fehlt: $CKPT" >&2; exit 1; }
i="${SLURM_ARRAY_TASK_ID:?nur als Array-Job sinnvoll}"
[ "$i" -lt "$N_TASKS" ] || { echo "FEHLER: Index $i >= $N_TASKS" >&2; exit 1; }

BLOCK=$((i / N_SEEDS))
SEED=$((i % N_SEEDS))
EXPERIMENT="pdbbind-ranker-${BLOCK}"

YAML="${CODE_DIR}/conf/experiments/${EXPERIMENT}.yaml"
[ -f "$YAML" ] || {
    echo "FEHLER: ${YAML} fehlt. Erst arc/make_ranker_blocks.sh laufen lassen." >&2
    exit 1
}

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

ZELLE="${OUT_ROOT}/block${BLOCK}__nfe${NFE}"
mkdir -p "$ZELLE"
cd "$CODE_DIR" || exit 1

echo "[rank] Aufgabe $i: Block $BLOCK, Seed $SEED, NFE $NFE"
echo "[rank] Experiment $EXPERIMENT, Ausgabe $ZELLE"

# SCHLUESSELNAME num_steps: `ode`, NICHT `diffusion`.
#   `diffusion.num_steps` ist der Name aus SigmaDock. Beide SigmaFlow-Arme
#   haben die Gruppe in conf/sampling/base.yaml:60 nach `ode` umbenannt, und
#   sample.py liest cfg.ode.num_steps. Mit dem alten Namen waere dieser Array
#   entweder an Hydra abgebrochen oder haette still mit der Vorgabe 25
#   gesampelt: fuenffache Kosten und die falschen Posen fuer einen Ranker,
#   der gerade das 5-Schritt-Regime abbilden soll. Gefunden 2026-09-02.
"$PYTHON" scripts/sample.py \
    ckpt="$CKPT" \
    data_dir="${ARC_DATA}" \
    experiment="${EXPERIMENT}" \
    output_dir="${ZELLE}" \
    run_tag="${MODEL}_block${BLOCK}_nfe${NFE}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NFE}" \
    graph.sample_conformer="${CONF_FLAG}" \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hardware.accelerator=cpu \
    hardware.devices=1 \
    hydra.run.dir="${ZELLE}/hydra_s${SEED}" \
  > "${ZELLE}/sample_s${SEED}.log" 2>&1
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "[rank] FEHLGESCHLAGEN (rc=${RC}), letzte Zeilen:" >&2
    tail -20 "${ZELLE}/sample_s${SEED}.log" >&2
    exit "$RC"
fi

# Die Trajektorien wegwerfen, sofern nicht ausdruecklich gewuenscht.
if [ "${KEEP_PRED:-0}" != "1" ]; then
    weg=$(find "$ZELLE" -name "predictions.pt" -newer "$YAML" 2>/dev/null | wc -l)
    find "$ZELLE" -name "predictions.pt" -delete 2>/dev/null
    echo "[rank] ${weg} predictions.pt geloescht (KEEP_PRED=1 behaelt sie)"
fi

POSEN=$(find "$ZELLE" -name "*_seed${SEED}.sdf" 2>/dev/null | wc -l)
echo "[rank] fertig: ${POSEN} Posen fuer Block ${BLOCK}, Seed ${SEED}"
[ "$POSEN" -gt 0 ] || { echo "[rank] FEHLER: keine Posen erzeugt." >&2; exit 1; }
