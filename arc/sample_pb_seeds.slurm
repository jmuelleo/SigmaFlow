#!/bin/bash -l
#
# SAMPLING — volles PoseBusters, N Seeds, als Job-Array.
#
# Ein Skript fuer alle drei Modelle. Die Unterschiede stehen in Variablen,
# nicht in drei fast identischen Kopien - drei Kopien driften auseinander und
# genau daraus entstehen unfaire Vergleiche.
#
# AUFRUF (Beispiele, echte Pfade im Runbook):
#
#   MODEL=sigmaflow_minimal \
#   CKPT=/data/.../SigmaFlow_Minimal/experiments/sigmadock/<ts>/checkpoints/last.ckpt \
#   sbatch --array=0-9 arc/sample_pb_seeds.slurm
#
#   MODEL=exp100 CKPT=... sbatch --array=0-9 arc/sample_pb_seeds.slurm
#   MODEL=exp110 CKPT=... sbatch --array=0-9 arc/sample_pb_seeds.slurm
#   MODEL=sigmadock CKPT=... sbatch --array=0-9 arc/sample_pb_seeds.slurm
#
# NUM_STEPS steuert die NFE. Fuer den "controlled comparison" auf beiden Seiten
# denselben Wert setzen; fuer den "method-default comparison" jeweils den
# Vorgabewert der Methode. Nicht mischen (Runbook Abschnitt 8).
#
#SBATCH --job-name=sample-pb
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | exp100 | exp110 | sigmadock}"
CKPT="${CKPT:?CKPT auf den last.ckpt-Pfad setzen}"
NUM_STEPS="${NUM_STEPS:-25}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

# COMPARISON trennt die beiden wissenschaftlich verschiedenen Vergleiche
# (Runbook 5.1). Sie duerfen nicht in einen Topf:
#   controlled = gleiche NFE auf beiden Seiten, misst die Generatoren
#   default    = jede Methode mit ihrem Vorgabewert, misst die Gesamtsysteme
COMPARISON="${COMPARISON:-controlled}"

# RANKING trennt Generatorqualitaet von Rankingqualitaet (Runbook 5.2).
#   raw    = postprocessing.scoring=null, keine methodeneigene Auswahl
#   ranked = mit methodeneigenem Scoring
RANKING="${RANKING:-raw}"

case "$COMPARISON" in controlled|default) ;; *) echo "COMPARISON: controlled|default"; exit 1 ;; esac
case "$RANKING"    in raw|ranked)         ;; *) echo "RANKING: raw|ranked";           exit 1 ;; esac

# Der Ausgabepfad traegt ALLE unterscheidenden Merkmale. Ohne RANKING und
# COMPARISON darin wuerde ein Ranked-Lauf mit gleicher NFE die Raw-Posen
# ueberschreiben - ein stiller Datenverlust, der erst bei der Auswertung
# auffiele, und dann als Methodenunterschied fehlinterpretiert werden koennte.
OUT_ROOT="${ARC_RUNS}/sampling/${MODEL}__${COMPARISON}__${RANKING}__nfe${NUM_STEPS}"

case "$MODEL" in
  sigmaflow_minimal)
      CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal";            ENV="$ARC_SF_ENV"; KIND=flow ;;
  exp100)
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam"; ENV="$ARC_SF_ENV"; KIND=flow ;;
  exp110)
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field"; ENV="$ARC_SF_ENV"; KIND=flow ;;
  sigmadock)
      CODE_DIR="${ARC_SIGMADOCK}";                          ENV="$ARC_SD_ENV"; KIND=dock ;;
  *) echo "Unbekanntes MODEL: $MODEL"; exit 1 ;;
esac

module load Mamba
if [ "$KIND" = "dock" ]; then
    source activate "$ENV"; PYTHON="$(which python)"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV"
    PYTHON="${ENV}/bin/python"
fi

cd "$CODE_DIR"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

echo "model=${MODEL}  seed=${SEED}  nfe=${NUM_STEPS}"
echo "ckpt=${CKPT}"
$PYTHON -c "import sigmadock, os; print('sigmadock aus:', os.path.abspath(sigmadock.__file__))"

# Herkunftsgate: derselbe Fehler, der Job 8512799 nach 5:44 gekillt hat.
$PYTHON - <<PYCHECK
import os, sigmadock
p = os.path.abspath(sigmadock.__file__)
expect = "${CODE_DIR}"
assert os.path.commonpath([p, expect]) == expect, f"FALSCHE Quelle: {p}"
print("Herkunft verifiziert")
PYCHECK

mkdir -p "$OUT_ROOT"

# Provenienz neben die Posen legen. Ohne das ist ein halbes Jahr spaeter nicht
# mehr rekonstruierbar, welcher Checkpoint welche SDFs erzeugt hat.
{
    echo "model       = ${MODEL}"
    echo "comparison  = ${COMPARISON}"
    echo "ranking     = ${RANKING}"
    echo "nfe         = ${NUM_STEPS}"
    echo "seed        = ${SEED}"
    echo "checkpoint  = ${CKPT}"
    echo "code_dir    = ${CODE_DIR}"
    echo "array_job   = ${SLURM_ARRAY_JOB_ID:-n/a}_${SLURM_ARRAY_TASK_ID:-n/a}"
    echo "utc         = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT_ROOT}/sampling_metadata_seed${SEED}.txt"

if [ "$RANKING" = "raw" ]; then
    SCORING_ARGS=(postprocessing.scoring=null postprocessing.bust_config=null)
else
    # Methodeneigenes Scoring aktiv lassen (Vorgabe der Konfiguration).
    SCORING_ARGS=()
fi

# Die Schrittzahl heisst in beiden Welten anders. SigmaDock integriert KEINE
# ODE und hat gar keinen ode-Block; ein unbedingtes ode.num_steps laesst Hydra
# mit "Key 'ode' is not in struct" abbrechen -- so geschehen in Array 8629911,
# alle 30 Tasks nach rund 35 s. Der Wert bedeutet in beiden Faellen dasselbe:
# die Zahl der Integrationsschritte, also die NFE, und genau die soll im
# "controlled comparison" auf beiden Seiten gleich sein.
if [ "$KIND" = "dock" ]; then
    STEP_ARG=(diffusion.num_steps="${NUM_STEPS}")   # SigmaDock, conf/sampling/base.yaml
else
    STEP_ARG=(ode.num_steps="${NUM_STEPS}")         # SigmaFlow-Varianten
fi

$PYTHON scripts/sample.py \
    ckpt="${CKPT}" \
    data_dir="${ARC_DATA}" \
    experiment=posebusters \
    output_dir="${OUT_ROOT}" \
    run_tag="${MODEL}_${COMPARISON}_${RANKING}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    "${STEP_ARG[@]}" \
    graph.sample_conformer=false \
    "${SCORING_ARGS[@]}" \
    hydra.run.dir="${OUT_ROOT}/hydra_out_seed${SEED}"

echo "Ergebnisse: ${OUT_ROOT}/results/posebusters/<model>/seed_${SEED}/"
