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

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | exp100 | sigmadock}"
CKPT="${CKPT:?CKPT auf den last.ckpt-Pfad setzen}"
NUM_STEPS="${NUM_STEPS:-25}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUT_ROOT="${ARC_RUNS}/sampling/${MODEL}_nfe${NUM_STEPS}"

case "$MODEL" in
  sigmaflow_minimal)
      CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal";            ENV="$ARC_SF_ENV"; KIND=flow ;;
  exp100)
      CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam"; ENV="$ARC_SF_ENV"; KIND=flow ;;
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
$PYTHON scripts/sample.py \
    ckpt="${CKPT}" \
    data_dir="${ARC_DATA}" \
    experiment=posebusters \
    output_dir="${OUT_ROOT}" \
    run_tag="${MODEL}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUT_ROOT}/hydra_out_seed${SEED}"

# `postprocessing.scoring=null` ist Absicht: das ist die RAW/GENERATOR-Auswertung
# ohne methodeneigenes Ranking. Fuer die "ranked"-Variante dasselbe Skript mit
# gesetztem scoring erneut fahren und in ein ANDERES OUT_ROOT schreiben,
# damit beide Zahlen nebeneinander stehen (Runbook Abschnitt 8).

echo "Ergebnisse: ${OUT_ROOT}/results/posebusters/<model>/seed_${SEED}/"
