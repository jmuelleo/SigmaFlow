#!/bin/bash -l
#
# Rankervergleich auf IDENTISCHEM Posensatz.
#
# DIE REGEL, DIE DEN VERGLEICH TRAEGT
#   Alle Ranker sehen DIESELBEN K Posen. Wuerde jeder Ranker eigene Posen
#   bekommen, mischte sich Generatorstreuung in den Rankervergleich -- und die
#   Generatorstreuung ist in diesem Projekt gemessen rund ZEHNMAL so gross wie
#   die Methodenunterschiede. Ein Rankervergleich auf verschiedenen Posensaetzen
#   misst praktisch nur Rauschen.
#
# VIER ZAHLEN, IMMER GEMEINSAM
#   Single      eine Ziehung, ungerankt          -> Generatorqualitaet
#   Random@K    zufaellige Auswahl aus K         -> die ehrliche Nullhypothese
#   Top-1@K     der jeweilige Ranker             -> Rankerqualitaet
#   Oracle@K    beste der K Posen                -> Obergrenze bei perfektem Ranking
#   Ohne Oracle@K ist ein Rankergewinn nicht einzuordnen: er koennte gross
#   aussehen und trotzdem fast nichts von der verfuegbaren Luecke schliessen.
#
# Usage:
#   SAMPLING_ROOT=... MODEL=sigmaflow_minimal sbatch arc/compare_rankers.slurm
#
#SBATCH --job-name=rankcmp
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
source "${REPO_ROOT}/arc/_common.sh"

SAMPLING_ROOT="${SAMPLING_ROOT:?SAMPLING_ROOT auf das Sampling-Verzeichnis setzen}"
MODEL="${MODEL:?MODEL setzen}"
PB_CSV="${PB_CSV:-}"                # CSV mit vinardo + PB-Checks
CONF_SCORES="${CONF_SCORES:-}"      # JSON aus dem Confidence-Head

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"

RUN_ID="RANKCMP_${MODEL}"
write_run_metadata "$RUN_ID" "${ARC_REPO}" "$PYTHON"
cd "$REPO_ROOT" || exit 1

run_eval() {
    local label="$1"; shift
    echo; echo "======== ${label} ========"
    "$PYTHON" -m SigmaFlow_Evaluation.evaluate_run \
        --sampling_root "${SAMPLING_ROOT}" \
        --true_dir "${ARC_DATA}/posebusters" \
        --model "${MODEL}" \
        --label "${label}" \
        --out_json "${RUN_DIR}/eval_${label}.json" \
        "$@" 2>&1 | tail -40
}

# 1) Ohne Ranker: Single Sample und Oracle@K. Laeuft immer.
run_eval "raw"

# 2) SigmaDocks feste Heuristik. Braucht gnina-Werte in PB_CSV.
if [ -n "$PB_CSV" ] && [ -f "$PB_CSV" ]; then
    "$PYTHON" -m SigmaFlow_Evaluation.ranking.heuristic_score \
        --pb_csv "$PB_CSV" --out "${RUN_DIR}/scores_heuristic.json"
    run_eval "heuristic" --scores "${RUN_DIR}/scores_heuristic.json"
else
    echo
    echo "[rankcmp] PB_CSV fehlt -> feste Heuristik uebersprungen."
    echo "          Ohne gnina/Vinardo ist SigmaDocks Ranking nicht reproduzierbar."
fi

# 3) Integrierte Confidence, falls trainiert.
if [ -n "$CONF_SCORES" ] && [ -f "$CONF_SCORES" ]; then
    run_eval "integrated" --scores "$CONF_SCORES"
else
    echo
    echo "[rankcmp] CONF_SCORES fehlt -> integrierte Confidence uebersprungen."
fi

echo
echo "[rankcmp] Alle Ergebnisse: ${RUN_DIR}/eval_*.json"
echo "[rankcmp] Erinnerung: Single, Random@K, Top-1@K und Oracle@K gehoeren"
echo "          IMMER gemeinsam in den Bericht."
