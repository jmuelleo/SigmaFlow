#!/bin/bash -l
#
# NACH DEM SAMPLING — Oracle@K und Symmetrie-Audit in einem Job.
#
# Beide Analysen brauchen nur die gesampelten SDFs und die Referenzliganden,
# keine GPU. Sie koennen auch lokal laufen, nachdem die SDFs per scp geholt
# wurden; auf ARC ist es nur bequemer, weil die Dateien schon dort liegen.
#
# AUFRUF
#   MODEL=sigmaflow_minimal NFE=25 sbatch arc/post_sampling_audits.slurm
#   MODEL=exp100            NFE=25 sbatch arc/post_sampling_audits.slurm
#   MODEL=sigmadock         NFE=25 sbatch arc/post_sampling_audits.slurm
#
#SBATCH --job-name=post-audits
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | exp100 | sigmadock}"
NFE="${NFE:-25}"
N_SEEDS="${N_SEEDS:-10}"
SAMPLE_ROOT="${ARC_RUNS}/sampling/${MODEL}_nfe${NFE}"
TRUE_DIR="${TRUE_DIR:?TRUE_DIR auf den Ordner mit <id>_ligands.sdf setzen}"

PYTHON="${ARC_SF_ENV}/bin/python"
module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

write_run_metadata "AUDIT-${MODEL}" "$ARC_REPO" "$PYTHON"

# Die Sampling-Ausgabe liegt unter results/posebusters/<model>/seed_<n>/.
# Der mittlere Ordnername steht erst zur Laufzeit fest, deshalb wird er gesucht.
PRED_PARENT="$(find "${SAMPLE_ROOT}/results" -maxdepth 3 -type d -name "seed_0" 2>/dev/null | head -1)"
if [ -z "$PRED_PARENT" ]; then
    echo "Keine seed_0-Ausgabe unter ${SAMPLE_ROOT}/results gefunden."
    find "${SAMPLE_ROOT}" -maxdepth 4 -type d | head -20
    exit 1
fi
PRED_GLOB="$(dirname "$PRED_PARENT")/seed_{seed}/*.sdf"
echo "Vorhersagen: ${PRED_GLOB}"

echo
echo "############ 1. Oracle@K"
$PYTHON "${ARC_REPO}/arc/oracle_k_audit.py" \
    --true_dir "${TRUE_DIR}" \
    --pred_glob "${PRED_GLOB}" \
    --n_seeds "${N_SEEDS}" \
    --label "${MODEL} (NFE ${NFE})" \
  2>&1 | tee "${RUN_DIR}/oracle_k_${MODEL}.log"

echo
echo "############ 2. Symmetrie-Audit"
# Das Skript erwartet, aus seinem eigenen Ordner zu laufen (relative Importe
# von ligand_reference). Deshalb dorthin wechseln und die Pfade uebergeben.
cd "${ARC_REPO}/SigmaFlow_Variants/posebusters_full_comparison"
TRUE_DIR="${TRUE_DIR}" PRED_SEED0="$(dirname "$PRED_PARENT")/seed_0" \
$PYTHON symmetry_flip_analysis.py 2>&1 | tee "${RUN_DIR}/symmetry_${MODEL}.log" || {
    echo "[!] Symmetrie-Audit uebersprungen - das Skript hat feste Pfade fuer den"
    echo "    12h-Vergleichslauf. Fuer neue Laeufe TRUE_DIR/PRED_GLOB oben im"
    echo "    Skript anpassen oder lokal nach dem scp fahren."
}

echo
echo "Ergebnisse: ${RUN_DIR}"
