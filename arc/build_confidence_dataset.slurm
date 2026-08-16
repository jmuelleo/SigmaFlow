#!/bin/bash -l
#
# EXP-105 Stufe 2: Posendatensatz mit Qualitaetslabels erzeugen.
#
# WOZU
#   Der integrierte Confidence-Head braucht Posen VARIIERENDER QUALITAET mit
#   bekanntem RMSD. Die gibt es erst, wenn ein Generator existiert -- waehrend
#   des generativen Trainings sieht das Netz nur Punkte auf dem
#   Wahrscheinlichkeitspfad, bei t=1 exakt die Kristallpose.
#
# SPLIT-TRENNUNG - der Punkt, an dem Leakage entsteht
#   Der Datensatz wird AUSSCHLIESSLICH aus dem TRAININGSsplit erzeugt
#   (pdbbind-general). PoseBusters ist Testsatz und wird hier nie angefasst.
#   Ein Head, der auf PoseBusters-Posen trainiert und dort evaluiert wuerde,
#   saehe beliebig gut aus und waere wertlos.
#
# Usage:
#   CKPT=/pfad/zu/last.ckpt MODEL=sigmaflow_minimal sbatch arc/build_confidence_dataset.slurm
#
#SBATCH --job-name=confdata
#SBATCH --partition=short
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
source "${REPO_ROOT}/arc/_common.sh"
setup_wandb_hardening

CKPT="${CKPT:?CKPT auf einen Generator-Checkpoint setzen}"
MODEL="${MODEL:-sigmaflow_minimal}"
K="${K:-10}"                       # Posen je Komplex
N_COMPLEXES="${N_COMPLEXES:-2000}" # aus dem TRAININGSsplit
TRAIN_SPLIT="${TRAIN_SPLIT:-pdbbind-general}"

if [ "$TRAIN_SPLIT" = "posebusters" ]; then
    echo "FEHLER: posebusters ist der TESTsatz. Ein darauf trainierter"
    echo "        Confidence-Head waere durch Leakage wertlos."
    exit 1
fi

CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

RUN_ID="CONFDATA_${MODEL}_K${K}"
write_run_metadata "$RUN_ID" "$CODE_DIR" "$PYTHON"
OUT="${RUN_DIR}/poses"
mkdir -p "$OUT"

echo "[confdata] Split=${TRAIN_SPLIT}  K=${K}  Komplexe=${N_COMPLEXES}"
echo "[confdata] Checkpoint: ${CKPT}"

cd "$CODE_DIR" || exit 1
for SEED in $(seq 0 $((K - 1))); do
    echo "--- Seed ${SEED} von $((K - 1)) ---"
    "$PYTHON" scripts/sample.py \
        ckpt="${CKPT}" \
        data_dir="${ARC_DATA}" \
        experiment="${TRAIN_SPLIT}" \
        output_dir="${OUT}" \
        run_tag="conf_${MODEL}_s${SEED}" \
        seed="${SEED}" \
        num_seeds=1 \
        ode.num_steps=25 \
        graph.sample_conformer=false \
        hydra.run.dir="${OUT}/hydra_s${SEED}" \
      > "${OUT}/sample_s${SEED}.log" 2>&1 \
      || echo "[confdata] Seed ${SEED} fehlgeschlagen, siehe Log"
done

echo "[confdata] Posen unter: ${OUT}"
echo "[confdata] Labels erzeugen:"
echo "  python -m SigmaFlow_Evaluation.evaluate_run --sampling_root ${OUT} \\"
echo "      --true_dir ${ARC_DATA}/${TRAIN_SPLIT} --label CONFDATA \\"
echo "      --out_json ${RUN_DIR}/labels.json"
