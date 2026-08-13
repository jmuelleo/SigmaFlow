#!/bin/bash -l
#
# SigmaDock auf denselben 10 Dummy-Komplexen - Gegenstueck zu
# d_frame_fix/slurm/sample_dummy_compare.sh. Nur gemeinsam aussagekraeftig.
#
# ⚠️ Derselbe Vorbehalt: die 10 Dummy-Komplexe sind mit hoher Wahrscheinlichkeit
# Trainingsdaten (pdbbind-general) und wurden fuer Overfit-Laeufe benutzt.
# KEIN Testset - ausschliesslich zur visuellen Kontrolle in PyMOL.
#
# UMGEBUNG: SigmaDock laeuft in myenv, NICHT in sigmaflow_env. Unter
# sigmaflow_env loest `import sigmadock` auf das SigmaFlow-Paket auf und
# stirbt mit ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'.
# Dieser Fehler hat Job 8512799 5:44 Laufzeit gekostet.
#
# WICHTIG: Damit die Dateinamen zu SigmaFlow passen, MUSS hier dieselbe
# Ligandendatei gelesen werden. dummy_train.yaml benutzt
# sdf_regex=".*_ligand\.sdf$" (SINGULAR, die kanonische Kopie). Wird das
# ueberschrieben, vergleicht man hinterher verschiedene Molekuele -
# genau der Fehler, der am 2026-08-13 die "42 Ausreisser" erzeugt hat.
#
# Usage (aus dem SigmaDock-Klon-Root):
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/sample_dummy_compare_sigmadock.sh
#
#SBATCH --job-name=sigmadock-dummy-cmp
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the SigmaDock 12h checkpoint}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix/notebooks}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sampling_output_dummy_compare}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "SigmaDock, Dummy-Set, Seed ${SEED}"
python -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"
python - <<'PYCHECK'
import sigmadock.diff.denoiser  # noqa: F401 - Praesenz ist die Assertion
import sigmadock, os
assert "SigmaDock_Reproduction" in os.path.abspath(sigmadock.__file__), \
    f"WRONG sigmadock package loaded: {sigmadock.__file__} - aborting."
print("original SigmaDock (diffusion denoiser) verified")
PYCHECK

python scripts/sample.py \
    ckpt="${CKPT_DIR}" \
    data_dir="${DATA_DIR}" \
    experiment=dummy_train \
    output_dir="${OUTPUT_DIR}" \
    run_tag=dummy_compare \
    seed="${SEED}" \
    num_seeds=1 \
    diffusion.num_steps=25 \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "Results: ${OUTPUT_DIR}/results/dummy_train/<model>/seed_${SEED}/"
echo "ERINNERUNG: Trainingsdaten-Ueberlappung - nur zum Ansehen, nicht als Benchmark."
