#!/bin/bash -l
#
# ZIEL-1-EXPERIMENT: Qualitaet gegen Zahl der Netzauswertungen (NFE),
# SigmaDock-Arm. Gegenstueck zu d_frame_fix/slurm/nfe_sweep_sigmaflow.sh.
# NUR GEMEINSAM AUSSAGEKRAEFTIG.
#
# WARUM DIESER LAUF FEHLT UND GEBRAUCHT WIRD
# SigmaFlow wurde bereits gesweept, SigmaDock nie. Ohne diesen Arm ist die
# Aussage "Flow Matching braucht weniger Integrationsschritte" NICHT belegbar -
# zulaessig waere nur "SigmaFlow braucht die von SigmaDock geerbten 25 nicht".
# Das ist ein wesentlich schwaecherer Satz, und die Differenz zwischen beiden
# ist genau der Unterschied zwischen einem zitierbaren Ergebnis und einer
# Beobachtung ueber die eigene Konfiguration.
#
# WICHTIG - andere Konfigurationsschluessel als bei SigmaFlow:
#   SigmaFlow: ode.num_steps        (deterministische ODE)
#   SigmaDock: diffusion.num_steps  (Reverse-Diffusion)
# Der Rest ist bewusst identisch gehalten.
#
# NOTE zu noise_scale: der Default ist 0.0 ("0 = no noise, better fidelity"),
# SigmaDocks Reverse-Prozess ist damit ebenfalls deterministisch. Die einzige
# Zufallsquelle ist auf beiden Seiten die Anfangsziehung. Das macht den
# Vergleich fair - nicht aendern.
#
# UMGEBUNG: myenv, NICHT sigmaflow_env. Unter sigmaflow_env loest
# `import sigmadock` auf das SigmaFlow-Paket auf. Dieser Fehler hat Job
# 8512799 5:44 Laufzeit gekostet.
#
# --array ...%1 = strikt seriell, damit die Wall-Clock interpretierbar ist.
#
# Usage (aus dem SigmaDock-Klon-Root):
#   CKPT_DIR=experiments/sigmadock/<timestamp>/checkpoints/last.ckpt \
#     sbatch slurm/nfe_sweep_sigmadock.sh
#
#SBATCH --job-name=sigmadock-nfe
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-20%1
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the SigmaDock 12h checkpoint}"
DATA_DIR="${DATA_DIR:-/data/stat-cadd/shug8458/data}"
BASE_OUT="${BASE_OUT:-$(pwd)/sampling_output_nfe_sweep}"

# IDENTISCH mit nfe_sweep_sigmaflow.sh - sonst ist der Vergleich hin.
STEP_LIST=(1 2 5 10 25 50 100)
N_STEPS_OPTS=${#STEP_LIST[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_STEPS=${STEP_LIST[$(( IDX % N_STEPS_OPTS ))]}
SEED=$(( IDX / N_STEPS_OPTS ))
OUTPUT_DIR="${BASE_OUT}/steps_${NUM_STEPS}"

echo "=== SigmaDock NFE-Sweep: ${NUM_STEPS} Schritte, Seed ${SEED} ==="
echo "START_EPOCH=$(date +%s)"
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
    experiment=posebusters \
    output_dir="${OUTPUT_DIR}" \
    run_tag=nfe_sweep \
    seed="${SEED}" \
    num_seeds=1 \
    diffusion.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hydra.run.dir="${OUTPUT_DIR}/hydra_out_seed${SEED}"

echo "END_EPOCH=$(date +%s)"
echo "Ran: SigmaDock, ${NUM_STEPS} Schritte, Seed ${SEED}"
