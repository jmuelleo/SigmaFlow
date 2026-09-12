#!/bin/bash -l
#
# RUN-B24-SD — SigmaDock (Original-Diffusion), 24h Baseline
#
# Gegenstueck: arc/train_sigmaflow_minimal_24h.slurm (RUN-B24-SF).
#
# UMGEBUNG — NICHT "modernisieren"
#   SigmaDock laeuft in `myenv`, NICHT in `sigmaflow_env`, und wird mit
#   `source activate` aktiviert. Das ist kein Schoenheitsfehler: sigmaflow_env
#   hat SigmaFlow_Development in seine site-packages installiert, dort loest
#   `import sigmadock` auf das SigmaFlow-Paket auf und stirbt mit
#       ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'
#   (SigmaFlow hat sigma_flow_generator.py, kein denoiser.py). Genau dieser
#   Fehler hat Job 8512799 nach 5:44 gekillt. Der Block unten ist wortgleich
#   aus der bekannten guten 12h-Kontrolle uebernommen.
#
# BUDGETDEFINITION
#   Identisch zur SigmaFlow-Seite: `--max_epochs 12` (24h-kalibriert),
#   nicht die Walltime. Siehe Kommentar im SigmaFlow-Skript.
#
# Usage: sbatch arc/train_sigmadock_24h.slurm
#
#SBATCH --job-name=B24-SD-control
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
setup_wandb_hardening

CODE_DIR="${ARC_SIGMADOCK}"
SEED=0

module load Mamba
source activate "$ARC_SD_ENV"        # bewusst `source activate`, siehe oben

cd "$CODE_DIR"
PYTHON="$(which python)"

echo "python:  $PYTHON"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# ---------------------------------------------------------------- SANITY GATE
python - <<'PYCHECK'
import os
import sigmadock
import sigmadock.diff.denoiser  # noqa: F401  - blosse Existenz ist die Zusicherung

path = os.path.abspath(sigmadock.__file__)
print("sigmadock geladen aus:", path)
assert "SigmaDock_Reproduction" in path, f"FALSCHES sigmadock-Paket: {path}"
for forbidden in ("SigmaFlow_Minimal", "EXP-100", "SigmaFlow_Development", "SigmaFlow_Variants"):
    assert forbidden not in path, f"FALSCHE Quelle ({forbidden}): {path}"
print("Original-SigmaDock (Diffusions-Denoiser) verifiziert")
PYCHECK

write_run_metadata "RUN-B24-SD" "$CODE_DIR" "$PYTHON"
echo "seed = ${SEED}" >> "${RUN_DIR}/run_metadata.txt"
echo "budget = max_epochs 12 (24h-kalibriert, identisch zu RUN-B24-SF)" >> "${RUN_DIR}/run_metadata.txt"

python -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

MARKER="${TMPDIR}/start_marker"; touch "$MARKER"

# ---------------------------------------------------------------- TRAINING
# Bis auf die beiden letzten Flags identisch zur SigmaFlow-Seite.
#   --rot_score_method space  und  --rot_score_scaling rms
# sind METHODENSPEZIFISCH: sie steuern SigmaDocks Score-Parametrisierung auf
# SO(3) und haben in einem Flow-Matching-Modell kein Gegenstueck. Sie sind
# Teil der bekannten guten SigmaDock-Konfiguration und werden mitgefuehrt,
# damit die Referenz ihre eigene beste Einstellung behaelt.
python scripts/train.py \
    --data_dir "${ARC_DATA}" \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size 8 \
    --num_workers 6 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 12 \
    --val_check_interval 50 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --rot_score_method space \
    --rot_score_scaling rms \
    --seed ${SEED} \
    --offline_run \
    --debug \
  2>&1 | tee "${RUN_DIR}/train_stdout.log"

record_experiment_dir "$CODE_DIR" "$MARKER"

echo
echo "RUN-B24-SD fertig. Metadaten und Log: ${RUN_DIR}"
echo "Gegenstueck fuer den Vergleich: RUN-B24-SF (arc/train_sigmaflow_minimal_24h.slurm)"
