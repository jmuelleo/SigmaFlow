#!/bin/bash -l
#
# RUN-B24-SF — SigmaFlow Minimal, 24h Baseline
#
# Gegenstueck: arc/train_sigmadock_24h.slurm (RUN-B24-SD).
# Beide muessen zusammen ausgewertet werden; einzeln sind sie wertlos.
#
# BUDGETDEFINITION (wichtig, siehe Runbook Abschnitt 4)
#   Das Budget ist NICHT die Walltime, sondern `--max_epochs`. Der Wert speist
#   `max_steps = max_epochs * len(train_datafront) / batch_size` und damit den
#   LR-Scheduler (scripts/train.py:218). Die kalibrierte Reihe lautet:
#       6h  -> max_epochs 3   (~7.100 Schritte)
#      12h  -> max_epochs 6   (~14.250 Schritte)
#      24h  -> max_epochs 12  (~28.500 Schritte)   <- dieser Lauf
#   Die alten 24h-Laeufe (Jobs 8465054/8465055) benutzten `--max_epochs 1000`.
#   Der LR-Zeitplan war damit auf 2,4 Mio Schritte ausgelegt, gelaufen sind ~28k,
#   die Lernrate blieb also praktisch konstant. Genau deshalb sind sie keine
#   gueltige Baseline.
#
# Walltime 24h ueberschreitet das 12h-Limit von `short`, deshalb `medium`.
#
# Usage: sbatch arc/train_sigmaflow_minimal_24h.slurm
#
#SBATCH --job-name=B24-SF-minimal
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

CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
PYTHON="${ARC_SF_ENV}/bin/python"
SEED=0

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR"

# Das eigene src/ MUSS vor der in sigmaflow_env installierten Kopie von
# SigmaFlow_Development stehen. Ohne diese Zeile laedt `import sigmadock`
# still den falschen Code und der Lauf ist wertlos.
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

echo "python:  $PYTHON"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# ---------------------------------------------------------------- SANITY GATE
# Drei Dinge, die einzeln schon einen 24h-Lauf entwerten wuerden.
$PYTHON - <<'PYCHECK'
import inspect, os, sys
import sigmadock
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator

path = os.path.abspath(sigmadock.__file__)
print("sigmadock geladen aus:", path)

# 1) Wirklich SigmaFlow_Minimal - nicht Development, Variants, archive oder EXP-100.
assert "SigmaFlow_Minimal" in path, f"FALSCHE Quelle: {path}"
for forbidden in ("EXP-100", "SigmaFlow_Development", "SigmaFlow_Variants", "archive"):
    assert forbidden not in path, f"FALSCHE Quelle ({forbidden}): {path}"

# 2) Der Frame-Fix muss im GELADENEN Modul stehen, nicht nur im Ordner.
src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src, \
    "FRAME FIX FEHLT im geladenen sigma_flow_generator.py"
print("frame fix verified")

# 3) EXP-100 darf hier NICHT aktiv sein - das waere die Variante, nicht die Baseline.
assert not hasattr(SigmaFlowGenerator, "get_fragment_com_and_rot_reparam"), \
    "EXP-100-Code im Minimal-Lauf gefunden - das ist die falsche Codebasis"
print("EXP-100 nicht aktiv (korrekt fuer die Baseline)")
PYCHECK

write_run_metadata "RUN-B24-SF" "$CODE_DIR" "$PYTHON"
echo "seed = ${SEED}" >> "${RUN_DIR}/run_metadata.txt"
echo "budget = max_epochs 12 (24h-kalibriert)" >> "${RUN_DIR}/run_metadata.txt"

# Preflight: kostet eine Minute und erspart im schlechten Fall den Verlust des
# gesamten Laufs erst nach dem ~10-35 min dauernden Datafront-Aufbau.
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

MARKER="${TMPDIR}/start_marker"; touch "$MARKER"

# ---------------------------------------------------------------- TRAINING
# Jedes Flag hier hat ein exaktes Gegenstueck in train_sigmadock_24h.slurm.
# Abweichungen sind dort einzeln als methodenspezifisch markiert.
$PYTHON scripts/train.py \
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
    --seed ${SEED} \
    --offline_run \
    --debug \
  2>&1 | tee "${RUN_DIR}/train_stdout.log"

record_experiment_dir "$CODE_DIR" "$MARKER"

echo
echo "RUN-B24-SF fertig. Metadaten und Log: ${RUN_DIR}"
echo "Gegenstueck fuer den Vergleich: RUN-B24-SD (arc/train_sigmadock_24h.slurm)"
