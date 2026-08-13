#!/bin/bash -l
#
# EXP-100: Zustandsparametrisierung relativ zu einem inferenzverfuegbaren
# Referenzkonformer, gegen SigmaFlow-Minimal.
#
# EINZIGE INTERVENTION
#   Trainingsziel und Referenzgeometrie. Statt `R_1 = I` relativ zur
#   kristallorientierten Fragmentgeometrie wird der leakage-freie
#   ETKDGv3+MMFF-Konformer als Referenzrahmen benutzt und die echte
#   Kabsch-Zielrotation `R_1 != I` supervidiert.
#
# IDENTISCH ZU MINIMAL
#   Daten, Split, Seed, Architektur, Optimizer, Lernrate, Loss-Form,
#   Loss-Gewichte, Source-Verteilungen (N(0,I) bzw. Haar), Integrator, NFE,
#   Sampling-Schedule, Batchgroesse, Trainingsbudget.
#
# ERWARTUNG
#   EXP-100 muss NICHT besser sein. Die Frage ist, ob SigmaFlow in einer
#   inferenzsauberen Parametrisierung mit echter unbekannter Zielrotation
#   ueberhaupt trainierbar ist. Ein Gleichstand waere bereits das Ergebnis,
#   das EXP-102 (informative Source) freischaltet.
#
# VOR DEM START
#   1) `mkdir -p slurm_logs` in diesem Verzeichnis.
#   2) Der Minimal-Kontrolllauf muss mit IDENTISCHEM --seed und identischem
#      Budget gefahren werden - sonst ist der Vergleich wertlos:
#         cd ../../SigmaFlow_Minimal && sbatch slurm/train_pdbbind_general_12h_framefix.sh
#      Falls dieses Skript andere Flags hat als die unten stehenden, gilt:
#      die Flags ANGLEICHEN, nicht dieses Skript anpassen.
#   3) 12h passt in `short`. Fuer laengere Laeufe auf `medium` wechseln
#      (48h-Limit, bestaetigt 2026-08-05).
#
# Usage: sbatch slurm/train_exp100_vs_minimal_12h.sh
#
#SBATCH --job-name=exp100-state-reparam-12h
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

REPO=/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
cd "$REPO/SigmaFlow_FM_Specific/EXP-100_state_reparam"

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# Die Variante MUSS ihr eigenes src/ benutzen, nicht ein global installiertes
# sigmadock. Das hier ist die Zeile, die einen stillen Fehlvergleich verhindert.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
$PYTHON -c "import sigmadock, pathlib; p = pathlib.Path(sigmadock.__file__).resolve(); print('sigmadock aus:', p); assert 'EXP-100' in str(p), 'FALSCHE sigmadock-Quelle - Abbruch'"

# Sanity-Gate: die Reparametrisierung muss im geladenen Code wirklich aktiv sein.
$PYTHON -c "
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator as G
import inspect
assert hasattr(G, 'get_fragment_com_and_rot_reparam'), 'EXP-100-Methode fehlt'
assert 'R_ref' in inspect.signature(G._apply_transformations).parameters, 'R_ref-Umbenennung fehlt'
print('EXP-100 aktiv: Reparametrisierung und R_ref-Signatur vorhanden')
"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/data \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size 8 \
    --num_workers 6 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 1000 \
    --val_check_interval 50 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --seed 0 \
    --offline_run \
    --debug

echo
echo "EXP-100 fertig. Checkpoints unter experiments/sigmadock/<timestamp>/checkpoints/"
echo "Vergleich NUR gegen einen Minimal-Lauf mit identischem Seed und Budget."
