#!/bin/bash -l
#
# 12h pdbbind-general run of the FRAME-FIX variant. Second stage of the
# staged big-run plan, on top of the confirmed frame fix.
#
# WHY A FRESH RUN AND NOT A RESUME OF THE 6h CHECKPOINT
# The 6h run (job 8530243) COMPLETED - it reached max_epochs=3 and its LR
# schedule annealed all the way down. Resuming from it would restart at a
# floor learning rate, or need the schedule re-stretched, and the result
# would be neither a clean 12h run nor comparable to a fresh 12h SigmaDock
# baseline. Every earlier "resume for N more hours" stage in this project
# produced exactly that kind of muddled comparison. Fresh, correctly
# calibrated, comparable.
#
# LR SCHEDULE - CALIBRATED, DO NOT RAISE "FOR SAFETY"
# max_epochs feeds max_steps which feeds the LR scheduler. Setting it too
# high leaves the run stuck in warmup at 5-20% of peak LR; that silently
# ruined every pdbbind run before the LR fix (and the old 12h script in the
# variant folders still carries --max_epochs 1000 for exactly that reason).
#
# Measured on job 8530243 (same data, same GPU, same batch size):
#     3 epochs = 7,050 steps in 5:44:59 (incl. ~10 min dataset setup)
#     -> ~2,350 steps/epoch, ~1,260 steps/hour
# For a 12h budget:
#     6 epochs = 14,100 steps ~= 11:20 incl. setup -> ~40 min margin.
#     7 epochs would need ~13h and would be cut off by the walltime,
#     leaving the LR schedule unfinished.
#
# CONTROL GROUP / COMPARABILITY
# The natural comparison is the 6h frame-fix run 8530243 (does more compute
# help?). A SigmaDock 12h run under the FIXED LR schedule does NOT exist yet
# - the old SigmaDock 12h jobs predate the LR fix. Do not compare this run
# against the 6h SigmaDock (8512922) and call it a like-for-like result.
#
# SUCCESS CRITERION: the locality gap (bonded minus non-bonded relative
# rotation error) should improve on 8530243's -12.8 deg [CI -18.3, -7.2],
# ideally moving toward SigmaDock's -22.0 deg. Secondary: absolute fragment
# rotation error should finally drop below the 126.5 deg random baseline -
# it did NOT at 6h (128.1 deg), which is the main open question.
# Measure with SigmaFlow_Variants/posebusters_full_comparison/
# full_metrics.py and fragment_locality.py after sampling the full 209 set.
#
# Usage: sbatch slurm/train_pdbbind_general_12h_framefix.sh
#
#SBATCH --job-name=sigmaflow-framefix-12h
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# NOTE on the partition: short has a 12h limit, so a 12h job sits exactly on
# its boundary. medium (48h limit) avoids that edge case, and is what the
# earlier 12h runs used.
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# --- wandb robustness (job 8525523 died here after wasting 36 min) ---
# That run reached wandb.init() only AFTER building the 19k-complex datafront,
# then failed with:
#   ServicePollForTokenError: Failed to read port info after 30.0 seconds
# i.e. wandb's helper process could not publish its port file on the node's
# shared /tmp within the default 30 s timeout. Three mitigations:
#   1) raise that exact timeout,
#   2) keep wandb off the shared /tmp entirely,
#   3) probe wandb BEFORE the expensive dataset setup (see preflight below),
#      so a bad node costs one minute instead of thirty-six.
# Side effect worth knowing: TMPDIR now lives on NFS, so the DataLoader
# workers' cleanup at exit produces harmless
#   OSError: [Errno 16] Device or resource busy: '.nfsXXXX'
# tracebacks AFTER "`Trainer.fit` stopped". Those are noise, not failures.
export WANDB__SERVICE_WAIT=300
export TMPDIR="/data/stat-cadd/shug8458/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"

# Force this folder's own src/ ahead of sigmaflow_env's installed
# SigmaFlow_Development copy - without this, `import sigmadock` would
# silently load the UNMODIFIED code and this run would be a no-op.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "python:  $PYTHON"
echo "version: $($PYTHON --version)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

# Verify the frame fix is actually in the loaded file, not just in the folder.
# A partial upload can leave the path looking right while the code is old.
$PYTHON - <<'PYCHECK'
import inspect
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src, \
    "FRAME FIX MISSING in the loaded sigma_flow_generator.py - aborting."
print("frame fix verified in loaded module")
PYCHECK

# Preflight: fail fast if wandb cannot start its service on THIS node.
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

$PYTHON scripts/train.py \
    --data_dir /data/stat-cadd/shug8458/data \
    --train_exps pdbbind-general \
    --val_exps posebusters \
    --test_exps posebusters \
    --batch_size 8 \
    --num_workers 6 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 6 \
    --val_check_interval 50 \
    --early_stopping_patience 0 \
    --trans_score_weight 2.0 \
    --rot_score_weight 0.5 \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 12h pdbbind-general, FRAME FIX, max_epochs=6 (LR schedule calibrated to the 12h budget)"
