#!/bin/bash -l
#
# 12h pdbbind-general CONTROL run for the ORIGINAL SigmaDock (diffusion),
# with the CORRECTED learning-rate schedule. Counterpart to
# SigmaFlow_Variants/d_frame_fix/slurm/train_pdbbind_general_12h_framefix.sh
# - the two together are the clean 12h comparison.
#
# To be placed in the separate ARC SigmaDock clone
# (/data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock);
# no local copy of that repo exists here, see STATUS.md PAUSE-PUNKT #9.
#
# WHY THIS RUN EXISTS
# No SigmaDock 12h run under the FIXED LR schedule exists yet - the older
# SigmaDock 12h jobs all predate the LR fix and sat inside warmup at 5-20%
# of peak LR. Comparing the new SigmaFlow 12h run against them, or against
# the 6h SigmaDock (8512922), would confound method, compute and schedule.
#
# COMPUTE MATCHING - THE POINT OF max_epochs=6
# Steps per epoch follow from dataset size and batch size, which are
# identical for both methods (pdbbind-general, batch 8) -> ~2,350 steps per
# epoch. max_epochs=6 therefore gives BOTH methods exactly 14,100 optimizer
# steps. That, not the wall-clock, is what makes the comparison fair.
# Measured reference points, both COMPLETED before their 6h walltime:
#     SigmaFlow frame fix (8530243): 3 epochs = 7,050 steps in 5:44:59
#     SigmaDock           (8512922): 3 epochs             in 5:29:23
# -> 6 epochs land at roughly 11:00 for SigmaDock, inside the 12h budget.
#
# DO NOT raise max_epochs "for safety". max_epochs feeds max_steps which
# feeds the LR scheduler (warmup over lr_warmup_frac=1/16 of max_steps, then
# cosine annealing). The old --max_epochs 1000 is exactly the bug that
# invalidated every earlier pdbbind run, for SigmaDock as much as for
# SigmaFlow - both use the same train.py and the same config defaults.
#
# NOTE --rot_score_method / --rot_score_scaling are REAL, live flags in
# SigmaDock (verified against the known-good train_pdbbind_general_24h.sh in
# this clone), unlike SigmaFlow's removed --rot_vector_field_scaling.
# Do not "harmonise" these away.
#
# Usage (from the SigmaDock clone root):
#   sbatch slurm/train_pdbbind_general_12h_lrfix_sigmadock.sh
#
#SBATCH --job-name=sigmadock-pdbbind-12h-lrfix
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# NOTE on the partition: short has a 12h limit, so a 12h job sits exactly on
# its boundary. medium (48h limit) avoids that edge case. Same choice as the
# SigmaFlow 12h counterpart.
#
# IMPORTANT: slurm_logs/ must already exist before you run `sbatch` on this
# script (create it once with `mkdir -p slurm_logs` from this directory).

set -euo pipefail

# ENVIRONMENT: SigmaDock runs in myenv, NOT sigmaflow_env, activated with
# `source activate` and using the plain `python` on PATH. This is NOT
# cosmetic: sigmaflow_env has SigmaFlow_Development installed into its
# site-packages, so under that env `import sigmadock` resolves to the
# SigmaFlow package and dies with
#   ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'
# (SigmaFlow has sigma_flow_generator.py, not denoiser.py). That exact
# mistake killed job 8512799 after 5:44. This block is copied verbatim from
# the known-good 6h control - do not "modernise" it to match the SigmaFlow
# scripts.
module load Mamba
source activate /data/stat-cadd/shug8458/myenv

cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock

# --- wandb robustness, mirrored from the SigmaFlow 12h script ---
# Job 8525523 lost 36 minutes to
#   ServicePollForTokenError: Failed to read port info after 30.0 seconds
# because wandb's helper process could not publish its port file on the
# node's shared /tmp in time. train.py calls wandb.init() only AFTER building
# the 19k-complex datafront, so such a failure is expensive.
#
# Deliberate deviation from the 6h control (8512922), which did NOT have
# this: at 12h the downside of losing a run is twice as large, and the
# SigmaFlow 12h counterpart carries the same three mitigations - keeping
# both sides identical here is the more symmetric choice. None of this
# touches the model: it only moves temp files and raises one timeout.
# Side effect: TMPDIR now lives on NFS, so DataLoader workers produce
# harmless "OSError: [Errno 16] Device or resource busy: '.nfsXXXX'"
# tracebacks AFTER "`Trainer.fit` stopped". Noise, not failures.
export WANDB__SERVICE_WAIT=300
export TMPDIR="/data/stat-cadd/shug8458/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"

echo "python:  $(which python)"
echo "version: $(python --version)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
python -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

# Fail fast if the WRONG sigmadock package got picked up. The diffusion
# denoiser exists only in the original SigmaDock; SigmaFlow replaced it with
# sigma_flow_generator.py. This is the exact failure mode that killed job
# 8512799 - there it surfaced 5:44 into the run instead of immediately.
python - <<'PYCHECK'
import sigmadock.diff.denoiser  # noqa: F401  - presence is the assertion
import sigmadock, os
assert "SigmaDock_Reproduction" in os.path.abspath(sigmadock.__file__), \
    f"WRONG sigmadock package loaded: {sigmadock.__file__} - aborting."
print("original SigmaDock (diffusion denoiser) verified")
PYCHECK

# Preflight: fail fast if wandb cannot start its service on THIS node.
# Costs ~1 min; without it a bad node is only discovered after the ~10-35 min
# datafront build.
python -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

python scripts/train.py \
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
    --rot_score_method space \
    --rot_score_scaling rms \
    --offline_run \
    --debug

echo "Checkpoints under: experiments/sigmadock/<timestamp>/checkpoints/ (see stdout for the exact timestamp)"
echo "Ran: 12h pdbbind-general SigmaDock CONTROL, LR-FIXED (max_epochs=6, 14,100 steps - matched to the SigmaFlow 12h run)"
