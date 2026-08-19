#!/bin/bash -l
#
# DATENSATZGROESSEN MESSEN -- CPU-only, kurzer Batch-Job.
#
# WARUM ALS JOB UND NICHT INTERAKTIV
#   pdbbind/general-set/ hat rund 19.000 Unterverzeichnisse. Der Scan ist reine
#   NFS-Latenz und dauert Minuten. Auf dem Login-Knoten belastet er die
#   gemeinsame Infrastruktur, und in einer interaktiven Sitzung wird er vom
#   Zeitlimit abgeschnitten -- genau das ist am 2026-08-19 mit Job 8606878
#   passiert, mitten im Lauf.
#
# ERGEBNIS
#   n_train ist eine der drei Zahlen, aus denen FINAL_MAX_EPOCHS folgt. Die
#   beiden anderen (samples/s je Arm) brauchen eine GPU, diese nicht. Sie kann
#   deshalb schon fertig sein, bevor die erste GPU frei wird.
#
# Usage:
#   sbatch arc/probe_sizes.slurm
#   ARM=sigmadock sbatch arc/probe_sizes.slurm     # gegen den SigmaDock-Klon
#
#SBATCH --job-name=probesize
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

ARM="${ARM:-sigmaflow_minimal}"

# Die Umgebung muss zum Baum passen. Ist myenv aktiv, loest `import sigmadock`
# aus site-packages auf den SigmaDock-Klon auf -- auch aus dem
# SigmaFlow_Minimal-Verzeichnis heraus (Job 8512799). --expect-in-path faengt
# das ab, statt still am falschen Code zu messen.
case "$ARM" in
  sigmaflow_minimal)
      CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal"
      EXPECT="SigmaFlow_Minimal"
      module load Mamba
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$ARC_SF_ENV"
      PYTHON="${ARC_SF_ENV}/bin/python"
      export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"
      ;;
  sigmadock)
      CODE_DIR="${ARC_SIGMADOCK}"
      EXPECT="SigmaDock_Reproduction"
      module load Mamba
      source activate "$ARC_SD_ENV"
      PYTHON="$(which python)"
      ;;
  *) echo "ARM unbekannt: $ARM"; exit 1 ;;
esac

cd "$CODE_DIR" || exit 1
write_run_metadata "PROBESIZE-${ARM}" "$CODE_DIR" "$PYTHON"

"$PYTHON" "${ARC_REPO}/arc/probe_datafront_size.py" \
    --data-dir "${ARC_DATA}" \
    --train pdbbind-general --val posebusters --test posebusters \
    --expect-in-path "${EXPECT}" \
    --json-out "${RUN_DIR}/datafront_sizes_${ARM}.json" \
    --env-out  "${RUN_DIR}/n_train_${ARM}.env" \
  2>&1 | tee "${RUN_DIR}/probe_sizes.log"

RC="${PIPESTATUS[0]}"
echo
echo "=============================================================="
echo "rc=${RC}"
echo "  ${RUN_DIR}/datafront_sizes_${ARM}.json"
echo "  ${RUN_DIR}/n_train_${ARM}.env"
echo
echo "  n_train geht in arc/calculate_final_epochs.py --n-train <...>"
echo "=============================================================="
exit "$RC"
