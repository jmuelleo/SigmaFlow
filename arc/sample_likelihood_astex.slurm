#!/bin/bash -l
#
# LOG-DICHTE JE POSE AUF ASTEX -- CPU-only
#
# WOZU
#   Abschnitt 9.3.2 der Arbeit: die Posen nach der Modell-Log-Dichte ranken
#   statt nach der Paper-Heuristik, auf DEMSELBEN Posensatz. Weil das Sampling
#   eine deterministische ODE integriert, ist das Modell ein Continuous
#   Normalizing Flow, und die Dichte faellt entlang der Vorwaertstrajektorie
#   ab -- ohne Rueckwaertsintegration, weil man x_0 selbst gezogen hat.
#
# WARUM NICHT scripts/sample.py, SONDERN run_likelihood.py
#   Das Skript tauscht NUR den Sampler aus. Datensatz, Fragmentierung,
#   Checkpoint und SDF-Export laufen unveraendert, also sind die Posen bei
#   gleichem Seed bitgleich dieselben wie in astex_redock/. Nur so ist der
#   Vergleich gegen die Heuristik der, den 9.3.2 verlangt.
#
# WARUM data.batch_size=1
#   Die exakte Divergenz buendelt 6F vjps in einen Autograd-Aufruf und haelt
#   dafuer den Vorwaertsgraphen im Speicher. Mehrere Komplexe gleichzeitig
#   sprengen das.
#
# WARUM ode.num_steps UND NICHT diffusion.num_steps
#   `diffusion` ist der SigmaDock-Name. Beide SigmaFlow-Arme haben die Gruppe
#   in conf/sampling/base.yaml:60 nach `ode` umbenannt; der alte Name faellt
#   entweder bei Hydra durch oder sampelt still mit der Vorgabe 25.
#
# Aufruf:
#   LIST=1 CKPT=<pfad> bash arc/sample_likelihood_astex.slurm
#   CKPT=<pfad> sbatch --array=0-39 arc/sample_likelihood_astex.slurm
#
#SBATCH --job-name=logp-astex
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -uo pipefail

# SLURM fuehrt eine KOPIE dieses Skripts in /var/spool/slurmd/jobNNN/ aus.
# ${BASH_SOURCE[0]} zeigt deshalb nicht ins Repo -- absoluter Pfad, wie in
# allen anderen arc/*.slurm. (Daran starb Array 8691800 vollstaendig.)
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:-sigmaflow_twohead}"
CKPT="${CKPT:?CKPT auf den Snapshot setzen, der die vorhandenen Astex-Posen erzeugt hat}"
NFE="${NFE:-5}"
EXPERIMENT="${EXPERIMENT:-astex}"
CONF_FLAG="${CONF_FLAG:-true}"          # sampled, wie bei den Astex-Laeufen
OUT_ROOT="${OUT_ROOT:-${ARC_RUNS}/logp_astex_${MODEL}__nfe${NFE}}"

case "$MODEL" in
  sigmaflow_minimal) CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal" ;;
  sigmaflow_twohead) CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field" ;;
  *) echo "MODEL nicht vorgesehen: $MODEL (nur die Flow-Arme haben eine Dichte)" >&2; exit 1 ;;
esac

if [ "${LIST:-0}" = "1" ]; then
    echo "Modell      : $MODEL"
    echo "Checkpoint  : $CKPT   $([ -f "$CKPT" ] && echo vorhanden || echo FEHLT)"
    echo "Codebaum    : $CODE_DIR"
    echo "Experiment  : $EXPERIMENT   NFE: $NFE   Konformer: $CONF_FLAG"
    echo "Ausgabe     : $OUT_ROOT"
    echo "Runner      : ${REPO_ROOT}/SigmaFlow_Variants/likelihood/run_likelihood.py"
    ls -l "${REPO_ROOT}/SigmaFlow_Variants/likelihood/"*.py 2>/dev/null
    exit 0
fi

[ -f "$CKPT" ] || { echo "FEHLER: CKPT fehlt: $CKPT" >&2; exit 1; }
SEED="${SLURM_ARRAY_TASK_ID:?nur als Array-Job sinnvoll}"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"
cd "$CODE_DIR" || exit 1

echo "[logp] Seed $SEED, NFE $NFE, Modell $MODEL"
echo "[logp] Ausgabe $OUT_ROOT"

CODE_DIR="$CODE_DIR" \
LOGP_CSV="${OUT_ROOT}/logp_s${SEED}.csv" \
"$PYTHON" "${REPO_ROOT}/SigmaFlow_Variants/likelihood/run_likelihood.py" \
    ckpt="$CKPT" \
    data_dir="${ARC_DATA}" \
    experiment="${EXPERIMENT}" \
    output_dir="${OUT_ROOT}" \
    run_tag="logp_${MODEL}_nfe${NFE}_s${SEED}" \
    seed="${SEED}" \
    num_seeds=1 \
    ode.num_steps="${NFE}" \
    graph.sample_conformer="${CONF_FLAG}" \
    postprocessing.scoring=null \
    postprocessing.bust_config=null \
    hardware.accelerator=cpu \
    hardware.devices=1 \
    data.batch_size=1 \
    hydra.run.dir="${OUT_ROOT}/hydra_s${SEED}" \
  > "${OUT_ROOT}/logp_s${SEED}.log" 2>&1
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "[logp] FEHLGESCHLAGEN (rc=${RC}), letzte Zeilen:" >&2
    tail -30 "${OUT_ROOT}/logp_s${SEED}.log" >&2
    exit "$RC"
fi

CSV="${OUT_ROOT}/logp_s${SEED}.csv"
N=$(($(wc -l < "$CSV" 2>/dev/null || echo 1) - 1))
echo "[logp] fertig: ${N} Dichten in ${CSV}"
# SLURM COMPLETED ist kein Beweis, dass etwas entstanden ist.
[ "$N" -gt 0 ] || { echo "[logp] FEHLER: keine Dichten geschrieben." >&2; exit 1; }
