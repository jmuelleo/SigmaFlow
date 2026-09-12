#!/bin/bash -l
#
# CPU-SONDIERUNG FUERS SAMPLING
#
# Frage: laeuft das Sampling ohne GPU, und wie schnell? GPU-Knoten sind bei
# reduzierter Kapazitaet knapp, CPU-Knoten nicht. Wenn der Faktor ertraeglich
# ist, wird die spaetere Auswertung auf den offiziellen 308 Komplexen (rund
# 44 GPU-h je Modell) planbar, statt ans Ende geschoben zu werden.
#
# BAUART: kein --gres, sonst identisch zum etablierten Pfad aus
# arc/sample_pb_seeds.slurm. Begrenzt wird ueber `timeout`; gezaehlt wird,
# wie viele Posen in der Zeit entstanden sind. Daraus faellt die Rate ab,
# ohne dass der Lauf fertig werden muss.
#
# NICHT mit srun aus einer laufenden interaktiven Zuteilung starten -- eine
# verschachtelte Anforderung scheitert an "More processors requested than
# permitted", und die folgenden Zeilen laufen dann versehentlich auf dem
# Login-Knoten weiter.
#
# Usage:
#   MODEL=exp110 CKPT=/pfad/zu/last.ckpt sbatch arc/sample_cpu_probe.slurm
#
#SBATCH --job-name=sample-cpu-probe
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
setup_wandb_hardening

MODEL="${MODEL:?MODEL setzen, z.B. exp110}"
CKPT="${CKPT:?CKPT auf den last.ckpt-Pfad setzen}"
BUDGET_S="${BUDGET_S:-2400}"          # reine Rechenzeit, ohne Aufbau
NUM_STEPS="${NUM_STEPS:-25}"

case "$MODEL" in
  sigmaflow_minimal) CODE_DIR="${ARC_REPO}/SigmaFlow_Minimal" ;;
  exp100)            CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam" ;;
  exp110)            CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field" ;;
  *) echo "Unbekanntes MODEL: $MODEL"; exit 1 ;;
esac

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"

cd "$CODE_DIR"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

# RUN_DIR wird von write_run_metadata gesetzt, nicht von _common.sh selbst.
# Ohne diesen Aufruf waere RUN_DIR unter set -u undefiniert und der Job
# stuerbe sofort.
write_run_metadata "SAMPLE-CPU-PROBE-${MODEL}" "$CODE_DIR" "$PYTHON"

OUT="${RUN_DIR}/cpu_probe"
mkdir -p "$OUT"

echo "=============================================================="
echo "  model      : ${MODEL}"
echo "  code_dir   : ${CODE_DIR}"
echo "  ckpt       : ${CKPT}"
echo "  cpus       : ${SLURM_CPUS_PER_TASK:-?}"
echo "  budget     : ${BUDGET_S} s reine Laufzeit"
echo "  nfe        : ${NUM_STEPS}"
echo "=============================================================="
$PYTHON -c "import torch; print('  torch:', torch.__version__, '| cuda verfuegbar:', torch.cuda.is_available())"
$PYTHON -c "import os, sigmadock; print('  sigmadock aus:', os.path.abspath(sigmadock.__file__))"

# Herkunftsgate, wie im GPU-Sampler.
$PYTHON - <<PYCHECK
import os, sigmadock
p = os.path.abspath(sigmadock.__file__)
assert os.path.commonpath([p, "${CODE_DIR}"]) == "${CODE_DIR}", f"FALSCHE Quelle: {p}"
print("  Herkunft verifiziert")
PYCHECK

echo
echo "--- Sampling auf CPU, harte Zeitgrenze ${BUDGET_S} s ---"
T0=$(date +%s)
set +e
timeout "$BUDGET_S" "$PYTHON" scripts/sample.py \
    ckpt="${CKPT}" \
    data_dir="${ARC_DATA}" \
    experiment=posebusters \
    output_dir="${OUT}" \
    run_tag="cpuprobe" \
    seed=0 \
    num_seeds=1 \
    ode.num_steps="${NUM_STEPS}" \
    graph.sample_conformer=false \
    hardware.accelerator=cpu \
    hardware.devices=1 \
    hydra.run.dir="${OUT}/hydra" \
  > "${RUN_DIR}/cpu_probe.log" 2>&1
RC=$?
set -e
T1=$(date +%s)
WALL=$(( T1 - T0 ))

echo "  rc=${RC} $([ "$RC" -eq 124 ] && echo '(timeout, hier ERWARTET)')"
echo "  Laufzeit: ${WALL} s"
echo
echo "--- letzte Logzeilen ---"
tail -20 "${RUN_DIR}/cpu_probe.log"

N=$(find "$OUT" -name "*.sdf" 2>/dev/null | wc -l)
echo
echo "=============================================================="
echo "ERGEBNIS"
echo "=============================================================="
echo "  erzeugte Posen : ${N}"
if [ "$N" -gt 0 ]; then
    "$PYTHON" - "$N" "$WALL" <<'PYRATE'
import sys
n, wall = int(sys.argv[1]), int(sys.argv[2])
# Der Aufbau der Datafront steckt mit drin, die Rate ist also eine UNTERE
# Schranke; der eingeschwungene Durchsatz liegt hoeher.
print(f"  untere Schranke: {n/wall*3600:.0f} Komplexe/h")
print(f"  -> 209 Komplexe (ein Seed) hoechstens {209/(n/wall)/3600:.2f} h")
print(f"  -> 10 Seeds seriell hoechstens {10*209/(n/wall)/3600:.1f} h")
print(f"  -> 10 Seeds als Array parallel hoechstens {209/(n/wall)/3600:.2f} h Wall-clock")
print()
print("  Vergleich GPU: der Sampling-Job hat 3 h Limit je Seed.")
print("  Ist die CPU-Zahl oben deutlich darunter, lohnt der CPU-Weg.")
PYRATE
else
    echo "  KEINE Posen erzeugt -- Log oben ansehen."
    echo "  Haeufigste Ursachen: Checkpoint nicht ladbar, Datensatz nicht"
    echo "  gefunden, oder der Aufbau der Datafront hat das Budget gefressen."
fi
echo "  Logs: ${RUN_DIR}"
