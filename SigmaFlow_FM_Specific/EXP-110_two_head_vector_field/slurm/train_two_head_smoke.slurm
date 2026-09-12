#!/bin/bash -l
#
# EXP-110 SMOKE — kurzer echter GPU-Lauf, bevor 12 GPU-Stunden riskiert werden.
#
# EXP-110 ist lokal auf CPU vollstaendig validiert (sechs Auditskripte, 167
# Checks, plus ein vollstaendiger Produktionspfadlauf aus einem frischen
# Worktree). Was lokal NICHT geprueft werden konnte:
#   * laeuft es auf der GPU,
#   * findet es die ECHTEN Datensaetze auf ARC statt der 14 Dummy-Komplexe,
#   * vertraegt der Dataloader 6 Worker im Compute-Job,
#   * wie viel VRAM der zweite Kopf tatsaechlich kostet,
#   * wie schnell ein Schritt wirklich ist, also ob 6 Epochen in 12 h passen.
# Genau das misst dieser Lauf.
#
# BAUART: modelliert nach arc/train_exp100_sanity.slurm, dem etablierten
# Smoke-Pfad dieses Projekts. Der Trainingsaufruf ist ZEICHENGLEICH mit dem
# aus slurm/train_two_head_12h.slurm; begrenzt wird ueber `timeout`, nicht
# ueber geaenderte Trainer-Limits. Es wird also wirklich dieselbe
# Konfiguration gefahren, nur kuerzer.
#
# Laufzeit rund 40 min, passt in `short`.
#
# Usage: sbatch slurm/train_two_head_smoke.slurm
#
#SBATCH --job-name=twohead-smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"
setup_wandb_hardening

CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field"
PYTHON="${ARC_SF_ENV}/bin/python"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

# ============================================================ 0. FINGERABDRUCK
echo "=============================================================="
echo "0. RUNTIME-FINGERABDRUCK"
echo "=============================================================="
echo "  cwd            : $(pwd)"
echo "  git HEAD       : $(cd "$REPO_ROOT" && git rev-parse HEAD)"
echo "  git kurz       : $(cd "$REPO_ROOT" && git log -1 --oneline)"
echo "  git status     : $(cd "$REPO_ROOT" && git status --short | wc -l) geaenderte Dateien"
echo "  python         : $PYTHON"
echo "  version        : $($PYTHON --version 2>&1)"
echo "  PYTHONPATH     : ${PYTHONPATH}"
echo "  node           : $(hostname)"
$PYTHON - <<'PYENV'
import torch
print("  torch          :", torch.__version__)
print("  torch.cuda     :", torch.version.cuda)
print("  cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU            : {p.name}, {p.total_memory/1e9:.1f} GB")
PYENV
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ============================================================ 1. IMPORTFALLEN
# Welche Datei wird fuer jede zentrale Klasse tatsaechlich geladen? Ein
# global installiertes sigmadock in site-packages wuerde hier sofort auffallen.
echo
echo "=============================================================="
echo "1. IMPORTHERKUNFT — wird wirklich EXP-110 trainiert?"
echo "=============================================================="
$PYTHON - <<'PYIMP'
import os
import sys

WANT = "EXP-110_two_head_vector_field"
print("  sys.path[0..2] :", sys.path[:3])

import sigmadock
import sigmadock.config
import sigmadock.data
import sigmadock.datafronts
import sigmadock.trainer
from sigmadock.diff import sampling, sigma_flow_generator
from sigmadock.net import model

mods = {
    "sigmadock": sigmadock,
    "config": sigmadock.config,
    "data": sigmadock.data,
    "datafronts": sigmadock.datafronts,
    "trainer": sigmadock.trainer,
    "net.model": model,
    "diff.sigma_flow_generator": sigma_flow_generator,
    "diff.sampling": sampling,
}
bad = []
for name, m in mods.items():
    f = os.path.abspath(m.__file__)
    ok = WANT in f
    print(f"  {'OK ' if ok else 'XXX'} {name:28s} {f}")
    if not ok:
        bad.append(name)
assert not bad, f"Diese Module kommen NICHT aus EXP-110: {bad}"

# Gibt es ein zweites installiertes sigmadock, das haette gewinnen koennen?
import importlib.util
spec = importlib.util.find_spec("sigmadock")
print("  aufgeloest ueber:", spec.origin)
print("  IMPORTHERKUNFT OK")
PYIMP

# ============================================================ 2. SANITY GATE
echo
echo "=============================================================="
echo "2. SANITY GATE (identisch zum 12h-Skript)"
echo "=============================================================="
$PYTHON - <<'PYCHECK'
import inspect, os
import sigmadock
from sigmadock.net.model import EquiformerV2
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator as G

path = os.path.abspath(sigmadock.__file__).replace(os.sep, "/")
WANT = "/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src/sigmadock/__init__.py"
assert path.endswith(WANT), f"FALSCHER BAUM: {path}"

src_model = inspect.getsource(EquiformerV2)
assert "self.trans_block" in src_model and "self.rot_block" in src_model
assert "self.force_block" not in src_model
assert "return trans_field, rot_field" in src_model
assert hasattr(G, "pool_fragment_fields")
src_vf = inspect.getsource(G._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ omega_world @ R_t' in src_vf
assert 'updates["omega"]' not in src_vf
src_fwd = inspect.getsource(G.forward)
assert "pool_fragment_fields" in src_fwd
for dead in ("_compute_fragment_dynamics", "_predict_fragment_updates"):
    assert dead not in src_fwd

variant_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
for exp in ("pdbbind-general", "posebusters"):
    f = os.path.join(variant_dir, "conf", "experiments", exp + ".yaml")
    assert os.path.isfile(f), "Experiment-Config FEHLT: " + f
print("  SANITY GATE OK")
PYCHECK

# ============================================================ 3. DATENSAETZE
echo
echo "=============================================================="
echo "3. DATENSAETZE AUF ARC"
echo "=============================================================="
# precheck_dataset steht nicht in _common.sh, sondern lokal in
# arc/throughput_sweep.slurm. Statt die bewiesene _common.sh anzufassen,
# wird dieselbe Logik hier inline benutzt: der etablierte Prober
# arc/probe_datafront_size.py, mit --expect-in-path als Baumkontrolle.
check_ds() {
    local ds="$1"
    echo "  --- ${ds} ---"
    if "$PYTHON" "${ARC_REPO}/arc/probe_datafront_size.py" \
            --data-dir "${ARC_DATA}" \
            --train "$ds" --val posebusters --test posebusters \
            --expect-in-path "EXP-110_two_head_vector_field" \
            > "${RUN_DIR}/precheck_${ds}.log" 2>&1; then
        grep -E "^train|^val|^GEMESSEN|Paare" "${RUN_DIR}/precheck_${ds}.log" \
            | head -4 | sed 's/^/      /'
    else
        echo "ABBRUCH -- nicht aufloesbar oder leer"
        tail -12 "${RUN_DIR}/precheck_${ds}.log" | sed 's/^/      /'
        echo "  Bekannt leer auf ARC: pdbbind-core, pdbbind-refined."
        exit 1
    fi
}
check_ds pdbbind-general
check_ds posebusters
echo "  ARC_DATA = ${ARC_DATA}"
df -h "${ARC_DATA}" | tail -1 | sed 's/^/  Platz: /'
ulimit -n | sed 's/^/  file descriptors: /'
echo "  /dev/shm: $(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $2" gesamt, "$4" frei"}')"

write_run_metadata "EXP110-SMOKE" "$CODE_DIR" "$PYTHON"

# ============================================================ 4. TRAINING
echo
echo "=============================================================="
echo "4. ECHTES TRAINING AUF DER GPU"
echo "   Aufruf ZEICHENGLEICH mit train_two_head_12h.slurm,"
echo "   nur durch timeout begrenzt."
echo "=============================================================="
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('  wandb preflight OK')"

# VRAM waehrend des Laufs mitschreiben, wie im Durchsatz-Sweep.
VRAM_LOG="${RUN_DIR}/vram.log"
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG" 2>/dev/null || true
    sleep 5
  done ) &
VRAM_PID=$!
trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

MARKER="${TMPDIR}/start_marker"; touch "$MARKER"
T0=$(date +%s)

timeout 1500 $PYTHON scripts/train.py \
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
    --debug \
  2>&1 | tee "${RUN_DIR}/train_stdout.log" || echo "  [info] timeout ist hier ERWARTET"

T1=$(date +%s)
kill $VRAM_PID 2>/dev/null || true
record_experiment_dir "$CODE_DIR" "$MARKER"

# ============================================================ 5. AUSWERTUNG
echo
echo "=============================================================="
echo "5. MESSWERTE"
echo "=============================================================="
echo "  Gesamtdauer Abschnitt 4: $(( T1 - T0 )) s"
PEAK="$(sort -n "$VRAM_LOG" 2>/dev/null | tail -1)"
echo "  Peak VRAM (nvidia-smi) : ${PEAK:-unbekannt} MiB von 46068 MiB"
if [ -n "${PEAK:-}" ]; then
    awk -v p="$PEAK" 'BEGIN{printf "  Kopffreiheit           : %.1f %% frei\n", 100*(46068-p)/46068}'
fi
echo "  NaN-Treffer im Log     : $(grep -ci "nan" "${RUN_DIR}/train_stdout.log" 2>/dev/null || echo 0)"

D="$(ls -1dt "${CODE_DIR}"/experiments/sigmadock/*/ 2>/dev/null | head -1)"
echo "  Experimentordner       : ${D:-KEINER}"
if [ -n "${D:-}" ]; then
    echo "  Checkpoints            : $(ls "$D/checkpoints" 2>/dev/null | wc -l)"
    ls -la "$D/checkpoints" 2>/dev/null | tail -3 | sed 's/^/    /'
fi

# Schrittrate aus dem Log schaetzen und auf 6 Epochen hochrechnen.
$PYTHON - "$D" "$(( T1 - T0 ))" <<'PYRATE'
import re, sys
from pathlib import Path

d, wall = sys.argv[1], int(sys.argv[2])
log = Path(d).parent.parent.parent / "x"  # nur Platzhalter, echtes Log unten
print()
print("  --- Hochrechnung auf den 12h-Lauf ---")
ck = sorted(Path(d).glob("checkpoints/*.ckpt")) if d else []
step = None
for c in ck:
    m = re.search(r"step=(\d+)", c.name)
    if m:
        step = max(step or 0, int(m.group(1)))
if step:
    print(f"  hoechster Checkpoint-Step : {step}")
    print("  ACHTUNG: darin steckt der Datafront-Aufbau (10-35 min), die reine")
    print("  Schrittrate ist also HOEHER als step/wall. Verbindlich ist die")
    print("  Rate aus dem spaeteren Teil des Laufs, siehe train_stdout.log.")
    print(f"  untere Schranke           : {step/wall:.4f} Schritte/s")
    print(f"  -> 13750 Schritte braeuchten hoechstens {13750/(step/wall)/3600:.1f} h")
else:
    print("  kein Checkpoint mit Stepnummer gefunden")
print()
print("  Referenz 8541310: 13750 Schritte in 11:05:40 = 0.344 Schritte/s")
print("  Wenn die gemessene Rate unter 0.318 liegt (8 % langsamer),")
print("  passen 6 Epochen NICHT in 12 h.")
PYRATE

echo
echo "=============================================================="
echo "URTEIL"
echo "=============================================================="
echo "  Gruen wenn: Sanity Gate OK, Importherkunft OK, beide Datensaetze OK,"
echo "  Checkpoints entstanden, keine NaN, VRAM mit Reserve, Rate >= 0.318/s."
echo "  Metadaten: ${RUN_DIR}"
