#!/bin/bash -l
#
# EXP-100 SANITY — kurzer Lauf, bevor 24 GPU-Stunden riskiert werden.
#
# EXP-100 ist lokal auf CPU vollstaendig validiert (vier Testskripte, 0
# Fehlschlaege). Was lokal NICHT geprueft werden konnte:
#   * laeuft es auf der GPU / in float32-AMP,
#   * vertraegt es die ECHTEN Datensaetze statt der 10 Dummy-Komplexe,
#   * kostet die zusaetzliche ETKDGv3+MMFF-Konformergenerierung so viel
#     Dataloader-Zeit, dass die GPU hungert.
# Genau das misst dieser Lauf.
#
# Laufzeit ~40 min, passt in `short`.
#
# Usage: sbatch arc/train_exp100_sanity.slurm
#
#SBATCH --job-name=EXP100-sanity
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
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

CODE_DIR="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-100_state_reparam"
PYTHON="${ARC_SF_ENV}/bin/python"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"

cd "$CODE_DIR"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

echo "python:  $PYTHON"
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# ---------------------------------------------------------------- SANITY GATE
$PYTHON - <<'PYCHECK'
import inspect, os
import sigmadock
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator

path = os.path.abspath(sigmadock.__file__)
print("sigmadock geladen aus:", path)
assert "EXP-100" in path, f"FALSCHE Quelle: {path}"
assert hasattr(SigmaFlowGenerator, "get_fragment_com_and_rot_reparam"), "EXP-100-Methode fehlt"
assert "R_ref" in inspect.signature(SigmaFlowGenerator._apply_transformations).parameters, \
    "R_ref-Umbenennung fehlt"
src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ updates["omega"] @ R_t' in src, "FRAME FIX FEHLT"
print("EXP-100 aktiv, Frame-Fix vorhanden")
PYCHECK

write_run_metadata "EXP100-SANITY" "$CODE_DIR" "$PYTHON"

# ------------------------------------------------- 1. lokale Testbatterie auf ARC
# Dieselben vier Skripte, die lokal gruen sind. Wenn eines hier rot wird, liegt
# es an Umgebung oder Datensatz, nicht an der Mathematik - eine wertvolle
# Unterscheidung, die spaeter viel Suchzeit spart.
echo
echo "=== 1. Testbatterie (identisch zur lokalen) ==="
FAILED=0
for t in test_exp100 validate_mapping test_exp100_pipeline test_exp100_forward; do
    printf "  %-26s " "$t"
    if $PYTHON "tests/${t}.py" > "${RUN_DIR}/test_${t}.log" 2>&1; then
        echo "OK"
    else
        echo "FEHLGESCHLAGEN  -> ${RUN_DIR}/test_${t}.log"
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "ABBRUCH: Testbatterie auf ARC nicht gruen. Kein 24h-Lauf starten."
    exit 1
fi

# ------------------------------------------------- 2. echte Daten, ref_conf_pos
echo
echo "=== 2. ref_conf_pos auf dem ECHTEN Datensatz (nicht Dummy) ==="
$PYTHON - <<PYREAL 2>&1 | tee "${RUN_DIR}/real_data_check.log"
import time, torch
from sigmadock.data import SigmaDataset
from sigmadock.datafronts import DataFront
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
from torch_geometric.data import Batch

df = DataFront("${ARC_DATA}/pdbbind-general", pdb_regex=r".*_protein\.pdb\$", sdf_regex=r".*_ligand\.sdf\$")
print("Komplexe im Datafront:", len(df))
ds = SigmaDataset(datafront=df, pocket_com_noise=0.0, pocket_distance_cutoff=8.0,
                  pocket_distance_noise=0.0, prot_coordinate_distance_noise=0.0,
                  use_esm_embeddings=False, ignore_triangulation=False,
                  lig_coordinate_distance_noise=0.0, alignment_tries=0,
                  fragmentation_strategy="canonical", pb_check=False,
                  get_mol_info=True, seed=0, random_rotation=False,
                  sample_conformer=False, skip_bounds_check=True, force_retry=False)

items, t0, n_skip = [], time.time(), 0
for i in range(60):
    try:
        d = ds[i]
    except Exception as e:
        n_skip += 1; continue
    if d is None: n_skip += 1; continue
    assert hasattr(d, "ref_conf_pos"), f"ref_conf_pos fehlt bei Item {i}"
    assert d.ref_conf_pos.shape == d.ref_pos.shape, f"Formfehler bei {i}"
    assert torch.isfinite(d.ref_conf_pos).all(), f"nicht-endlich bei {i}"
    items.append(d)
dt = time.time() - t0
print(f"geladen: {len(items)}, uebersprungen: {n_skip}, {dt:.1f}s "
      f"({dt/max(len(items),1):.2f}s je Komplex, 1 Worker)")
print(f"-> hochgerechnet auf 6 Worker und batch_size 8: "
      f"{8*dt/max(len(items),1)/6:.2f}s je Batch reine Datenzeit")

b = Batch.from_data_list(items[:8])
gen = SigmaFlowGenerator(model=torch.nn.Identity(), sigma_min=0.0)
pos0, trans1, R1, nf = gen._get_initial_states(b)
ang = torch.rad2deg(torch.arccos(torch.clamp(
    (R1.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)))
print(f"R_1 auf echten Daten: median {float(ang.median()):.1f} Grad, "
      f"max {float(ang.max()):.1f} Grad")
assert float(ang.median()) > 20.0, "R_1 ist trivial - EXP-100 wirkt nicht"
assert torch.isfinite(pos0).all() and torch.isfinite(R1).all()
print("ECHTDATEN-CHECK BESTANDEN")
PYREAL

# ------------------------------------------------- 3. kurzer echter Trainingslauf
echo
echo "=== 3. Kurzes Training auf pdbbind-general (GPU, ~20 min) ==="
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

MARKER="${TMPDIR}/start_marker"; touch "$MARKER"

timeout 1500 $PYTHON scripts/train.py \
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
    --seed 0 \
    --offline_run \
    --debug \
  2>&1 | tee "${RUN_DIR}/train_stdout.log" || echo "[info] timeout ist hier ERWARTET"

record_experiment_dir "$CODE_DIR" "$MARKER"

echo
echo "=== SANITY-URTEIL ==="
grep -c "nan\|NaN" "${RUN_DIR}/train_stdout.log" 2>/dev/null | sed 's/^/  NaN-Treffer im Log: /' || true
echo "  Wenn oben alles gruen ist und Checkpoints entstanden sind:"
echo "    -> weiter mit sbatch arc/train_exp100_24h.slurm"
echo "  Metadaten: ${RUN_DIR}"
