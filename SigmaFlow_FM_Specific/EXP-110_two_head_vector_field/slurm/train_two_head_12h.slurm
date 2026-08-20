#!/bin/bash -l
#
# EXP-110 — 12h pdbbind-general Lauf der ZWEI-KOPF-Variante.
#
# GEGENSTUECK: Job 8541310, erzeugt von
#   SigmaFlow_Minimal/slurm/train_pdbbind_general_12h_framefix.sh
# (byte-identisch auch in SigmaFlow_Variants/d_frame_fix/slurm/). Verifiziert:
# SigmaFlow_Minimal/src und d_frame_fix/src sind identisch, und EXP-110 ist
# eine Kopie von SigmaFlow_Minimal @ 16a069a. Der Vergleich ist damit sauber.
#
# JEDE Trainingsgroesse ist aus jenem Skript uebernommen: batch_size 8,
# num_workers 6, max_epochs 6, val_check_interval 50,
# early_stopping_patience 0, trans_score_weight 2.0, rot_score_weight 0.5,
# --offline_run, --debug, Seed aus dem Config-Default (0), medium/l40s,
# 8 CPUs, 32G. Der EINZIGE beabsichtigte Unterschied ist der Auslesekopf.
#
# ---------------------------------------------------------------------------
# WALLTIME — BITTE VOR DEM ABSCHICKEN LESEN
#
#   Der zweite Attention-Kopf sind +12,8 % Parameter (16.899.826 gesamt,
#   1.920.449 je Kopf; das Skript misst es beim Start und schreibt es ins
#   Log). Grob ein zusaetzlicher Block auf sieben, also geschaetzt 10-15 %
#   mehr Rechenzeit je Schritt. Der Referenzlauf brauchte fuer 6 Epochen 11:05:40
#   von 12:00:00 -- die Marge betrug 54 Minuten.
#
#   Bei +14 % waeren es rund 12:39, also UEBER der Walltime. Der Lauf wuerde
#   mitten im Cosine-Anneal abgeschnitten, und genau das hat in diesem Projekt
#   schon einmal ganze Laufserien entwertet.
#
#   Default ist trotzdem 12:00:00, weil du einen GEMATCHTEN 12h-Lauf wolltest.
#   Wer die Zeitachse gegen die Gueltigkeit tauschen will:
#       WALLTIME=13:30:00 sbatch --time=13:30:00 slurm/train_two_head_12h.slurm
#   Der LR-Schedule bleibt dabei unveraendert (max_epochs 6), es aendert sich
#   nur, ob er zu Ende laeuft.
#
#   PRUEFEN NACH EINER STUNDE:
#       grep -oE "Epoch [0-9]+" slurm_logs/<jobid>.out | tail -1
#   Liegt die Schrittrate mehr als 8 % unter 1260 Schritten/Stunde, wird es
#   in 12 h nicht reichen.
# ---------------------------------------------------------------------------
#
# Usage: sbatch slurm/train_two_head_12h.slurm
#
#SBATCH --job-name=sigmaflow-twohead-12h
#SBATCH --partition=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#
# medium statt short aus demselben Grund wie im Referenzskript: short hat ein
# 12h-Limit, ein 12h-Job saesse exakt auf der Kante.
#
# slurm_logs/ muss vorher existieren:  mkdir -p slurm_logs

set -euo pipefail

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/stat-cadd/shug8458/sigmaflow_env

REPO=/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
CODE_DIR="${REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field"
cd "$CODE_DIR"

PYTHON=/data/stat-cadd/shug8458/sigmaflow_env/bin/python

# --- wandb-Haertung, woertlich aus dem Referenzskript ---------------------
export WANDB__SERVICE_WAIT=300
export TMPDIR="/data/stat-cadd/shug8458/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"

# Der eigene src-Baum MUSS vor der in sigmaflow_env installierten Kopie
# stehen. Ohne das laedt `import sigmadock` den unveraenderten Code und der
# Lauf waere ein teurer No-Op.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "python:    $PYTHON"
echo "version:   $($PYTHON --version)"
echo "code_dir:  $CODE_DIR"
echo "git:       $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo n/a)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
$PYTHON -c "import sigmadock; print('sigmadock loaded from:', sigmadock.__file__)"

# ---------------------------------------------------------------------------
# SANITY GATE — dass die ZWEI-KOPF-Variante geladen ist, und nicht Minimal.
# Ein Ordnername beweist nichts: eine unvollstaendige Uebertragung laesst den
# Pfad richtig aussehen, waehrend der Code alt ist.
# ---------------------------------------------------------------------------
$PYTHON - <<'PYCHECK'
import inspect, os, sys
import sigmadock
from sigmadock.net.model import EquiformerV2
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator as G

# Exakte Suffixpruefung statt einer Blacklist. Eine Blacklist waere hier
# falsch: die Repowurzel auf ARC heisst SigmaFlow_Development_JulianMueller
# und wuerde jedes Verbot auf "SigmaFlow_Development" ausloesen. Das Suffix
# schliesst jeden anderen Baum ohnehin aus und ist namensunabhaengig.
path = os.path.abspath(sigmadock.__file__).replace(os.sep, "/")
WANT = "/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src/sigmadock/__init__.py"
assert path.endswith(WANT), f"FALSCHER BAUM geladen: ist {path} -- soll auf {WANT} enden"

src_model = inspect.getsource(EquiformerV2)
assert "self.trans_block" in src_model, "trans_block fehlt"
assert "self.rot_block" in src_model, "rot_block fehlt"
assert "self.force_block" not in src_model, "ALTER force_block noch vorhanden"
assert "return trans_field, rot_field" in src_model, "forward gibt nicht zwei Felder zurueck"

assert hasattr(G, "pool_fragment_fields"), "pool_fragment_fields fehlt"
src_vf = inspect.getsource(G._compute_vector_field)
assert 'R_t.transpose(-1, -2) @ omega_world @ R_t' in src_vf, \
    "adjungierte Konjugation (Weltrahmen -> Koerperrahmen) FEHLT"
assert 'updates["omega"]' not in src_vf, "alter Newton-Euler-Pfad noch aktiv"

src_fwd = inspect.getsource(G.forward)
assert "pool_fragment_fields" in src_fwd, "forward benutzt das Pooling nicht"
for dead in ("_compute_fragment_dynamics", "_predict_fragment_updates"):
    assert dead not in src_fwd, f"{dead} ist im aktiven Trainingspfad"

print("SANITY GATE OK")
print("  Variante        : EXP-110 Zwei-Kopf-Vektorfeld")
print("  Modul           :", path)
print("  Koepfe          : trans_block + rot_block (geteilter Rumpf)")
print("  Pooling         : Mittel je Fragment")
print("  Rahmenwechsel   : R_t^T hat(omega) R_t  (Welt -> Koerper)")
print("  Newton-Euler    : nicht im aktiven Pfad")
PYCHECK

# Parameterzahl protokollieren. Das Modell wird GENAUSO gebaut wie in
# scripts/train.py (aus RunConfig), nicht mit ausgedachten Dimensionen --
# sonst waere die Zahl im Log huebsch und falsch.
$PYTHON - <<'PYCOUNT' || echo "  (Parameterzaehler uebersprungen -- nicht kritisch)"
from sigmadock.config import RunConfig
from sigmadock.net.model import EquiformerV2

a = RunConfig()
m = EquiformerV2(
    use_esm_embeddings=a.use_esm_embeddings,
    num_layers=a.num_layers, num_heads=a.num_heads,
    atom_feature_dims=a.atom_feature_dims, average_degrees=a.average_degrees,
    edge_feature_dims=a.edge_feature_dims,
    lmax_list=a.l_max_list, mmax_list=a.m_max_list,
    protein_ligand_interactions=a.include_protein_ligand_interactions,
    ligand_ligand_interactions=a.include_fragment_fragment_interactions,
    sphere_channels=a.sphere_channels, edge_channels=a.edge_channels,
    attn_hidden_channels=a.attn_hidden_channels,
    attn_alpha_channels=a.attn_alpha_channels,
    attn_value_channels=a.attn_value_channels,
    ffn_hidden_channels=a.ffn_hidden_channels,
    t_emb_dim=a.t_emb_dim, t_emb_type=a.t_emb_type, t_emb_scale=a.t_emb_scale,
    distance_expansion_dim=a.distance_expansion_dim,
    smearing_type=a.smearing_type, radial_cutoff_function=a.radial_type,
    rel_distance=a.rel_distance,
    alpha_drop=a.attention_dropout, drop_path_rate=a.edge_dropout,
    zero_init_last=a.zero_init_last, share_edge_mlp=False,
)
tot = sum(p.numel() for p in m.parameters())
tr  = sum(p.numel() for p in m.trans_block.parameters())
ro  = sum(p.numel() for p in m.rot_block.parameters())
print(f"  Parameter gesamt : {tot:,}")
print(f"  trans_block      : {tr:,}")
print(f"  rot_block        : {ro:,}")
print(f"  Aufschlag ggb. einem Kopf: +{100*ro/(tot-ro):.2f} %")
assert tr == ro, "Die Koepfe sind NICHT baugleich -- Konfiguration pruefen"
print("  Koepfe baugleich : ja")
PYCOUNT

# Preflight: scheitern, bevor die 19k-Datafront gebaut wird.
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

# ---------------------------------------------------------------------------
# Trainingsaufruf — Zeichen fuer Zeichen der Referenzlauf 8541310.
# Kein --seed: der Config-Default (0) gilt in beiden Laeufen.
# --debug bleibt drin, weil der Referenzlauf es hatte. Es kostet Durchsatz,
# aber Vergleichbarkeit geht hier vor.
# ---------------------------------------------------------------------------
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

echo "Checkpoints unter: experiments/sigmadock/<timestamp>/checkpoints/"
echo "Ran: EXP-110 Zwei-Kopf, 12h pdbbind-general, max_epochs=6"
echo "Gegenstueck: Job 8541310 (SigmaFlow Minimal Changes, 12h)"
