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

# Die Experiment-Configs muessen existieren. train.py loest --train_exps und
# --val_exps darueber auf; fehlen sie, stirbt der Job nach Sekunden mit
# "Experiment config not found". Genau das ist bei dieser Variante passiert:
# beim Kopieren gingen alle sieben conf/experiments/*.yaml verloren, weil die
# .gitignore-Regel **/experiments/ greift. Am 2026-08-21 nachgetragen.
# path = <variante>/src/sigmadock/__init__.py -> drei Ebenen hoch
variant_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
conf_dir = os.path.join(variant_dir, "conf", "experiments")
for exp in ("pdbbind-general", "posebusters"):
    f = os.path.join(conf_dir, exp + ".yaml")
    assert os.path.isfile(f), "Experiment-Config FEHLT: " + f

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

# ---------------------------------------------------------------------------
# FAIL-FAST-GATTER
#
# Es gibt vor diesem Lauf keinen GPU-Vorlauf. Alles, was auf ARC schiefgehen
# kann und lokal nicht pruefbar war, muss deshalb HIER auffallen, in der
# ersten Minute, und nicht nach dem 10-35 Minuten dauernden Datafront-Aufbau
# oder gar nach Stunden.
#
# Reihenfolge ist Absicht: erst das Billigste, das den teuersten Fehler
# faengt.
# ---------------------------------------------------------------------------

# (1) IMPORTHERKUNFT. Der schlimmste denkbare Ausgang ist nicht ein Absturz,
#     sondern ein Lauf, der zwoelf Stunden lang den FALSCHEN Code trainiert.
#     Ein global in site-packages installiertes sigmadock wuerde genau das
#     verursachen und sonst nirgends auffallen.
echo
echo "--- Importherkunft ---"
$PYTHON - <<'PYIMP'
import os
import sys

WANT = "EXP-110_two_head_vector_field"
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
bad = [n for n, m in mods.items() if WANT not in os.path.abspath(m.__file__)]
for n, m in mods.items():
    f = os.path.abspath(m.__file__)
    print(f"  {'OK ' if WANT in f else 'XXX'} {n:28s} {f}")
assert not bad, f"Diese Module kommen NICHT aus EXP-110: {bad}"
print("  Importherkunft OK")
PYIMP

# (2) DATENSAETZE. pdbbind-core und pdbbind-refined sind auf ARC leer, weil sie
#     nie aus pdbbind/raw/P-L.tar.gz entpackt wurden. Genau daran starben die
#     Sweeps 8583394/8583395 in allen zehn Konfigurationen. Der etablierte
#     Prober loest die Pfade ueber die Experiment-Config auf, statt sie zu
#     konstruieren, und prueft auf nichtleer.
echo
echo "--- Datensaetze ---"
for DS in pdbbind-general posebusters; do
    echo "  --- ${DS} ---"
    if ! $PYTHON "${REPO}/arc/probe_datafront_size.py" \
            --data-dir /data/stat-cadd/shug8458/data \
            --train "$DS" --val posebusters --test posebusters \
            --expect-in-path "EXP-110_two_head_vector_field" \
            > "${TMPDIR}/precheck_${DS}.log" 2>&1; then
        echo "  ABBRUCH: Datensatz '${DS}' nicht aufloesbar oder leer."
        tail -15 "${TMPDIR}/precheck_${DS}.log" | sed 's/^/      /'
        exit 1
    fi
    grep -E "^train|^val|^GEMESSEN|Paare" "${TMPDIR}/precheck_${DS}.log" \
        | head -4 | sed 's/^/      /'
done

# (3) Preflight: scheitern, bevor die 19k-Datafront gebaut wird.
echo
$PYTHON -c "import os, wandb; r = wandb.init(mode='offline', dir=os.environ['WANDB_DIR'], project='preflight'); r.finish(); print('wandb preflight OK')"

# ---------------------------------------------------------------------------
# ABBRUCHSICHERHEIT
#
# Die Persistenz selbst kommt aus dem Trainingscode und ist ausreichend:
# val_check_interval=50 bei accum_grad_batches=1 und
# check_val_every_n_epoch=None heisst, dass alle 50 Trainingsbatches
# validiert wird, und ModelCheckpoint(save_last=True) schreibt bei jedem
# dieser Ereignisse last.ckpt neu -- mit Optimierer, Scheduler, Epoche,
# global_step, hyper_parameters und EMA. Lightning schreibt ueber
# _atomic_save, ein Abbruch mitten im Schreiben laesst also die vorherige
# Datei heil. Verlust bei einem Walltime-Kill: hoechstens 50 Schritte,
# beim Tempo des Referenzlaufs rund zweieinhalb Minuten.
#
# KEIN --signal=SIGUSR1: Lightnings _slurm_sigusr_handler_fn wuerde zwar
# einen Checkpoint schreiben, danach aber `scontrol requeue` aufrufen und
# den Job neu starten. Das waere fuer ein zeitbudgetiertes Experiment genau
# falsch. Auf SIGTERM setzt Lightning received_sigterm, wirft beim naechsten
# Batch SIGTERMException, und train.py flusht im finally-Zweig wandb.
#
# Hier unten kommt deshalb NUR eine Statusmarkierung dazu -- kein Eingriff
# in Optimierung, Modell, Loss, Daten, Seed oder Zeitplan.
# ---------------------------------------------------------------------------
EXP_ROOT="${CODE_DIR}/experiments/sigmadock"
mkdir -p "$EXP_ROOT"
STAMP="${TMPDIR}/exp110_start_marker"
: > "$STAMP"
T0=$(date +%s)
GIT_COMMIT="$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo unbekannt)"

newest_exp_dir() {
    find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d -newer "$STAMP" 2>/dev/null | sort | tail -1
}

write_run_status() {
    # Das JSON baut Python, nicht die Shell. Ein frueherer Entwurf setzte es
    # per Heredoc zusammen und erzeugte ungueltiges JSON, sobald ein Pfad
    # einen Backslash enthielt -- im Trockentest aufgefallen.
    local rc="$1" label="$2" d t1
    d="$(newest_exp_dir)" || true
    t1=$(date +%s)
    if [ -z "$d" ]; then
        echo "[STATUS] ${label} (rc=${rc}), aber kein Experimentordner gefunden."
        return 0
    fi
    # timeout, damit das Lesen des Checkpoints die Abraeumfrist von SLURM
    # (KillWait, meist 30 s) nicht sprengt.
    timeout 25 "$PYTHON" "${CODE_DIR}/slurm/write_run_status.py" \
        "$d" "$label" "$rc" "$T0" "$t1" "${SLURM_JOB_ID:-manual}" "$GIT_COMMIT" \
        || echo "[STATUS] ${label} (rc=${rc}) -- RUN_STATUS.json konnte nicht geschrieben werden."
}

on_term() {
    # SLURM schickt bei Walltime erst SIGTERM, dann nach KillWait SIGKILL.
    trap - EXIT TERM INT   # sonst schreibt die EXIT-Falle gleich nochmal
    write_run_status 143 "WALLTIME_ODER_SIGTERM"
    exit 143
}
on_exit() {
    local rc=$?
    trap - EXIT TERM INT
    if [ "$rc" -eq 0 ]; then
        write_run_status 0 "COMPLETED"
    else
        write_run_status "$rc" "ABGEBROCHEN_rc${rc}"
    fi
    exit "$rc"
}
trap on_term TERM INT
trap on_exit EXIT

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
echo "Status:      $(newest_exp_dir)/RUN_STATUS.json"
echo "Gegenstueck: Job 8541310 (SigmaFlow Minimal Changes, 12h)"
