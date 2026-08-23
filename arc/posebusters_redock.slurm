#!/bin/bash -l
#
# POSEBUSTERS MIT PROTEIN (Modus "redock"), ein Array-Task je Seed.
#
# WARUM DIESER JOB EXISTIERT
#   PoseBusters waehlt den Pruefumfang nach den uebergebenen Spalten
#   (posebusters/cli.py:_select_mode):
#       mol_pred + mol_true               -> "regen", NUR innere Geometrie
#       mol_pred + mol_true + mol_cond    -> "redock", ZUSAETZLICH Kollision
#                                             und Volumenueberlapp am Protein
#   Ohne Protein fehlt genau die Pruefung, die in der Praxis dominiert:
#   minimum_distance_to_protein besteht nur bei rund 16-20 % der Posen. Am
#   2026-08-22 verschwand dadurch ein vermeintlicher Validitaetsvorsprung der
#   Flow-Matching-Arme vollstaendig (p = 5e-10 ohne Protein, p >= 0.72 mit).
#   Fuer die Thesis zaehlt nur die Zahl MIT Protein.
#
# WARUM ALS ARRAY
#   Lokal gemessen: rund 5 Minuten je Seed und Arm, also 40 Seeds x 3 Arme
#   = etwa 10 Stunden seriell. Als Array laeuft jeder Seed einzeln.
#
# Es wird NICHTS neu gesampelt. Dieser Job wertet vorhandene Posen aus.
#
# Usage:
#   MODEL=exp110            sbatch --array=0-39 arc/posebusters_redock.slurm
#   MODEL=sigmaflow_minimal sbatch --array=0-39 arc/posebusters_redock.slurm
#   MODEL=sigmadock         sbatch --array=0-39 arc/posebusters_redock.slurm
#
#SBATCH --job-name=pb-redock
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

set -euo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

MODEL="${MODEL:?MODEL setzen: sigmaflow_minimal | exp100 | exp110 | sigmadock}"
SEED="${SLURM_ARRAY_TASK_ID:?nur als Array starten, z.B. --array=0-39}"
COMPARISON="${COMPARISON:-controlled}"
RANKING="${RANKING:-raw}"
NUM_STEPS="${NUM_STEPS:-25}"
SUFFIX="${SUFFIX:-__cpu}"

SAMPLING_ROOT="${ARC_RUNS}/sampling/${MODEL}__${COMPARISON}__${RANKING}__nfe${NUM_STEPS}${SUFFIX}"
BENCH="${ARC_TRUE_DIR}"                       # posebusters_benchmark_set/<cid>/

# Der Ausgabepfad MUSS jede Unterscheidung des Eingabepfads tragen. Ohne
# das schreibt ein Lauf mit SUFFIX=__cpu__confsampled seine Tabellen ueber
# die des Laufs mit SUFFIX=__cpu -- stiller Datenverlust, der erst bei der
# Auswertung auffiele und dann als Methodenunterschied dastuende.
# Bei SUFFIX=__cpu bleibt OUT_TAG leer, der Pfad also byte-identisch zu
# allen bisherigen Laeufen.
OUT_TAG="${SUFFIX#__cpu}"
OUT_DIR="${ARC_RUNS}/posebusters_redock/${MODEL}${OUT_TAG}"
OUT="${OUT_DIR}/rd_${MODEL}_seed${SEED}.csv"

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"

echo "=============================================================="
echo "  node          : $(hostname)"
echo "  model         : ${MODEL}   seed: ${SEED}"
echo "  sampling_root : ${SAMPLING_ROOT}"
echo "  benchmark_set : ${BENCH}"
echo "  out           : ${OUT}"
echo "=============================================================="

# Fail-fast, bevor eine Stunde ins Leere laeuft.
[ -d "$SAMPLING_ROOT" ] || { echo "ABBRUCH: ${SAMPLING_ROOT} fehlt"; exit 1; }
[ -d "$BENCH" ]         || { echo "ABBRUCH: ${BENCH} fehlt"; exit 1; }

SEED_DIR="$(find "$SAMPLING_ROOT" -type d -name "seed_${SEED}" | head -1)"
[ -n "$SEED_DIR" ] || { echo "ABBRUCH: kein seed_${SEED} unter ${SAMPLING_ROOT}"; exit 1; }
N_SDF=$(find "$SEED_DIR" -name '*.sdf' | wc -l)
echo "  Posen in seed_${SEED}: ${N_SDF}"
[ "$N_SDF" -gt 0 ] || { echo "ABBRUCH: keine Posen"; exit 1; }

mkdir -p "$OUT_DIR"

"$PYTHON" - "$SEED_DIR" "$BENCH" "$OUT" <<'PYEOF'
import os, sys, glob
import pandas as pd
from posebusters import PoseBusters
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

seed_dir, bench, out = sys.argv[1], sys.argv[2], sys.argv[3]

rows, missing = [], []
for f in sorted(glob.glob(os.path.join(seed_dir, "*.sdf"))):
    cid = os.path.basename(f).split("__")[0]
    lig = os.path.join(bench, cid, f"{cid}_ligands.sdf")
    prot = os.path.join(bench, cid, f"{cid}_protein.pdb")
    if os.path.isfile(lig) and os.path.isfile(prot):
        rows.append({"mol_pred": f, "mol_true": lig, "mol_cond": prot})
    else:
        missing.append(cid)

if missing:
    print(f"  ohne Ligand oder Protein, uebersprungen: {len(missing)} -> {missing[:5]}")
if not rows:
    sys.exit("ABBRUCH: keine auswertbaren Posen")

# config explizit: die Automatik waehlt redock nur, wenn mol_cond dabei ist.
df = PoseBusters(config="redock").bust_table(pd.DataFrame(rows), full_report=False)

# Gegenprobe VOR dem Schreiben: ohne geladenes Protein waere es faktisch doch
# nur "regen", und eine so benannte Datei waere schlimmer als keine.
if "mol_cond_loaded" in df.columns and not bool(df["mol_cond_loaded"].all()):
    sys.exit(f"ABBRUCH: mol_cond_loaded nicht ueberall True "
             f"({int((~df['mol_cond_loaded'].astype(bool)).sum())} Posen ohne Protein)")

df.to_csv(out)

skip = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
rmsd = next((c for c in df.columns if c.startswith("rmsd")), None)
checks = [c for c in df.columns if c not in skip and c != rmsd]
valid = df[checks].all(axis=1)
print(f"  geschrieben: {out}  ({len(df)} Posen)")
print(f"  PB-valid mit Protein : {100 * valid.mean():.1f} %")
if rmsd is not None:
    acc = df[rmsd].astype(bool)
    print(f"  RMSD < 2 A           : {100 * acc.mean():.1f} %")
    print(f"  beides               : {100 * (valid & acc).mean():.1f} %")
PYEOF

echo "fertig: ${OUT}"
