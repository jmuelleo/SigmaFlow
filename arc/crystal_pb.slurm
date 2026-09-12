#!/bin/bash -l
#
# SIND DIE KRISTALLPOSEN SELBST POSEBUSTERS-GUELTIG? -- CPU-only
#
# WOZU
#   Unsere Modelle erreichen je Zug 27-36 % volle PB-Validitaet, und die
#   gewaehlten genauen Posen scheitern zu 70-81 % an Abstandspruefungen zum
#   Rezeptor. Bevor das als Modellschwaeche gilt, muss die Obergrenze bekannt
#   sein: bestehen die KRISTALLPOSEN diese Pruefungen ueberhaupt?
#
#   Lokal ist die ligandinterne Haelfte schon gemessen -- 97,6 % der 85
#   Astex-Kristallliganden bestehen alle vierzehn intrinsischen Pruefungen.
#   Die rezeptorseitige Haelfte braucht die Protein-PDBs und damit ARC.
#
# WAS GEPRUEFT WIRD
#   mol_pred = mol_true = die Kristallpose selbst, mol_cond = das Protein
#   beziehungsweise bei PDBBind die Tasche. Konfiguration `redock`, also
#   dieselbe Pruefmenge wie in der Auswertung. Der RMSD ist dabei
#   definitionsgemaess null und wird in der Auswertung ausgeblendet.
#
# BEI PDBBIND: TASCHE ODER PROTEIN, UMSCHALTBAR UEBER RECEPTOR
#   PDBBind liefert beide Dateien. `pdbbind-general.yaml` setzt
#   `pdb_regex: .*pocket[.]pdb$`, die TASCHE ist also das, worauf das Modell
#   konditioniert. Sie hat weniger Atome, die Abstandspruefungen fallen dort
#   milder aus. Das volle PROTEIN ist dagegen die faire Bezugsgroesse zum
#   Vergleich mit pb308 und astex, die beide `*_protein.pdb` benutzen.
#   Beide Laeufe lohnen sich und schreiben in getrennte Verzeichnisse.
#   Vorgabe ist `pocket`.
#
# WELCHER LIGAND
#   Die erste Datei nach `sort` und daraus das erste Molekuel -- also Kopie 0,
#   genau die, an der die gesamte Auswertungskette haengt (parsing.py:1044
#   nimmt `ligands[0]`, PoseBusters laedt aus einer Mehrkopien-SDF ebenfalls
#   die erste).
#
# Aufruf:
#   LIST=1 DATASET=pb308   bash arc/crystal_pb.slurm
#   DATASET=pb308          sbatch --array=0 arc/crystal_pb.slurm
#   DATASET=astex          sbatch --array=0 arc/crystal_pb.slurm
#   DATASET=pdbbind RECEPTOR=pocket  CHUNKS=20 sbatch --array=0-19%20 arc/crystal_pb.slurm
#   DATASET=pdbbind RECEPTOR=protein CHUNKS=20 sbatch --array=0-19%20 arc/crystal_pb.slurm
#
#   MERGE=1 DATASET=pdbbind RECEPTOR=pocket CHUNKS=20 bash arc/crystal_pb.slurm
#
#SBATCH --job-name=crystal-pb
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.out
#SBATCH --error=/data/stat-cadd/shug8458/arc_runs/slurm_logs/%A_%a.err

# Kein `set -e`: `module load` und `conda activate` liefern hier regelmaessig
# einen Rueckgabewert != 0, ohne dass etwas kaputt ist. Fehler werden statt
# dessen an jeder Stelle einzeln geprueft.
set -uo pipefail

# SLURM fuehrt eine KOPIE dieses Skripts in /var/spool/slurmd/jobNNN/ aus.
# ${BASH_SOURCE[0]} zeigt deshalb nicht ins Repo -- absoluter Pfad, wie in
# allen anderen arc/*.slurm. Daran starb Array 8691800 vollstaendig.
REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

DATASET="${DATASET:?DATASET setzen: pb308 | astex | pdbbind}"
CHUNKS="${CHUNKS:-1}"
case "$DATASET" in
  pb308)
      WURZEL="${ARC_DATA}/posebusters_v2_308"
      SDF_MUSTER="*_ligands.sdf"
      PDB_MUSTER="*_protein.pdb"
      ;;
  astex)
      WURZEL="${ARC_DATA}/posebusters_paper/astex_diverse_set"
      SDF_MUSTER="*_ligands.sdf"
      PDB_MUSTER="*_protein.pdb"
      ;;
  pdbbind)
      # PDBBind liefert BEIDE Dateien: <id>_pocket.pdb und <id>_protein.pdb.
      #   pocket  -> das, worauf das Modell konditioniert (pdbbind-general.yaml
      #              setzt pdb_regex: .*pocket[.]pdb$). Weniger Atome, die
      #              Abstandspruefungen fallen milder aus.
      #   protein -> die faire Bezugsgroesse zum Vergleich mit pb308 und astex,
      #              die beide *_protein.pdb benutzen.
      # Beide Laeufe lohnen sich; sie schreiben in getrennte Verzeichnisse.
      RECEPTOR="${RECEPTOR:-pocket}"
      case "$RECEPTOR" in
        pocket)  PDB_MUSTER="*pocket.pdb" ;;
        protein) PDB_MUSTER="*protein.pdb" ;;
        *) echo "RECEPTOR unbekannt: ${RECEPTOR} (pocket|protein)" >&2; exit 1 ;;
      esac
      WURZEL="${ARC_DATA}/pdbbind/general-set"
      SDF_MUSTER="*ligand*.sdf"
      OUT_DIR="${OUT_DIR:-${ARC_RUNS}/crystal_pb/${DATASET}_${RECEPTOR}}"
      ;;
  *) echo "DATASET unbekannt: ${DATASET}" >&2; exit 1 ;;
esac

OUT_DIR="${OUT_DIR:-${ARC_RUNS}/crystal_pb/${DATASET}}"

[ -d "$WURZEL" ] || { echo "ABBRUCH: ${WURZEL} fehlt" >&2; exit 1; }
N_DIRS=$(find "$WURZEL" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)

if [ "${LIST:-0}" = "1" ]; then
    echo "Datensatz  : ${DATASET}"
    echo "Wurzel     : ${WURZEL}"
    echo "Ordner     : ${N_DIRS}"
    echo "Ligand     : ${SDF_MUSTER}   Rezeptor: ${PDB_MUSTER}  (${RECEPTOR:-fest})"
    echo "Bloecke    : ${CHUNKS}   -> --array=0-$((CHUNKS - 1))"
    echo "Ausgabe    : ${OUT_DIR}"
    beispiel=$(find "$WURZEL" -mindepth 1 -maxdepth 1 | sort | head -1)
    echo "Beispiel   : $(basename "$beispiel")"
    ls "$beispiel" | head -5
    exit 0
fi

module load Mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARC_SF_ENV"
PYTHON="${ARC_SF_ENV}/bin/python"
mkdir -p "$OUT_DIR"

# ----------------------------------------------------------------- MERGE
if [ "${MERGE:-0}" = "1" ]; then
    "$PYTHON" - "$OUT_DIR" "$CHUNKS" <<'PYEOF'
import glob, os, sys
import pandas as pd

out, chunks = sys.argv[1], int(sys.argv[2])
teile = sorted(glob.glob(os.path.join(out, "crystal_*.csv")))
if len(teile) != chunks:
    print(f"WARNUNG: {len(teile)} Teildateien, {chunks} erwartet -- unvollstaendig.")
d = pd.concat([pd.read_csv(f) for f in teile], ignore_index=True)
d.to_csv(os.path.join(out, "crystal_alle.csv"), index=False)

LADE = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
rmsd = [c for c in d.columns if c.startswith("rmsd")]
pruef = [c for c in d.columns
         if d[c].dtype == bool and c not in LADE and c not in rmsd]

alle = d[pruef].all(axis=1)
print(f"\n{len(d)} Kristallposen, {len(pruef)} Pruefungen")
print(f"ALLE bestanden: {100*alle.mean():.2f} %   ({int(alle.sum())} von {len(d)})")
print("\nDurchfallquote je Pruefung (nur > 0):")
q = (~d[pruef]).mean().sort_values(ascending=False)
for k, v in q[q > 0].items():
    print(f"  {k:<44} {100*v:6.2f} %")
print(f"\ngeschrieben: {os.path.join(out, 'crystal_alle.csv')}")
PYEOF
    exit $?
fi

# ----------------------------------------------------------------- Sampling
IDX="${SLURM_ARRAY_TASK_ID:?nur als Array starten, z.B. --array=0}"
[ "$IDX" -lt "$CHUNKS" ] || { echo "FEHLER: Index ${IDX} >= ${CHUNKS}" >&2; exit 1; }

echo "=============================================================="
echo "  Knoten     : $(hostname)"
echo "  Datensatz  : ${DATASET}   Block ${IDX} von ${CHUNKS}"
echo "  Wurzel     : ${WURZEL}   (${N_DIRS} Ordner)"
echo "=============================================================="

OUT="${OUT_DIR}/crystal_${IDX}.csv"
"$PYTHON" - "$WURZEL" "$SDF_MUSTER" "$PDB_MUSTER" "$IDX" "$CHUNKS" "$OUT" <<'PYEOF'
import glob, os, sys, warnings
import pandas as pd
from posebusters import PoseBusters
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

wurzel, sdf_m, pdb_m, idx, chunks, out = sys.argv[1:7]
idx, chunks = int(idx), int(chunks)

# Reihum teilen, nicht am Stueck: benachbarte PDB-Codes stammen oft aus
# derselben Hinterlegung, ein zusammenhaengender Block waere keine
# Stichprobe des ganzen Satzes.
ordner = sorted(
    d for d in glob.glob(os.path.join(wurzel, "*")) if os.path.isdir(d)
)[idx::chunks]

zeilen, fehlend = [], []
for d in ordner:
    sdfs = sorted(glob.glob(os.path.join(d, sdf_m)))
    pdbs = sorted(glob.glob(os.path.join(d, pdb_m)))
    if sdfs and pdbs:
        zeilen.append({"mol_pred": sdfs[0], "mol_true": sdfs[0], "mol_cond": pdbs[0]})
    else:
        fehlend.append(os.path.basename(d))

print(f"Block {idx}: {len(ordner)} Ordner, {len(zeilen)} vollstaendige Paare")
if fehlend:
    print(f"  ohne Ligand oder Rezeptor: {len(fehlend)} -> {fehlend[:5]}")
if not zeilen:
    sys.exit("ABBRUCH: keine Paare")

t = PoseBusters(config="redock").bust_table(pd.DataFrame(zeilen), full_report=True)
t = t.reset_index()

# ACHTUNG: bust_table liefert eine Zeile je MOLEKUEL, nicht je Datei.
# posebusters und astex benutzen `*_ligands.sdf` (Plural) -- diese Dateien
# enthalten mehrere Ligandkopien, 308 Dateien werden zu 519 Zeilen. PDBBind
# hat `*ligand*.sdf` (Singular), ein Moleku"l je Datei. Der Komplexname muss
# deshalb aus der Pfadspalte kommen, nicht aus `zeilen` -- die urspruengliche
# Fassung setzte eine Liste der Laenge len(zeilen) und starb an genau dem
# Laengenunterschied (Jobs 8728067, 8728068).
sp = next((c for c in ("file", "mol_pred") if c in t.columns), t.columns[0])
t["komplex"] = t[sp].map(lambda p: os.path.basename(os.path.dirname(str(p))))

n_dateien = t[sp].nunique()
print(f"Tabelle: {len(t)} Zeilen aus {n_dateien} Dateien, "
      f"{t['komplex'].nunique()} Komplexe")
if len(t) != n_dateien:
    print(f"  HINWEIS: mehrere Ligandkopien je Datei "
          f"(im Schnitt {len(t) / n_dateien:.2f}). Die Auswertung muss je "
          f"Komplex EINE Kopie waehlen, sonst sind grosse Komplexe "
          f"ueberrepraesentiert.")

t.to_csv(out, index=False)

LADE = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
rmsd = [c for c in t.columns if c.startswith("rmsd")]
pruef = [c for c in t.columns if t[c].dtype == bool and c not in LADE and c not in rmsd]
print(f"ALLE bestanden: {100*t[pruef].all(axis=1).mean():.2f} %  ({len(t)} Posen)")
print(f"geschrieben: {out}")
PYEOF
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "[crystal] FEHLGESCHLAGEN (rc=${RC})" >&2
    exit "$RC"
fi

# SLURM COMPLETED ist kein Beweis, dass etwas entstanden ist.
N=$(($(wc -l < "$OUT" 2>/dev/null || echo 1) - 1))
echo "[crystal] fertig: ${N} Zeilen in ${OUT}"
[ "$N" -gt 0 ] || { echo "[crystal] FEHLER: leere Ausgabe." >&2; exit 1; }
