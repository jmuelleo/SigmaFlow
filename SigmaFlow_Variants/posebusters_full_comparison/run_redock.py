"""PoseBusters im Modus 'redock': mit Protein, also inklusive der
Kollisions- und Volumenpruefungen. Dieselben Posen wie bei 'regen',
nur ein anderes Preset -- es wird NICHT neu gesampelt.

Der Modus wird explizit gesetzt, nicht der Automatik ueberlassen:
_select_mode() waehlt 'redock' nur, wenn die Spalte mol_cond vorhanden ist.
"""
import sys, glob, os
import pandas as pd
from posebusters import PoseBusters
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

arm, root = sys.argv[1], sys.argv[2]
prot = {os.path.basename(p).replace("_protein.pdb", ""): p
        for p in glob.glob("pb_proteins/**/*_protein.pdb", recursive=True)}
buster = PoseBusters(config="redock")

for seed in range(10):
    out = f"rd_{arm}_seed{seed}.csv"
    if os.path.isfile(out):
        print(f"{arm} seed{seed}: existiert", flush=True); continue
    sd = next((p for p in glob.glob(f"{root}/**/seed_{seed}", recursive=True)
               if os.path.isdir(p)), None)
    if sd is None:
        print(f"{arm} seed{seed}: kein Verzeichnis", flush=True); continue
    rows = []
    for f in sorted(glob.glob(os.path.join(sd, "*.sdf"))):
        cid = os.path.basename(f).split("__")[0]
        t = f"true_ligands/{cid}_ligands.sdf"
        if cid in prot and os.path.isfile(t):
            rows.append({"mol_pred": f, "mol_true": t, "mol_cond": prot[cid]})
    df = buster.bust_table(pd.DataFrame(rows), full_report=False)
    df.to_csv(out)
    print(f"{arm} seed{seed}: {len(df)} Posen -> {out}", flush=True)
print("FERTIG", flush=True)
