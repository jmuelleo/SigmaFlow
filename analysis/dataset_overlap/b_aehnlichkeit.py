"""Aehnlichkeit jedes PDBbind-v2020-Komplexes zum naechsten PB308-Komplex.

Zwei Achsen, beide unabhaengig vom spaeteren R1-Schnitt:

  LIGAND    maximaler Tanimoto (Morgan r=2, 2048 bit) gegen die 308 Liganden
  PROTEIN   maximale k-mer-Containment (k=6) der Aminosaeuresequenz gegen die
            308 Proteine. Containment statt Jaccard, weil PDBbind-Konstrukte
            und PB308-Ketten unterschiedlich lang sind; ein Wert nahe 1
            heisst "dieselbe Proteinsequenz kommt darin vor".

Ausgabe: eine Zeile je PDBbind-Komplex.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
SP = sys.argv[1]
TRUE308 = sys.argv[2]
K = 6

AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
      "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
      "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
      "TYR": "Y", "VAL": "V", "MSE": "M"}
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def seq_aus_pdb(pfad):
    out = []
    with open(pfad, "rb") as f:
        for z in f:
            if z[:4] == b"ATOM" and z[12:16] == b" CA ":
                out.append(AA.get(z[17:20].decode("ascii", "ignore").strip(), "X"))
    return "".join(out)


def fp_aus_sdf(pfad):
    try:
        m = next(Chem.SDMolSupplier(pfad, sanitize=True, removeHs=True), None)
        if m is None:
            m = next(Chem.SDMolSupplier(pfad, sanitize=False, removeHs=True), None)
            if m is None:
                return None
            m.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                             catchErrors=True)
        return gen.GetFingerprint(m)
    except Exception:
        return None


# ------------------------------------------------------------------- PB308
codes308 = sorted(os.listdir(TRUE308))
fp308, seq308, het308, ok308 = [], [], [], []
for c in codes308:
    fp = fp_aus_sdf(os.path.join(TRUE308, c, f"{c}_ligand.sdf"))
    if fp is None:
        continue
    fp308.append(fp)
    seq308.append(seq_aus_pdb(os.path.join(TRUE308, c, f"{c}_protein.pdb")))
    het308.append(c.split("_", 1)[1])
    ok308.append(c)
print(f"PB308: {len(ok308)} von {len(codes308)} verwertbar", flush=True)

# invertierter k-mer-Index ueber die 308 Proteine
inv = {}
groesse308 = np.zeros(len(seq308), float)
for i, s in enumerate(seq308):
    km = {s[j:j + K] for j in range(len(s) - K + 1)}
    groesse308[i] = len(km)
    for x in km:
        inv.setdefault(x, []).append(i)
print(f"k-mer-Index: {len(inv)} verschiedene {K}-mere", flush=True)

# --------------------------------------------------------------- PDBbind
seqs = json.load(open(os.path.join(SP, "pdbbind", "sequenzen.json")))
zeilen = []
for n, (code, s) in enumerate(seqs.items(), 1):
    km = {s[j:j + K] for j in range(len(s) - K + 1)}
    treffer = np.zeros(len(seq308), float)
    for x in km:
        for i in inv.get(x, ()):
            treffer[i] += 1
    cont = treffer / np.maximum(np.minimum(len(km), groesse308), 1)
    j_bester = int(np.argmax(cont)) if len(cont) else -1

    fp = fp_aus_sdf(os.path.join(SP, "pdbbind", "sdf", f"{code}.sdf"))
    if fp is None:
        tmax, i_bester = np.nan, -1
    else:
        sim = np.array(DataStructs.BulkTanimotoSimilarity(fp, fp308))
        tmax, i_bester = float(sim.max()), int(np.argmax(sim))

    zeilen.append({
        "code": code,
        "prot_cont": float(cont.max()) if len(cont) else np.nan,
        "prot_partner": ok308[j_bester] if j_bester >= 0 else "",
        "lig_tanimoto": tmax,
        "lig_partner": ok308[i_bester] if i_bester >= 0 else "",
        "n_res": len(s),
    })
    if n % 2000 == 0:
        print(f"  {n} / {len(seqs)}", flush=True)

df = pd.DataFrame(zeilen)

# Index anhaengen: UniProt, Ligandcode, Jahr
idx = os.path.join(SP, "idx", "index")
rows = []
for z in open(os.path.join(idx, "INDEX_general_PL_name.2020")):
    if z.startswith("#"):
        continue
    t = z.split(None, 3)
    if len(t) >= 3:
        rows.append({"code": t[0], "jahr": int(t[1]), "uniprot": t[2],
                     "protname": t[3].strip() if len(t) > 3 else ""})
nm = pd.DataFrame(rows)
het = []
for z in open(os.path.join(idx, "INDEX_general_PL_data.2020")):
    if z.startswith("#"):
        continue
    c = z.split()[0]
    h = z[z.rfind("(") + 1:z.rfind(")")] if "(" in z else ""
    het.append({"code": c, "het": h})
df = df.merge(nm, on="code", how="left").merge(pd.DataFrame(het), on="code",
                                               how="left")
df["het308"] = df.het.isin(set(het308))
df.to_csv(os.path.join(SP, "aehnlichkeit_19443.csv"), index=False)
print(df.describe().to_string())
print(f"\ngeschrieben: {len(df)} Zeilen")
