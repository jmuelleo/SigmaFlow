"""Streaming-Auszug aus den beiden PDBbind-v2020-Tarballs.

Je Komplex: die Ligand-SDF (roh, klein) und die Aminosaeuresequenz des
Proteins aus den ATOM/CA-Zeilen. Ein Durchlauf ueber jedes Archiv, weil
gzip ohnehin sequentiell gelesen werden muss.
"""
import json
import os
import sys
import tarfile

AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
      "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
      "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
      "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
      "MSE": "M", "SEC": "U", "PYL": "O"}


def sequenz(roh: bytes) -> str:
    """Eine Kette pro Buchstabe, alle Ketten aneinander -- als Kennung
    reicht das; fuer die Aehnlichkeit wird ohnehin ueber k-mere gearbeitet."""
    out = []
    for z in roh.split(b"\n"):
        if z[:4] == b"ATOM" and z[12:16] == b" CA ":
            out.append(AA.get(z[17:20].decode("ascii", "ignore").strip(), "X"))
    return "".join(out)


ZIEL = sys.argv[1]
os.makedirs(os.path.join(ZIEL, "sdf"), exist_ok=True)
seqs = {}
n = 0
for pfad in sys.argv[2:]:
    with tarfile.open(pfad, "r:gz") as t:
        for m in t:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name.endswith("_ligand.sdf"):
                code = name[:-11]
                with open(os.path.join(ZIEL, "sdf", f"{code}.sdf"), "wb") as f:
                    f.write(t.extractfile(m).read())
            elif name.endswith("_protein.pdb"):
                code = name[:-12]
                seqs[code] = sequenz(t.extractfile(m).read())
                n += 1
                if n % 2000 == 0:
                    print(f"  {n} Proteine", flush=True)
    print(f"fertig: {pfad}", flush=True)

with open(os.path.join(ZIEL, "sequenzen.json"), "w") as f:
    json.dump(seqs, f)
print(f"{len(seqs)} Sequenzen, {len(os.listdir(os.path.join(ZIEL,'sdf')))} SDF")
