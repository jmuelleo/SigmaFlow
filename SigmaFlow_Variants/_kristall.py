"""Bestehen die KRISTALLPOSEN selbst die ligandinternen PoseBusters-Pruefungen?

Nur die `mol`-Konfiguration: sie braucht weder Protein noch Referenzpose und
prueft die intrinsische Geometrie des Liganden allein. Die rezeptorseitigen
Pruefungen fehlen hier, weil die Protein-PDBs lokal nicht vorliegen.

Benutzt wird `<cid>_ligand.sdf` (Einzahl) -- das ist Kopie 0, genau die, an
der die gesamte Auswertungskette haengt.
"""
import pathlib
import warnings

from posebusters import PoseBusters
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

W = pathlib.Path("astex/astex_diverse_set")
pfade = sorted(W.glob("*/*_ligand.sdf"))
print(f"{len(pfade)} Kristallliganden gefunden")

pb = PoseBusters(config="mol")
d = pb.bust(list(pfade), None, None, full_report=True).reset_index()

pruef = [c for c in d.columns if d[c].dtype == bool]
print(f"\n{len(pruef)} ligandinterne Pruefungen\n")
print(f"{'Pruefung':<38} {'bestanden':>10}")
for c in sorted(pruef, key=lambda k: d[k].mean()):
    print(f"{c:<38} {100 * d[c].mean():9.1f}%")

alle = d[pruef].all(axis=1)
print(f"\nALLE bestanden: {100 * alle.mean():.1f}%  ({int(alle.sum())} von {len(d)})")

sp = "file" if "file" in d.columns else d.columns[0]
schlecht = sorted({pathlib.Path(str(p)).parent.name for p in d.loc[~alle, sp]})
if schlecht:
    print(f"Durchgefallen ({len(schlecht)}): {', '.join(schlecht[:25])}")
