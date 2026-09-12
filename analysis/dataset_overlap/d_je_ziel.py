"""Je PB308-Ziel: hat die Entfernung der 406 seinen naechsten Nachbarn
gekostet, oder blieb ein gleich naher uebrig?

Die Gruppenvergleiche in c_die406.py sagen nur, wie die 406 IM MITTEL liegen.
Entscheidend fuer den Testsatz ist aber, ob ein einzelnes Ziel Nachbarschaft
verliert -- ein Ziel mit 50 nahen Verwandten im Trainingssatz merkt es nicht,
wenn sechs davon wegfallen.
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("aehnlichkeit_19443.csv")
weg = {z.strip().lower() for z in open("entfernt_406_zeilen.txt") if z.strip()}
df["entfernt"] = df.code.str.lower().isin(weg)

for spalte, partner, tag in [("prot_cont", "prot_partner", "PROTEIN >= 0.90"),
                             ("lig_tanimoto", "lig_partner", "LIGAND >= 0.70")]:
    t = 0.90 if spalte == "prot_cont" else 0.70
    nah = df[df[spalte] >= t]
    a = nah[nah.entfernt].groupby(partner).size()
    b = nah[~nah.entfernt].groupby(partner).size()
    ziele = sorted(set(a.index) | set(b.index))
    tab = pd.DataFrame({"in_R1": b.reindex(ziele).fillna(0).astype(int),
                        "entfernt": a.reindex(ziele).fillna(0).astype(int)})
    verwaist = tab[(tab.entfernt > 0) & (tab.in_R1 == 0)]
    print(f"\n{tag}  (naechster Nachbar dieses Ziels)")
    print(f"  PB308-Ziele mit mindestens einem nahen Nachbarn: {len(tab)}")
    print(f"  davon betroffen von der Entfernung: {(tab.entfernt>0).sum()}")
    print(f"  davon VERWAIST (kein naher Nachbar mehr in R1): "
          f"{len(verwaist)}")
    if len(verwaist):
        print(verwaist.to_string())
    bt = tab[tab.entfernt > 0]
    if len(bt):
        print(f"  bei den betroffenen Zielen bleiben im Median "
              f"{bt.in_R1.median():.0f} nahe Nachbarn uebrig "
              f"(Minimum {bt.in_R1.min()}, Maximum {bt.in_R1.max()})")

# Verliert ein Ziel seinen ALLERNAECHSTEN Nachbarn?
print("\n\nMaximale Aehnlichkeit je PB308-Ziel, mit gegen ohne die 406")
for spalte, partner in [("prot_cont", "prot_partner"),
                        ("lig_tanimoto", "lig_partner")]:
    mit = df.groupby(partner)[spalte].max()
    ohne = df[~df.entfernt].groupby(partner)[spalte].max()
    j = pd.DataFrame({"mit": mit, "ohne": ohne}).dropna()
    d = (j["mit"] - j["ohne"])
    print(f"  {spalte}: {int((d > 1e-9).sum())} von {len(j)} Zielen "
          f"verlieren ueberhaupt etwas; "
          f"Median des Verlusts dort {d[d>1e-9].median() if (d>1e-9).any() else 0:.4f}, "
          f"Maximum {d.max():.4f}")
    gross = j[d > 0.05]
    if len(gross):
        print(f"    Ziele mit Verlust > 0.05:")
        print(gross.assign(verlust=d[d > 0.05]).sort_values(
            "verlust", ascending=False).head(10).to_string())
