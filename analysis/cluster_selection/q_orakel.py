"""Orakelabdeckung UNBEDINGT ueber alle Komplexe -- "es gab ueberhaupt nichts".

Gegenstueck zu o_versagen.py: dort war der Nenner die Zahl der Fehlgriffe,
hier ist es der ganze Satz. Die Frage: fuehrt SigmaFlows breitere Erzeugung
dazu, dass seltener GAR KEINE brauchbare Pose im Pool liegt?

Zerlegung des Pool-Versagens in seine zwei Ursachen:
  A  keine einzige Pose mit RMSD < 2 A            -> Genauigkeitsversagen
  B  es gab genaue Posen, aber keine davon war    -> Validitaetsversagen
     zugleich PB-valide
A + B = "es gab nichts" = 1 - Orakel(kombiniert).

Gepaarte Tests gegen SigmaDock: McNemar exakt, ueber dieselben Komplexe.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=["pb308", "astex"], default="pb308")
a = p.parse_args()

df = pd.read_csv(os.path.join(HIER, f"cluster_{a.satz}.csv"),
                 usecols=["arm", "nfe", "S", "complex", "K", "contains_acc",
                          "complex_hat_correct", "ranker_richtig"])
df = df[np.isclose(df.S, 2.0)]          # Etiketten sind schwellenunabhaengig

je = {}
for arm, nfe in ZELLEN:
    z = df[(df.arm == arm) & (df.nfe == nfe)]
    if z.empty:
        continue
    g = z.groupby("complex")
    je[f"{arm} {nfe}"] = pd.DataFrame({
        "top1": g["ranker_richtig"].first(),
        "orakel": g["complex_hat_correct"].first(),
        "hat_acc": g["contains_acc"].any(),
        "K": g["K"].first(),
    })

ix = sorted(set.intersection(*[set(v.index) for v in je.values()]))
print(f"Satz {a.satz}: {len(ix)} Komplexe in allen fuenf Zellen\n")

basis = je["SigmaDock 25"].loc[ix]
zeilen = []
for name, t in je.items():
    t = t.loc[ix]
    nichts = ~t.orakel
    kein_acc = ~t.hat_acc
    nur_ungueltig = t.hat_acc & ~t.orakel
    r = {
        "Zelle": name,
        "K": int(t.K.median()),
        "Orakel %": 100 * t.orakel.mean(),
        "ES GAB NICHTS %": 100 * nichts.mean(),
        "n": int(nichts.sum()),
        "A: kein RMSD<2 %": 100 * kein_acc.mean(),
        "B: nur ungueltig %": 100 * nur_ungueltig.mean(),
        "Top-1 %": 100 * t.top1.mean(),
    }
    if name != "SigmaDock 25":
        # McNemar auf dem Ereignis "Orakel getroffen"
        b = int((basis.orakel & ~t.orakel).sum())
        c = int((~basis.orakel & t.orakel).sum())
        pv = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        r["vs SD"] = f"{100*(t.orakel.mean()-basis.orakel.mean()):+.2f}"
        r["disk."] = f"{b}:{c}"
        r["p"] = f"{pv:.4g}"
    else:
        r["vs SD"], r["disk."], r["p"] = "", "", ""
    zeilen.append(r)

t = pd.DataFrame(zeilen)
print(t.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

print("\nLesehilfe: 'ES GAB NICHTS' = 100 - Orakel = A + B.")
print("A ist Genauigkeitsversagen (keine Pose nah genug),")
print("B ist Validitaetsversagen (nah genug, aber keine davon PB-valide).\n")

# Direktvergleich der beiden Flow-Arme untereinander bei gleichem Fahrplan
for x, y in [("Minimal 25", "Separate 25"), ("Minimal 5", "Separate 5"),
             ("Minimal 25", "Minimal 5"), ("Separate 25", "Separate 5")]:
    if x not in je or y not in je:
        continue
    u, v = je[x].loc[ix].orakel, je[y].loc[ix].orakel
    b, c = int((u & ~v).sum()), int((~u & v).sum())
    pv = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"  {x:12s} gegen {y:12s}: "
          f"{100*v.mean()-100*u.mean():+6.2f} pp, {b}:{c}, p = {pv:.4g}")
