"""
Anatomie des Versagens: wenn Top-1 falsch ist -- woran lag es?

Drei ineinander geschachtelte Fragen je Zelle, Ziel "RMSD<2 UND PB-valide",
Cluster: vollstaendige Verkettung auf dem paarweisen RMSD, geschnitten bei S.

  1. Wie oft versagt die Zelle ueberhaupt?           (1 - ranker_richtig)
  2. GEGEBEN Versagen: war ueberhaupt eine richtige   (complex_hat_correct)
     Pose im Pool?  -> reiner AUSWAHLFEHLER
  3. GEGEBEN Versagen: gab es einen Cluster mit
     Reinheit >= r und Groesse >= m, der NICHT
     gewaehlt wurde?                                 -> uebersehener Modus

Die Groessenbedingung ist noetig, weil ein Cluster aus einer einzigen
richtigen Pose per Konstruktion Reinheit 1,0 hat. Deshalb wird 3 ueber
mehrere Werte von r und m gefahren.
"""
import argparse
import os

import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
SPALTEN = ["satz", "arm", "nfe", "S", "complex", "cluster", "K",
           "contains_correct", "correct_fraction", "selected_by_ranker",
           "complex_hat_correct", "ranker_richtig", "g_gr", "g_relgr"]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=["pb308", "astex"], default="pb308")
p.add_argument("--S", type=float, default=2.0)
p.add_argument("--reinheit", default="0.5,0.7,0.9")
p.add_argument("--mindest", default="1,3,5")
a = p.parse_args()

R_WERTE = [float(x) for x in a.reinheit.split(",")]
M_WERTE = [int(x) for x in a.mindest.split(",")]

df = pd.read_csv(os.path.join(HIER, f"cluster_{a.satz}.csv"), usecols=SPALTEN)
df = df[np.isclose(df.S, a.S)]
print(f"Satz {a.satz}, Clusterschwelle {a.S} A, {len(df)} Clusterzeilen\n")

# ---------------------------------------------------------------- Teil 1+2
zeilen = []
for arm, nfe in ZELLEN:
    z = df[(df.arm == arm) & (df.nfe == nfe)]
    if z.empty:
        continue
    # eine Zeile je Komplex: die Etiketten sind ueber alle Cluster konstant
    kx = z.groupby("complex").agg(
        richtig=("ranker_richtig", "first"),
        hat=("complex_hat_correct", "first"),
        K=("K", "first"),
        n_cluster=("cluster", "size"),
    )
    n = len(kx)
    fehl = kx[~kx.richtig]
    rettbar = int(fehl.hat.sum())
    zeilen.append({
        "Zelle": f"{arm} {nfe}",
        "N": n,
        "Erfolg %": 100 * kx.richtig.mean(),
        "Versagen %": 100 * (~kx.richtig).mean(),
        "n_fehl": len(fehl),
        "davon rettbar %": 100 * rettbar / len(fehl) if len(fehl) else np.nan,
        "n_rettbar": rettbar,
        "davon Pool leer %": 100 * (len(fehl) - rettbar) / len(fehl)
        if len(fehl) else np.nan,
    })
t1 = pd.DataFrame(zeilen)
print("TEIL 1+2 -- Versagensquote und wie viel davon reiner Auswahlfehler ist")
print(t1.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
print()

# ------------------------------------------------------------------ Teil 3
print(f"TEIL 3 -- GEGEBEN Versagen: Anteil mit einem uebersehenen Cluster")
print("(Reinheit >= r, Groesse >= m Posen, NICHT vom Ranker gewaehlt)")
print("Nenner ist immer die Zahl der Fehlgriffe der Zelle.\n")

for m in M_WERTE:
    aus = []
    for arm, nfe in ZELLEN:
        z = df[(df.arm == arm) & (df.nfe == nfe)]
        if z.empty:
            continue
        kx = z.groupby("complex")["ranker_richtig"].first()
        fehl_ix = kx.index[~kx.values]
        zf = z[z["complex"].isin(fehl_ix)]
        r = {"Zelle": f"{arm} {nfe}", "n_fehl": len(fehl_ix)}
        for rr in R_WERTE:
            gut = zf[(zf.correct_fraction >= rr) & (zf.g_gr >= m)
                     & (~zf.selected_by_ranker)]
            r[f"r>={rr:g}"] = 100 * gut["complex"].nunique() / len(fehl_ix)
        aus.append(r)
    tt = pd.DataFrame(aus)
    print(f"  Mindestgroesse m = {m} Pose(n):")
    print(tt.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()

# ------------------------------------------------ Teil 4: relative Groesse
print("TEIL 4 -- dasselbe mit RELATIVER Mindestgroesse (Anteil an K),")
print("weil die 5-Schritt-Zellen K = 200 statt 40 Ziehungen haben.\n")
for mrel in (0.05, 0.10):
    aus = []
    for arm, nfe in ZELLEN:
        z = df[(df.arm == arm) & (df.nfe == nfe)]
        if z.empty:
            continue
        kx = z.groupby("complex")["ranker_richtig"].first()
        fehl_ix = kx.index[~kx.values]
        zf = z[z["complex"].isin(fehl_ix)]
        r = {"Zelle": f"{arm} {nfe}", "K": int(z.K.median()),
             "n_fehl": len(fehl_ix)}
        for rr in R_WERTE:
            gut = zf[(zf.correct_fraction >= rr) & (zf.g_relgr >= mrel)
                     & (~zf.selected_by_ranker)]
            r[f"r>={rr:g}"] = 100 * gut["complex"].nunique() / len(fehl_ix)
        aus.append(r)
    tt = pd.DataFrame(aus)
    print(f"  Mindestmasse {100*mrel:.0f} % der Ziehungen:")
    print(tt.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()

# --------------------------------------- Teil 5: Kontext des Fehlgriffs
print("TEIL 5 -- Kontext: was hat der Ranker im Fehlfall stattdessen gewaehlt?\n")
aus = []
for arm, nfe in ZELLEN:
    z = df[(df.arm == arm) & (df.nfe == nfe)]
    if z.empty:
        continue
    kx = z.groupby("complex")["ranker_richtig"].first()
    fehl_ix = kx.index[~kx.values]
    zf = z[z["complex"].isin(fehl_ix)]
    gew = zf[zf.selected_by_ranker]
    # nur die rettbaren Faelle: dort GAB es eine richtige Pose
    rett = zf[zf.complex_hat_correct]
    rg = rett.groupby("complex")
    aus.append({
        "Zelle": f"{arm} {nfe}",
        "n_fehl": len(fehl_ix),
        "Cluster je Komplex": zf.groupby("complex").size().mean(),
        "Reinheit gew. Cluster": gew.correct_fraction.mean(),
        "Groesse gew. Cluster": gew.g_gr.mean(),
        "Cluster m. korr. Pose": rg.apply(
            lambda g: int(g.contains_correct.sum())).mean(),
        "beste Reinheit dort": rg.correct_fraction.max().mean(),
    })
print(pd.DataFrame(aus).to_string(index=False,
                                  float_format=lambda x: f"{x:6.3f}"))
