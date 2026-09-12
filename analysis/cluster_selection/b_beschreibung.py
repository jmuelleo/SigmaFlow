"""Teil B -- trennt irgendein Merkmal gute von schlechten Clustern?

DREI EBENEN, VON DER SCHWAECHSTEN ZUR AUSSAGEKRAEFTIGSTEN

  1. GLOBAL
     AUROC ueber alle Cluster aller Komplexe, Etikett `contains_correct`.
     Schwaechste Ebene: sie misst zum grossen Teil, ob ein Komplex ueberhaupt
     loesbar ist, nicht ob DIESER Cluster der richtige ist. Ein Merkmal kann
     hier 0,8 erreichen und trotzdem nutzlos sein.

  2. INNERHALB DES KOMPLEXES
     AUROC nur zwischen den Clustern desselben Komplexes, gemittelt. Das ist
     die Frage, die zur Inferenzzeit gestellt wird.

  3. GEPAART GEGEN DEN AKTUELLEN VERLIERER  <- die entscheidende Ebene
     Nur die rettbaren Fehlgriffe: der Ranker hat den falschen Cluster
     gewaehlt, ein korrekt-haltiger existiert. Fuer jedes Merkmal wird
     gezaehlt, wie oft der BESTE korrekt-haltige Cluster ueber dem gewaehlten
     falschen liegt. Ueber 50 Prozent heisst: das Merkmal koennte den Fehler
     korrigieren. Unter 50 Prozent heisst: es wuerde ihn verstaerken.

     Dazu ein Vorzeichentest und die Stabilitaet ueber die fuenf Zellen. Ein
     Merkmal, das in drei Zellen ueber und in zwei unter 50 Prozent liegt, ist
     Rauschen, egal wie gut sein Mittelwert aussieht.

WARUM NICHT NUR DER MITTELWERT
    Die fuenf Zellen sind verschiedene Modelle und Ziehungszahlen. Ein
    Merkmal, das nur bei einer Zelle traegt, ist fuer die Fragestellung
    wertlos -- gesucht ist etwas, das ueber Modelle hinweg gilt.

Aufruf:
    python analysis/cluster_selection/b_beschreibung.py --satz pb308 --S 2.0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
ETIKETT = {"contains_correct", "contains_acc", "best_pose_correct",
           "correct_fraction", "best_rmsd", "selected_by_ranker",
           "complex_hat_correct", "ranker_richtig"}
META = {"satz", "arm", "nfe", "S", "complex", "cluster", "K"}

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--S", type=float, default=2.0)
p.add_argument("--ziel", default="contains_correct",
               choices=["contains_correct", "contains_acc"])
p.add_argument("--top", type=int, default=25)
a = p.parse_args()

t = pd.read_csv(os.path.join(_HIER, f"cluster_{a.satz}.csv"))
t = t[t.S == a.S].copy()
MERK = [c for c in t.columns
        if c not in ETIKETT | META and not c.endswith("__r")]
RANG = [c + "__r" for c in MERK if c + "__r" in t.columns]
ALLE = MERK + RANG
Z = a.ziel


def auc(x, y):
    y = np.asarray(y, bool)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print(f"########## {a.satz.upper()}, Clusterschwelle {a.S} A, Ziel {Z} ##########")
print(f"{len(t)} Cluster, {len(MERK)} Merkmale\n")

# ---------------------------------------------------------------- Ebene 3
# WARUM NICHT "bester korrekter gegen den gewaehlten"
#     Diese naheliegende Kennzahl ist verzerrt. Sie vergleicht das MAXIMUM
#     ueber n korrekt-haltige Cluster gegen EINEN falschen. Unter reinem
#     Zufall gewinnt ein Maximum aus n Ziehungen gegen eine einzelne mit
#     Wahrscheinlichkeit n/(n+1) -- bei drei korrekten Clustern also 75 %.
#     Fast jedes Merkmal landet damit scheinbar weit ueber 50 %, ohne irgend
#     etwas zu messen.
#
# WAS STATTDESSEN GEMESSEN WIRD
#     Die Frage, die zur Inferenzzeit wirklich gestellt wird: sortiere ALLE
#     Cluster des Komplexes nach dem Merkmal und nimm den obersten. Enthaelt
#     der eine korrekte Pose? Der Nullwert dazu ist der Anteil korrekter
#     Cluster am Komplex, also die Trefferquote eines Muenzwurfs, und wird je
#     Fall einzeln berechnet und mitgefuehrt. Berichtet wird der UEBERSCHUSS.
#
#     Zusaetzlich die gepaarte Quote in korrigierter Form: der gewaehlte
#     Cluster wird aus der Vergleichsmenge ausgeschlossen (sonst zaehlen
#     Faelle mit, in denen er selbst eine korrekte Pose enthaelt und nur die
#     falsche herausgibt -- ein Fehler INNERHALB des Clusters, den keine
#     Clusterwahl heilt), und ihr eigener analytischer Nullwert n/(n+1) wird
#     abgezogen.
paare = []
innen_selbst = 0
for (arm, nfe, code), g in t.groupby(["arm", "nfe", "complex"]):
    if g.ranker_richtig.iloc[0] or not g.complex_hat_correct.iloc[0]:
        continue
    w = g[g.selected_by_ranker]
    if len(w) != 1:
        continue
    ist_korrekt = g[Z].to_numpy(bool)
    andere = ~g.selected_by_ranker.to_numpy(bool)
    if w[Z].iloc[0]:
        innen_selbst += 1
    b = g[ist_korrekt & andere]
    if len(b) == 0:
        continue
    n_c = len(b)
    zeile = {"arm": arm, "nfe": nfe, "complex": code,
             "_null_paar": n_c / (n_c + 1.0),
             "_null_top1": float(ist_korrekt.mean())}
    for c in ALLE:
        wv = float(w[c].iloc[0])
        alle_v = g[c].to_numpy(float)
        bv = b[c].to_numpy(float)
        bv = bv[np.isfinite(bv)]
        if not np.isfinite(wv) or len(bv) == 0 or not np.isfinite(alle_v).all():
            zeile[c] = np.nan
            zeile["T_" + c] = np.nan
            continue
        zeile[c] = 1.0 if bv.max() > wv else (0.5 if bv.max() == wv else 0.0)
        # Sortiere ALLE Cluster nach dem Merkmal, nimm den obersten.
        zeile["T_" + c] = float(ist_korrekt[int(np.argmax(alle_v))])
    paare.append(zeile)
P = pd.DataFrame(paare)
print(f"Rettbare Fehlgriffe: {len(P)}  "
      + str(P.groupby(['arm', 'nfe']).size().to_dict()))
print(f"Davon ausgeschlossen, weil der gewaehlte Cluster SELBST eine korrekte "
      f"Pose enthaelt: {innen_selbst}")
print(f"  (dort liegt der Fehler innerhalb des Clusters, nicht zwischen den "
      f"Clustern -- keine Clusterwahl kann ihn heilen)")
print(f"Nullwert der gepaarten Quote im Mittel: "
      f"{100 * P['_null_paar'].mean():.1f} %")
print(f"Nullwert der Top-1-Quote (Zufallscluster) im Mittel: "
      f"{100 * P['_null_top1'].mean():.1f} %\n")

# ------------------------------------------------- Ebene 2, vektorisiert
# Die AUC innerhalb eines Komplexes ist die Mann-Whitney-Statistik. Statt
# 1535 Gruppen x 296 Merkmale einzeln zu schleifen (Stunden), wird je Merkmal
# EIN gruppierter Rang gebildet und daraus die Statistik geschlossen
# berechnet. Das ist dieselbe Zahl in Sekunden.
gkey = ["arm", "nfe", "complex"]
gg = t.groupby(gkey)
n1 = gg[Z].transform("sum")
n0 = gg[Z].transform("size") - n1
brauchbar = (n1 > 0) & (n0 > 0)          # nur gemischte Komplexe zaehlen
auc_innen = {}
for c in ALLE:
    rk = gg[c].rank()
    s = (rk * t[Z])[brauchbar]
    z_ = pd.DataFrame({"s": s, "n1": n1[brauchbar], "n0": n0[brauchbar]})
    z_["g"] = list(zip(*[t.loc[brauchbar, k] for k in gkey]))
    agg = z_.groupby("g").agg(s=("s", "sum"), n1=("n1", "first"),
                              n0=("n0", "first"))
    auc_innen[c] = float(((agg.s - agg.n1 * (agg.n1 + 1) / 2)
                          / (agg.n1 * agg.n0)).mean())

erg = []
for c in ALLE:
    if c not in P.columns:
        continue
    d = P[[c, "T_" + c, "_null_paar", "_null_top1"]].dropna()
    if len(d) < 20:
        continue
    # Gepaarte Quote, um ihren analytischen Nullwert bereinigt
    quote = float(d[c].mean() - d["_null_paar"].mean())
    # Top-1: sortiere alle Cluster nach dem Merkmal, nimm den obersten
    top1 = float(d["T_" + c].mean())
    null1 = float(d["_null_top1"].mean())
    # Wilcoxon-artiger Vorzeichentest gegen den fallweisen Nullwert
    diff = d["T_" + c] - d["_null_top1"]
    n_ja, n_nein = int((diff > 0).sum()), int((diff < 0).sum())
    pv = binomtest(n_ja, n_ja + n_nein, 0.5).pvalue if n_ja + n_nein else 1.0
    # Stabilitaet: in wie vielen Zellen liegt der Ueberschuss ueber null?
    je = []
    for x, y in ZELLEN:
        s = d[(P.arm == x) & (P.nfe == y)]
        if len(s):
            je.append(float(s["T_" + c].mean() - s["_null_top1"].mean()))
    stab = sum(1 for q in je if q > 0)
    erg.append({"merkmal": c, "top1": 100 * top1, "null1": 100 * null1,
                "ueber": 100 * (top1 - null1), "quote": 100 * quote,
                "n": len(d), "p": pv, "stabil": stab, "n_zellen": len(je),
                "auc_global": auc(t[c], t[Z]), "auc_innen": auc_innen[c],
                "spanne": (max(je) - min(je)) * 100 if je else np.nan})
E = pd.DataFrame(erg).sort_values("ueber", ascending=False)
E.to_csv(os.path.join(_HIER, f"b_merkmale_{a.satz}_{a.S}.csv"), index=False)

print("=== Ebene 3, rettbare Fehlgriffe ===")
print("  Top-1  : Alle Cluster nach dem Merkmal sortieren, den obersten nehmen --")
print("           enthaelt er eine korrekte Pose?")
print("  Zufall : derselbe Wert fuer einen zufaellig gezogenen Cluster.")
print("  Ueber  : die Differenz. NUR DIESE ZAHL IST AUSSAGEKRAEFTIG.")
print("  Quote  : gepaarte Quote, um ihren analytischen Nullwert bereinigt.\n")
kopf = "Merkmal"
print(f"  {kopf:<30}{'Top-1':>8}{'Zufall':>8}{'Ueber':>8}{'Quote':>8}{'p':>9}"
      f"{'stabil':>8}{'AUC innen':>11}")
print("  --- die fuenfzehn besten ---")
for _, r in E.head(15).iterrows():
    print(f"  {r.merkmal:<30}{r.top1:7.1f}%{r.null1:7.1f}%{r.ueber:+7.1f}"
          f"{r.quote:+8.1f}{r.p:9.4f}{int(r.stabil)}/{int(r.n_zellen):<6}"
          f"{r.auc_innen:11.3f}")
print("  --- die fuenf schlechtesten ---")
for _, r in E.tail(5).iterrows():
    print(f"  {r.merkmal:<30}{r.top1:7.1f}%{r.null1:7.1f}%{r.ueber:+7.1f}"
          f"{r.quote:+8.1f}{r.p:9.4f}{int(r.stabil)}/{int(r.n_zellen):<6}"
          f"{r.auc_innen:11.3f}")

print("\n=== Wie viele Merkmale schlagen den Zufall wirklich? ===")
print(f"  Ueberschuss ueber 0                  : "
      f"{int((E.ueber > 0).sum())} von {len(E)}")
print(f"  Ueberschuss ueber 5 Punkten          : {int((E.ueber > 5).sum())}")
print(f"  ueber 5 Punkten UND in allen Zellen  : "
      f"{int(((E.ueber > 5) & (E.stabil == E.n_zellen)).sum())}")
print(f"  ueber 10 Punkten UND in allen Zellen : "
      f"{int(((E.ueber > 10) & (E.stabil == E.n_zellen)).sum())}")
print(f"  ... davon mit p < 0,05               : "
      f"{int(((E.ueber > 10) & (E.stabil == E.n_zellen) & (E.p < 0.05)).sum())}")
best = E[(E.ueber > 5) & (E.stabil == E.n_zellen) & (E.p < 0.05)]
if len(best):
    print("\n  Kandidaten: Ueberschuss > 5 Punkte, in allen Zellen positiv, p < 0,05")
    print(f"    {kopf:<32}{'Top-1':>8}{'Zufall':>8}{'Ueber':>8}{'p':>9}")
    for _, r in best.head(30).iterrows():
        print(f"    {r.merkmal:<32}{r.top1:7.1f}%{r.null1:7.1f}%"
              f"{r.ueber:+7.1f}{r.p:9.4f}")
else:
    print("\n  KEIN Merkmal ueberlebt alle drei Bedingungen.")
