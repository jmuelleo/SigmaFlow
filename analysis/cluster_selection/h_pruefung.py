"""Teil H -- die eingefrorene Regel gegen alles, was sie noch nicht gesehen hat.

DIE REGEL
    1. Bestimme den besten Mixed Score im Komplex.
    2. Kandidaten sind alle Cluster, deren bester Mixed Score hoechstens
       delta darunter liegt.
    3. Unter diesen: nimm den groessten Cluster.
    4. Darin: nimm die bestbewertete Pose (unveraenderter Mixed Score).

    delta = 0,5 und "groesster Cluster" wurden auf der Trainingshaelfte von
    PoseBusters aus rund 1500 Kandidatenregeln gewaehlt. Ab hier wird nichts
    mehr angepasst.

WARUM DIESE FORM UND NICHT "NIMM DEN GROESSTEN CLUSTER"
    Die ungefilterte Fassung verliert 11,2 Punkte. Der Unterschied ist das
    Band: es laesst die klaren Faelle unangetastet und greift nur ein, wo
    mehrere Cluster ueberhaupt um den Spitzenplatz konkurrieren. Ein Merkmal
    kann in einer Teilmenge tragen und ueber alles gemittelt schaden -- genau
    das ist hier der Fall, und deshalb ist die Torform noetig.

DIE PRUEFUNGEN
    1. PoseBusters, Testhaelfte -- gepaarter Bootstrap ueber Komplexe und
       McNemar auf den diskordanten Faellen.
    2. PoseBusters, beide Schwellen (2,0 und 1,0 A).
    3. Astex -- ein Benchmark, den keine Zeile dieser Untersuchung gesehen
       hat, mit anderen Proteinen und anderer Trefferquote.
    4. Je Zelle einzeln, weil eine Regel, die nur bei einem Modell traegt,
       fuer die Aussage wertlos ist.
    5. Eine delta-Kurve, um zu sehen, ob 0,5 ein Grat oder ein Plateau ist.
       Ein scharfes Maximum waere ein Zeichen fuer Ueberanpassung.

Aufruf:
    python analysis/cluster_selection/h_pruefung.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
SCHL = ["arm", "nfe", "complex"]
DELTA = 0.5
MERKMAL = "g_gr__r"          # Rang der Clustergroesse im Komplex
SEED = 20260909
rng = np.random.default_rng(SEED)


def anwenden(t, delta=DELTA, merkmal=MERKMAL):
    """Basis, Regel und Orakel als drei parallele Reihen ueber die Komplexe."""
    best = t.groupby(SCHL)["s_h_max"].transform("max")
    im_band = (best - t["s_h_max"]) <= delta
    # Im Band nach dem Merkmal, ausserhalb garantiert darunter. Gleichstaende
    # im Merkmal entscheidet der Mixed Score -- sonst haenge das Ergebnis an
    # der Clusternummer, also an der Seedreihenfolge, also an nichts.
    k = np.where(im_band,
                 100 + t[merkmal].fillna(-1) + 1e-3 * t["s_h_max__r"],
                 t["s_h_max__r"])
    tmp = t[SCHL].copy()
    tmp["_k"] = k
    i = tmp.groupby(SCHL, sort=False)["_k"].idxmax()
    regel = t.loc[i].set_index(SCHL)["best_pose_correct"]
    basis = t[t.selected_by_ranker].set_index(SCHL)["best_pose_correct"]
    orakel = t.groupby(SCHL)["best_pose_correct"].max()
    return regel, basis.reindex(regel.index), orakel.reindex(regel.index)


def bericht(name, regel, basis, orakel, boot=4000):
    n_a = int((~regel & basis).sum())      # zerstoert
    n_b = int((regel & ~basis).sum())      # gerettet
    pv = binomtest(n_b, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    # Gepaarter Bootstrap ueber KOMPLEXE: Cluster und Zellen desselben
    # Komplexes sind nicht unabhaengig, Ziehen auf Zeilenebene wuerde das
    # Intervall zu eng machen.
    df = pd.DataFrame({"r": regel, "b": basis}).reset_index()
    kx = df["complex"].unique()
    idx = {c: df.index[df["complex"] == c].to_numpy() for c in kx}
    r_, b_ = df.r.to_numpy(), df.b.to_numpy()
    d = np.empty(boot)
    for i in range(boot):
        j = np.concatenate([idx[c] for c in rng.choice(kx, len(kx), True)])
        d[i] = 100 * (r_[j].mean() - b_[j].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    g = 100 * (regel.mean() - basis.mean())
    sp = 100 * (orakel.mean() - basis.mean())
    print(f"  {name:<34}{100*basis.mean():7.2f}%{100*regel.mean():8.2f}%"
          f"{g:+8.2f}  [{lo:+6.2f},{hi:+6.2f}]{n_b:7d}{n_a:6d}"
          f"{pv:9.4f}{100*g/sp if sp > 0 else np.nan:8.1f}%")
    return g, lo, hi, pv


print("#" * 100)
print(f"  Regel: im Band delta = {DELTA} unter dem besten Mixed Score den "
      f"GROESSTEN Cluster,")
print("         darin die bestbewertete Pose. Auf der Trainingshaelfte von "
      "PoseBusters gewaehlt.")
print("#" * 100)
kopf = "Prueffall"
print(f"\n  {kopf:<34}{'Basis':>7}{'Regel':>8}{'Gewinn':>8}"
      f"{'95%-KI':>17}{'rett.':>7}{'zerst.':>6}{'p':>9}{'Luecke':>9}")

# ---- 1./2. PoseBusters, beide Schwellen, Test und alles ----------------
tp = pd.read_csv(os.path.join(_HIER, "cluster_pb308.csv"))
kx = np.array(sorted(tp["complex"].unique()))
np.random.default_rng(SEED).shuffle(kx)
TEST = set(kx[len(kx) // 2:])         # dieselbe Aufteilung wie in d_regeln

for S in (2.0, 1.0):
    t = tp[tp.S == S]
    r, b, o = anwenden(t)
    m = r.index.get_level_values(2).isin(TEST)
    print()
    bericht(f"PB308 {S} A, Testhaelfte (154)", r[m], b[m], o[m])
    bericht(f"PB308 {S} A, alle 307", r, b, o)

# ---- 3. Astex, voellig unberuehrt --------------------------------------
ta = pd.read_csv(os.path.join(_HIER, "cluster_astex.csv"))
print()
for S in (2.0, 1.0):
    t = ta[ta.S == S]
    r, b, o = anwenden(t)
    bericht(f"ASTEX {S} A, alle 85 (unberuehrt)", r, b, o)

# ---- 4. Je Zelle -------------------------------------------------------
print(f"\n  --- je Zelle, PB308 bei 2,0 A ---")
t = tp[tp.S == 2.0]
r, b, o = anwenden(t)
for arm, nfe in ZELLEN:
    m = ((r.index.get_level_values(0) == arm)
         & (r.index.get_level_values(1) == nfe))
    bericht(f"{LANG[arm]}, {nfe} Schritte", r[m], b[m], o[m], boot=2000)

print(f"\n  --- je Zelle, ASTEX bei 2,0 A ---")
t = ta[ta.S == 2.0]
r, b, o = anwenden(t)
for arm, nfe in ZELLEN:
    m = ((r.index.get_level_values(0) == arm)
         & (r.index.get_level_values(1) == nfe))
    bericht(f"{LANG[arm]}, {nfe} Schritte", r[m], b[m], o[m], boot=2000)

# ---- 5. delta-Kurve ----------------------------------------------------
print("\n=== 5. Ist delta = 0,5 ein Grat oder ein Plateau? ===")
print("    Ein scharfes Maximum waere ein Zeichen fuer Ueberanpassung an die")
print("    Trainingshaelfte; ein breites Plateau spricht fuer einen echten Effekt.\n")
kopf2 = "delta"
print(f"  {kopf2:>6}{'PB308 Test':>13}{'PB308 alle':>13}{'ASTEX':>13}"
       f"{'Eingriff PB':>14}")
for d in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0):
    t = tp[tp.S == 2.0]
    r, b, o = anwenden(t, delta=d)
    m = r.index.get_level_values(2).isin(TEST)
    ta2 = ta[ta.S == 2.0]
    ra, ba, oa = anwenden(ta2, delta=d)
    best = t.groupby(SCHL)["s_h_max"].transform("max")
    eing = 100 * float((((best - t["s_h_max"]) <= d)
                        .groupby([t[c] for c in SCHL]).sum() > 1).mean())
    print(f"  {d:6.2f}{100*(r[m].mean()-b[m].mean()):+12.2f}"
          f"{100*(r.mean()-b.mean()):+13.2f}"
          f"{100*(ra.mean()-ba.mean()):+13.2f}{eing:13.1f}%")
print("\n  Eingriff = Anteil der Komplexe, in denen ueberhaupt mehr als ein")
print("  Cluster im Band liegt; nur dort kann die Regel etwas aendern.")
