"""Teil D -- deterministische Clusterwahl, erschoepfend durchsucht.

DIE VEREINFACHUNG, DIE ALLES SCHNELL MACHT
    Jede Regel der Form "waehle einen Cluster, wende darin den bestehenden
    Mixed Score an" hat ein Ergebnis, das vollstaendig durch das Etikett
    `best_pose_correct` des gewaehlten Clusters bestimmt ist. Die gesamte
    Auswertung reduziert sich damit auf: Cluster nach einem Kriterium
    sortieren, den obersten nehmen, Etikett nachschlagen.

DIE REGELFAMILIEN
    fest      Ein Merkmal, absteigend. Enthaelt die Basis (hoechster
              Mixed-Score-Maximalwert) und die einfachen Vergleiche
              (groesster Cluster, hoechster Median-gnina, Zufall).
    band      Kandidaten sind die Cluster innerhalb von delta unter dem besten
              Mixed Score; darin entscheidet ein zweites Merkmal. Das ist die
              vom Nutzer priorisierte Familie: sie kann den Fehler nur dort
              korrigieren, wo mehrere Cluster ueberhaupt konkurrieren, und
              laesst die klaren Faelle unangetastet.
    tor       Wie `band`, aber nur wenn der Komplex global unsicher ist --
              gemessen am Rueckhalt der gewaehlten Pose oder an der Zahl der
              Konkurrenten im Band. Sonst bleibt die Basis stehen.

DIE ENTSCHEIDENDE BUCHFUEHRUNG
    Eine Regel, die fuenfzehn Fehlgriffe rettet und zwanzig Treffer zerstoert,
    ist wertlos. Deshalb wird jede Regel mit DREI Zahlen berichtet: gerettet,
    zerstoert, netto. Und der Anteil der geschlossenen Orakelluecke, weil die
    absolute Punktzahl ohne den Spielraum nicht einzuordnen ist.

AUFTEILUNG
    delta und die Merkmalswahl werden NUR auf den Trainingskomplexen bestimmt.
    Die Aufteilung erfolgt nach Komplex und gilt fuer alle fuenf Zellen
    gemeinsam -- derselbe Komplex darf nicht in einer Zelle im Training und in
    einer anderen im Test stehen, sonst leckt dieselbe Bindungstasche.

Aufruf:
    python analysis/cluster_selection/d_regeln.py --satz pb308 --S 2.0
"""
import argparse
import os

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--S", type=float, default=2.0)
p.add_argument("--anteil-train", type=float, default=0.5)
p.add_argument("--seed", type=int, default=20260909)
p.add_argument("--top", type=int, default=20)
a = p.parse_args()

t = pd.read_csv(os.path.join(_HIER, f"cluster_{a.satz}.csv"))
t = t[t.S == a.S].copy()
rng = np.random.default_rng(a.seed)

# ---- Aufteilung nach KOMPLEX, ueber alle Zellen gemeinsam ---------------
komplexe = np.array(sorted(t["complex"].unique()))
rng.shuffle(komplexe)
n_tr = int(a.anteil_train * len(komplexe))
TRAIN, TEST = set(komplexe[:n_tr]), set(komplexe[n_tr:])
t["teil"] = np.where(t["complex"].isin(TRAIN), "train", "test")
print(f"########## {a.satz.upper()}, Schwelle {a.S} A ##########")
print(f"{len(komplexe)} Komplexe: {len(TRAIN)} Training, {len(TEST)} Test\n")

MERK = [c for c in t.columns if c.endswith("__r")]
BASIS_SP = "s_h_max"


def bewerte(t_, wahl_sp):
    """Ergebnis einer Regel: die Spalte `wahl_sp` ist der Sortierschluessel."""
    g = t_.sort_values(wahl_sp, ascending=False).groupby(
        ["arm", "nfe", "complex"], sort=False).head(1)
    return g


def kennzahlen(g, t_):
    """Erfolg, Rettungen, Zerstoerungen, Orakelanteil -- je Zelle und gepoolt."""
    basis = t_[t_.selected_by_ranker].set_index(["arm", "nfe", "complex"])
    orakel = t_.groupby(["arm", "nfe", "complex"])["best_pose_correct"].max()
    g = g.set_index(["arm", "nfe", "complex"])
    b = basis["best_pose_correct"].reindex(g.index)
    n = g["best_pose_correct"]
    return pd.DataFrame({"neu": n, "basis": b,
                         "orakel": orakel.reindex(g.index)})


def zeile(name, kk):
    e_neu = 100 * kk.neu.mean()
    e_bas = 100 * kk.basis.mean()
    e_ora = 100 * kk.orakel.mean()
    gerettet = int((kk.neu & ~kk.basis).sum())
    zerstoert = int((~kk.neu & kk.basis).sum())
    luecke = ((e_neu - e_bas) / (e_ora - e_bas)) if e_ora > e_bas else np.nan
    return {"regel": name, "erfolg": e_neu, "gegen_basis": e_neu - e_bas,
            "gerettet": gerettet, "zerstoert": zerstoert,
            "netto": gerettet - zerstoert, "luecke": 100 * luecke}


# ------------------------------------------------------------ feste Regeln
t["_zufall"] = rng.random(len(t))
FEST = [("Basis: max Mixed", "s_h_max"),
        ("Zufall", "_zufall"),
        ("groesster Cluster", "g_gr"),
        ("hoechster Median Mixed", "s_h_med"),
        ("hoechster Median gnina", "s_g_med"),
        ("hoechster top3 Mixed", "s_h_top3"),
        ("hoechste PB-Validitaet", "p_valid"),
        ("hoechste Kontaktzahl", "k_kon45_med"),
        ("hoechste Kohaerenz", "k_kohaerenz")]

erg = []
for nm, sp in FEST:
    if sp not in t.columns:
        continue
    erg.append(zeile(nm, kennzahlen(bewerte(t, sp), t)))

# ---------------------------------------------------------- Bandfamilie
# Kandidaten: Cluster, deren bester Mixed Score hoechstens delta unter dem
# global besten liegt. Innerhalb des Bandes entscheidet ein zweites Merkmal.
t["_bestglobal"] = t.groupby(["arm", "nfe", "complex"])["s_h_max"].transform("max")
t["_abstand"] = t["_bestglobal"] - t["s_h_max"]

band_erg = []
for delta in (0.1, 0.25, 0.5, 1.0, 2.0):
    im_band = t["_abstand"] <= delta
    for sp in MERK:
        if sp in (BASIS_SP + "__r",):
            continue
        # Sortierschluessel: im Band nach dem Merkmal, ausserhalb nie gewaehlt
        t["_k"] = np.where(im_band, t[sp].fillna(-1) + 10.0, t["s_h_max__r"] - 100)
        g = bewerte(t, "_k")
        kk = kennzahlen(g, t)
        # Wie oft greift die Regel ueberhaupt ein? Eine Regel, die in 3 Prozent
        # der Faelle etwas aendert, kann hoechstens 3 Punkte bewegen -- ohne
        # diese Zahl ist ein kleiner Gewinn nicht von Rauschen zu trennen.
        eingriff = float((~g.set_index(["arm", "nfe", "complex"])
                          ["selected_by_ranker"]).mean())
        # Auswahl NUR auf Training
        tr = kk[kk.index.get_level_values(2).isin(TRAIN)]
        band_erg.append({"delta": delta, "merkmal": sp,
                         "eingriff": 100 * eingriff,
                         "train_gain": 100 * (tr.neu.mean() - tr.basis.mean()),
                         "kk": kk})

band_erg.sort(key=lambda x: -x["train_gain"])
print("=== Bandfamilie: die zwanzig besten auf dem TRAINING ===")
print(f"  {'delta':>6}  {'Merkmal':<30}{'Eingriff':>10}{'Gewinn train':>14}"
      f"{'Gewinn test':>13}{'gerettet':>10}{'zerstoert':>11}{'Luecke test':>13}")
for x in band_erg[:a.top]:
    kk = x["kk"]
    te = kk[kk.index.get_level_values(2).isin(TEST)]
    z = zeile("", te)
    print(f"  {x['delta']:6.2f}  {x['merkmal']:<30}{x['eingriff']:9.1f}%"
          f"{x['train_gain']:+13.2f}{z['gegen_basis']:+13.2f}"
          f"{z['gerettet']:10d}{z['zerstoert']:11d}{z['luecke']:12.1f}%")

# Die beste Bandregel nach TRAINING, dann auf TEST berichtet
beste = band_erg[0]
kk = beste["kk"]
te = kk[kk.index.get_level_values(2).isin(TEST)]
erg.append(zeile(f"BAND d={beste['delta']} / {beste['merkmal']} (Test)", te))

print(f"\n=== Feste Regeln, alle Komplexe ===")
E = pd.DataFrame(erg)
print(f"  {'Regel':<44}{'Erfolg':>9}{'vs Basis':>10}{'gerettet':>10}"
      f"{'zerstoert':>11}{'netto':>7}{'Luecke':>9}")
for _, r in E.iterrows():
    print(f"  {r.regel:<44}{r.erfolg:8.1f}%{r.gegen_basis:+10.2f}"
          f"{r.gerettet:10.0f}{r.zerstoert:11.0f}{r.netto:+7.0f}"
          f"{r.luecke:8.1f}%")

# Orakel und Spielraum zur Einordnung
ora = t.groupby(["arm", "nfe", "complex"])["best_pose_correct"].max()
bas = t[t.selected_by_ranker].set_index(["arm", "nfe", "complex"])["best_pose_correct"]
print(f"\n  Cluster-Orakel {100*ora.mean():.1f} %, Basis {100*bas.mean():.1f} %, "
      f"Spielraum {100*(ora.mean()-bas.mean()):+.1f} Punkte")

pd.DataFrame([{k: v for k, v in x.items() if k != "kk"}
              for x in band_erg]).to_csv(
    os.path.join(_HIER, f"d_band_{a.satz}_{a.S}.csv"), index=False)
E.to_csv(os.path.join(_HIER, f"d_fest_{a.satz}_{a.S}.csv"), index=False)
print(f"\nGeschrieben nach {_HIER}")
