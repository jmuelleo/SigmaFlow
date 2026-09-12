"""Rankingstrategien, die Clusterinformation nutzen -- alle gegen dieselbe Basis.

DIE FRAGE
    Der Mixed Score bewertet jede Pose einzeln und wirft weg, ob ihre Umgebung
    sie stuetzt. Die Modenanalyse zeigt aber, dass Cluster bei 1 Angstroem
    weitgehend REIN sind -- entweder fast alle Posen darin sind korrekt oder
    fast keine. Diese Information muesste sich nutzen lassen.

DIE GEPRUEFTEN STRATEGIEN

  BEZUG
    mixed        Argmax des Mixed Scores. Der Score des Papers.
    f24+gnina    Erst alle 24 PB-Pruefungen als harter Filter, dann Argmax
                 der gnina-Affinitaet. Bisher die einzige Strategie, die den
                 Mixed Score schlaegt.

  GROESSE ALS VORFILTER
    sz>=m        Posen in Clustern unter m Mitgliedern werden verworfen,
                 danach ganz normal Argmax. Die Idee: eine Pose, die allein
                 steht, ist ein Ausreisser -- egal wie gut sie punktet.

  CLUSTER WAEHLEN, DANN BESTE DARIN
    top2, top3   Clusterwert = Mittel der besten zwei bzw. drei Mitglieder.
                 Anders als ein Anteil bevorzugt das kleine Cluster nicht:
                 zwei gute Posen sind zwei gute Posen, ob der Cluster fuenf
                 oder fuenfzig hat.

  RUECKHALT ALS GEWICHT, OHNE HARTE CLUSTER
    sup^g        Argmax von (Mixed Score) x (Anteil der Posen innerhalb von
                 1 Angstroem)^g. Das ist die weiche Fassung: keine Grenze,
                 keine Gruppenbildung, nur eine Gewichtung nach lokalem
                 Rueckhalt. Dieses Mass hatte als Konfidenzindikator die
                 hoechste AUC (0,83), also ist es der aussichtsreichste
                 Kandidat.

WARUM 1 ANGSTROEM UND NICHT 2
    Bei 2 A sind kaum noch Cluster rein -- der 100-Prozent-Anteil faellt von
    16 auf 4 Prozent. Die Information, um die es geht, existiert dort nicht.

Aufruf:
    python SigmaFlow_Variants/strategien.py --satz pb308
    python SigmaFlow_Variants/strategien.py --satz astex --schwelle 1.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE, lade_zelle  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwelle", type=float, default=1.0)
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--out", default=None)
a = p.parse_args()
S = a.schwelle
AUS = a.out or os.path.join(_HIER, f"strategien_{a.satz}_{S}.csv")


def wahl(heur, gn, f24, d, ziel):
    """Alle Strategien fuer EINEN Komplex. Gibt {Name: Treffer} zurueck."""
    n = len(heur)
    aus = {}
    aus["mixed"] = int(np.argmax(heur))
    aus["f24+gnina"] = (np.where(f24)[0][np.argmax(gn[f24])] if f24.any()
                        else int(np.argmax(gn)))

    lab = (np.ones(n, int) if n <= 1 else
           fcluster(linkage(squareform(d, checks=False), method="complete"),
                    t=S, criterion="distance"))
    gruppen = [np.where(lab == cl)[0] for cl in np.unique(lab)]
    groesse = {i: len(gr) for gr in gruppen for i in gr}
    gr_arr = np.array([groesse[i] for i in range(n)])

    for m in (2, 3, 5):
        maske = gr_arr >= m
        aus[f"sz>={m}"] = (int(np.where(maske)[0][np.argmax(heur[maske])])
                           if maske.any() else int(np.argmax(heur)))
        # dasselbe fuer die beste Bezugsstrategie
        mk = maske & f24
        aus[f"f24+sz>={m}"] = (int(np.where(mk)[0][np.argmax(gn[mk])])
                               if mk.any() else aus["f24+gnina"])

    for m in (2, 3):
        punkte = [np.sort(heur[gr])[::-1][:m].mean() for gr in gruppen]
        sieger = gruppen[int(np.argmax(punkte))]
        aus[f"top{m}"] = int(sieger[int(np.argmax(heur[sieger]))])

    # Rueckhalt: Anteil der Posen innerhalb der Schwelle um jede Pose.
    sup = (d < S).mean(axis=1)
    for g in (0.25, 0.5, 1.0):
        aus[f"sup^{g}"] = int(np.argmax(heur * sup ** g))
    for g in (0.25, 0.5):
        w = gn * sup ** g
        aus[f"f24+sup^{g}"] = (int(np.where(f24)[0][np.argmax(w[f24])])
                               if f24.any() else int(np.argmax(w)))
    return {k: bool(ziel[v]) for k, v in aus.items()}


zeilen = []
for arm, nfe in ZELLEN:
    z = next((z for z in SAETZE[a.satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    for code, v in posencache.hole(a.satz, arm, nfe, leise=True).items():
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        zeilen.append({"arm": arm, "nfe": nfe, "complex": code,
                       **wahl(g["heur"].to_numpy(), -g["affinity"].to_numpy(),
                              np.asarray(v["valid"], bool), v["d"].astype(float),
                              np.asarray(v[a.ziel], bool))})

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)
STRAT = [c for c in t.columns if c not in ("arm", "nfe", "complex")]

KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")
print(f"########## {a.satz.upper()}, Ziel {KRIT}, Clusterschwelle {S} A ##########\n")
print(f"  {'Strategie':<14}" + "".join(
    f"{LANG[arm][:9] + '/' + str(nfe):>14}" for arm, nfe in ZELLEN) + f"{'Mittel':>9}")
basis = {}
for s_ in STRAT:
    werte = []
    for arm, nfe in ZELLEN:
        d = t[(t.arm == arm) & (t.nfe == nfe)]
        werte.append(100 * d[s_].mean() if len(d) else np.nan)
    if s_ == "mixed":
        basis = dict(zip([f"{a_}{n_}" for a_, n_ in ZELLEN], werte))
    print(f"  {s_:<14}" + "".join(f"{w:13.1f}%" for w in werte)
          + f"{np.nanmean(werte):8.1f}%")

print(f"\n  Differenz zur Basis 'mixed':")
print(f"  {'Strategie':<14}" + "".join(
    f"{LANG[arm][:9] + '/' + str(nfe):>14}" for arm, nfe in ZELLEN) + f"{'Mittel':>9}")
for s_ in STRAT:
    if s_ == "mixed":
        continue
    diffs = []
    for arm, nfe in ZELLEN:
        d = t[(t.arm == arm) & (t.nfe == nfe)]
        diffs.append(100 * (d[s_].mean() - d["mixed"].mean()) if len(d) else np.nan)
    marke = " <--" if np.nanmean(diffs) > 0 else ""
    print(f"  {s_:<14}" + "".join(f"{x:+13.1f} " for x in diffs)
          + f"{np.nanmean(diffs):+8.1f}{marke}")
print(f"\n{len(t)} Zeilen nach {AUS}")
