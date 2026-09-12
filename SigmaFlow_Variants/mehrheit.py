"""Mehrheitsentscheid unter den besten m Posen.

DIE UMKEHRUNG
    Alle bisherigen Versuche haben ZUERST die Clusterstruktur benutzt und
    dabei gute Posen wegen schlechter Nachbarschaft verworfen. Hier ist es
    umgekehrt: erst die besten m Posen nach Score auswaehlen, dann NUR unter
    diesen die Clusterstruktur befragen.

    Damit kann die Regel nichts Gutes mehr wegwerfen -- die Kandidatenmenge
    ist bereits die Elite, die Clusterstruktur entscheidet nur noch, welche
    davon.

DAS PROTOKOLL
    1. Posen nach Score sortieren (Mixed Score oder gnina nach hartem
       Validitaetsfilter).
    2. Die besten m nehmen.
    3. Zaehlen, wie viele davon in welchem Cluster liegen.
    4. Den Cluster mit den meisten Vertretern unter den m waehlen.
    5. Darin die bestbewertete Pose nehmen.

    Gleichstand: der Cluster gewinnt, der die insgesamt bestbewertete Pose
    enthaelt. Ohne diese Regel waere das Ergebnis von der Reihenfolge der
    Clusternummern abhaengig -- also von der Seednummer, also von nichts.

    Bei m = 1 faellt die Regel exakt auf die Basis zurueck. Das ist der
    eingebaute Test: weicht die Spalte fuer m = 1 von der Basis ab, ist ein
    Fehler in der Umsetzung.

ZWEI VARIANTEN FUER SCHRITT 5
    "in m"  nur unter den m besten des Siegerclusters waehlen
    "voll"  im ganzen Siegercluster waehlen, auch unter Posen ausserhalb der
            Elite. Das kann eine bessere Pose finden, oeffnet die Auswahl
            aber wieder fuer schwach bewertete.

Aufruf:
    python SigmaFlow_Variants/mehrheit.py --satz pb308
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
from scipy.stats import binomtest  # noqa: E402

LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwellen", default="0.5,1.0,2.0")
p.add_argument("--m", default="1,3,5,10,20")
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
MS = [int(x) for x in a.m.split(",")]
AUS = os.path.join(_HIER, f"mehrheit_{a.satz}.csv")


def mehrheit(ordnung, lab, score, m, voll):
    """Cluster mit den meisten Vertretern unter den besten m; darin die beste."""
    top = ordnung[:m]
    if len(top) == 0:
        return int(ordnung[0])
    cl, anz = np.unique(lab[top], return_counts=True)
    beste = int(anz.max())
    kand = cl[anz == beste]
    if len(kand) > 1:
        # Gleichstand: der Cluster der insgesamt bestbewerteten Pose gewinnt.
        for i in ordnung:
            if lab[i] in kand:
                kand = [lab[i]]
                break
    sieger = kand[0]
    if voll:
        m_ = np.where(lab == sieger)[0]
    else:
        m_ = top[lab[top] == sieger]
    return int(m_[int(np.argmax(score[m_]))])


zeilen = []
for arm, nfe in ZELLEN:
    z = next((z for z in SAETZE[a.satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    for code, v in posencache.hole(a.satz, arm, nfe, leise=True).items():
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        heur = g["heur"].to_numpy()
        gn = -g["affinity"].to_numpy()
        valid = np.asarray(v["valid"], bool)
        ziel = np.asarray(v[a.ziel], bool)
        d = v["d"].astype(float)
        n = len(gn)

        # Zwei Ordnungen: der Mixed Score, und gnina unter den validen Posen
        # (valide zuerst, darin nach gnina -- das ist die bisher beste Regel).
        ord_mixed = np.argsort(-heur)
        schluessel = np.where(valid, gn + 1e6, gn)   # valide immer vorne
        ord_f24 = np.argsort(-schluessel)

        r = {"arm": arm, "nfe": nfe, "complex": code,
             "mixed": bool(ziel[ord_mixed[0]]),
             "f24+gnina": bool(ziel[ord_f24[0]])}
        for S in SCHW:
            lab = (np.ones(n, int) if n <= 1 else
                   fcluster(linkage(squareform(d, checks=False),
                                    method="complete"),
                            t=S, criterion="distance"))
            for m in MS:
                r[f"MX m{m}@{S}"] = bool(
                    ziel[mehrheit(ord_mixed, lab, heur, m, False)])
                r[f"F24 m{m}@{S}"] = bool(
                    ziel[mehrheit(ord_f24, lab, schluessel, m, False)])
                r[f"F24voll m{m}@{S}"] = bool(
                    ziel[mehrheit(ord_f24, lab, schluessel, m, True)])
        zeilen.append(r)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)
STRAT = [c for c in t.columns if c not in ("arm", "nfe", "complex")]


def mittel(s):
    return np.mean([100 * t[(t.arm == x) & (t.nfe == y)][s].mean()
                    for x, y in ZELLEN])


basis = mittel("f24+gnina")
print(f"########## {a.satz.upper()}, {len(t) // len(ZELLEN)} Komplexe, "
      f"Ziel RMSD<2 und PB-valide ##########\n")
kopf = (f"  {'Regel':<18}"
        + "".join(f"{LANG[x][:9] + '/' + str(y):>13}" for x, y in ZELLEN)
        + f"{'Mittel':>9}{'vs Basis':>10}")
print(kopf)
for s_ in ("mixed", "f24+gnina"):
    zl = [100 * t[(t.arm == x) & (t.nfe == y)][s_].mean() for x, y in ZELLEN]
    print(f"  {s_:<18}" + "".join(f"{q:12.1f}%" for q in zl)
          + f"{np.mean(zl):8.1f}%{np.mean(zl) - basis:+9.1f}")
print("  " + "-" * 92)
for s_ in sorted((s for s in STRAT if s not in ("mixed", "f24+gnina")),
                 key=lambda s: -mittel(s))[:16]:
    zl = [100 * t[(t.arm == x) & (t.nfe == y)][s_].mean() for x, y in ZELLEN]
    m = np.mean(zl)
    print(f"  {s_:<18}" + "".join(f"{q:12.1f}%" for q in zl)
          + f"{m:8.1f}%{m - basis:+9.1f}" + (" <--" if m > basis else ""))

print("\n=== Gepaart gegen 'f24+gnina', die drei besten Regeln ===")
for s_ in sorted((s for s in STRAT if s not in ("mixed", "f24+gnina")),
                 key=lambda s: -mittel(s))[:3]:
    x, y = t["f24+gnina"].to_numpy(bool), t[s_].to_numpy(bool)
    na, nb = int((x & ~y).sum()), int((y & ~x).sum())
    pv = binomtest(nb, na + nb, 0.5).pvalue if na + nb else 1.0
    print(f"  {s_:<18}{100 * y.mean():5.1f} gegen {100 * x.mean():5.1f}   "
          f"diskordant {nb}:{na}   p = {pv:.4f}")
print(f"\n{len(t)} Zeilen nach {AUS}")
