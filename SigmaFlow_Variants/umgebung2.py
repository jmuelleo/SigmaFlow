"""Die zwei vorgeschlagenen Regeln, woertlich genommen, je Zelle und Schwelle.

VORSCHLAG 1 -- DER ANTEIL ALS EXPONENT
    s = MixedScore ^ (Anteil valider Posen im Cluster)

    Das ist etwas anderes als eine Multiplikation und hat eine Tuecke: fuer
    Werte UEBER eins zieht ein Exponent unter eins den Score herunter, fuer
    Werte UNTER eins zieht er ihn herauf. Die Regel ist also nicht monoton in
    der Umgebungsguete, sondern kippt bei s = 1. Da der Mixed Score
    -Affinitaet mal p^4 ist und Affinitaeten typisch bei -5 bis -12 liegen,
    liegen die meisten Werte ueber eins -- aber nicht alle.

    Deshalb wird zusaetzlich die monotone Variante geprueft: exp(log(s) * a),
    also dasselbe mit einer Verschiebung, die das Kippen verhindert, sowie
    die schlichte Multiplikation als Vergleich.

VORSCHLAG 2 -- NUR CLUSTER MIT GENUEGEND VALIDEN POSEN
    Nur Posen aus Clustern, in denen mindestens q der Mitglieder valide sind,
    kommen infrage; darin dann Argmax. OHNE zusaetzliche Bedingung an die
    Pose selbst -- das ist der Unterschied zur frueheren Fassung, in der die
    gewaehlte Pose selbst valide sein musste.

    Die Idee dahinter ist, dass eine leicht mangelhafte Pose in guter
    Nachbarschaft besser sein kann als eine formal valide Pose im Nirgendwo.

Aufruf:
    python SigmaFlow_Variants/umgebung2.py --satz pb308
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
p.add_argument("--schwellen", default="0.5,1.0,2.0")
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
AUS = os.path.join(_HIER, f"umgebung2_{a.satz}.csv")

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
        rf = (int(np.where(valid)[0][np.argmax(gn[valid])]) if valid.any()
              else int(np.argmax(gn)))
        r = {"arm": arm, "nfe": nfe, "complex": code,
             "mixed": bool(ziel[int(np.argmax(heur))]),
             "f24+gnina": bool(ziel[rf])}
        for S in SCHW:
            lab = (np.ones(n, int) if n <= 1 else
                   fcluster(linkage(squareform(d, checks=False),
                                    method="complete"),
                            t=S, criterion="distance"))
            cval = np.empty(n)
            for cl in np.unique(lab):
                m = lab == cl
                cval[m] = valid[m].mean()

            # Vorschlag 1, woertlich
            r[f"heur^cval@{S}"] = bool(ziel[int(np.argmax(heur ** cval))])
            r[f"gn^cval@{S}"] = bool(ziel[int(np.argmax(gn ** cval))])
            # monotone Varianten zum Vergleich
            r[f"heur*cval@{S}"] = bool(ziel[int(np.argmax(heur * cval))])
            r[f"heur*cval^.5@{S}"] = bool(
                ziel[int(np.argmax(heur * np.sqrt(cval)))])

            # Vorschlag 2, ohne Bedingung an die Pose selbst
            for q in (0.25, 0.5, 0.75):
                m = cval >= q
                r[f"cl>={q}, heur@{S}"] = bool(
                    ziel[int(np.where(m)[0][np.argmax(heur[m])])]) if m.any() \
                    else bool(ziel[int(np.argmax(heur))])
                r[f"cl>={q}, gnina@{S}"] = bool(
                    ziel[int(np.where(m)[0][np.argmax(gn[m])])]) if m.any() \
                    else bool(ziel[int(np.argmax(gn))])
        zeilen.append(r)

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)
STRAT = [c for c in t.columns if c not in ("arm", "nfe", "complex")]

print(f"########## {a.satz.upper()}, {len(t) // len(ZELLEN)} Komplexe ##########\n")
kopf = (f"  {'Regel':<22}"
        + "".join(f"{LANG[arm][:9] + '/' + str(nfe):>13}" for arm, nfe in ZELLEN)
        + f"{'Mittel':>9}{'vs Basis':>10}")
basis = np.mean([100 * t[(t.arm == x) & (t.nfe == y)]["f24+gnina"].mean()
                 for x, y in ZELLEN])
for block, titel in ((["mixed", "f24+gnina"], "Bezugsgroessen"),
                     ([s for s in STRAT if "^cval" in s or "*cval" in s],
                      "Vorschlag 1 -- Anteil als Exponent bzw. Faktor"),
                     ([s for s in STRAT if s.startswith("cl>=")],
                      "Vorschlag 2 -- nur Cluster mit genug validen Posen")):
    print(f"=== {titel} ===")
    print(kopf)
    for s_ in block:
        zl = [100 * t[(t.arm == x) & (t.nfe == y)][s_].mean() for x, y in ZELLEN]
        m = np.mean(zl)
        marke = " <--" if m > basis else ""
        print(f"  {s_:<22}" + "".join(f"{x:12.1f}%" for x in zl)
              + f"{m:8.1f}%{m - basis:+9.1f}{marke}")
    print()
print(f"{len(t)} Zeilen nach {AUS}")
