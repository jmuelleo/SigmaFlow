"""Validitaet ueber alle vier Zellen des Versuchsplans.

    Schrittzahl {5, 25} x Konformerquelle {bound, sampled} x drei Arme

Die vierte Zelle (sampled x 5) ist seit dem 24.08. gerechnet. Damit laesst
sich die entscheidende Frage beantworten: ist SigmaDocks Zusammenbruch bei
grober Integration ein reiner Integrationseffekt, oder haengt ein Teil davon
am informativen Rotationsprior, der bei fuenf Schritten nicht mehr zur
Geltung kommt?

Der Test ist ein Wechselwirkungsvergleich. Wenn der Abfall von 25 auf 5
Schritte in beiden Konformerquellen GLEICH gross ist, sind Prior und
Schrittzahl unabhaengig -- dann ist die Robustheit unkonfundiert. Faellt
SigmaDock im Paper-Setup deutlich flacher, war ein Teil des Absturzes bei
`bound` ein Prior-Effekt.

Diese Rechnung braucht keine RMSD-Werte und laeuft daher, waehrend die
SDF-Dateien noch auf ARC gepackt werden.

AUFRUF   python vierzellen_validitaet.py
"""
from __future__ import annotations

import csv
import glob
import re

import numpy as np

# (Anzeigename, Schritte, Konformer, Praefix der redock-Tabellen, Seedzahl)
ZELLEN = [
    ("25 Schr. / bound", 25, "bound", {"Minimal": "minimal80",
                                       "Separate": "exp110_80",
                                       "SigmaDock": "sigmadock80"}, 80),
    ("25 Schr. / sampled", 25, "sampled", {"Minimal": "minimalcs2",
                                           "Separate": "exp110cs2",
                                           "SigmaDock": "sigmadockcs2"}, 80),
    ("5 Schr. / bound", 5, "bound", {"Minimal": "minimal5",
                                     "Separate": "exp1105",
                                     "SigmaDock": "sigmadock5"}, 40),
    ("5 Schr. / sampled", 5, "sampled", {"Minimal": "minimaln5cs",
                                         "Separate": "exp110n5cs",
                                         "SigmaDock": "sigmadockn5cs"}, 40),
]
ARME = ("Minimal", "Separate", "SigmaDock")
BOOT = 4000
RNG = np.random.default_rng(0)


def tf(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def spalten(rows):
    cols = list(rows[0])
    rmsd_col = next((c for c in cols if c.startswith("rmsd")), "__none__")
    geladen = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
    checks = [c for c in cols
              if c not in ({"file", "molecule", "position", rmsd_col, ""} | geladen)]
    prot = [c for c in checks
            if any(t in c for t in ("protein", "cofactor", "water", "volume", "distance"))]
    return checks, [c for c in checks if c not in prot], prot


def komplex_id(pfad: str) -> str:
    return pfad.replace(chr(92), "/").split("/")[-1].split("__")[0]


def lade(praefix: str, max_seeds: int):
    """-> {(cid, seed): (valid_ohne_protein, valid_mit_protein)}"""
    V = {}
    for f in glob.glob(f"rd_{praefix}_seed*.csv"):
        s = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        if s >= max_seeds:
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8", errors="replace")))
        checks, intra, _ = spalten(rows)
        for r in rows:
            V[(komplex_id(r["file"]), s)] = (all(tf(r[c]) for c in intra),
                                             all(tf(r[c]) for c in checks))
    return V


def matrix(V, cids, seeds, idx):
    return np.array([[V[(c, s)][idx] for s in seeds] for c in cids])


def paar(A: np.ndarray, B: np.ndarray):
    """Bootstrap ueber Komplexe. -> (Differenz in pp, p, an-der-Grenze)."""
    d = 100 * (B.mean() - A.mean())
    nc = A.shape[0]
    boot = np.empty(BOOT)
    for i in range(BOOT):
        ix = RNG.integers(0, nc, nc)
        boot[i] = 100 * (B[ix].mean() - A[ix].mean())
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return d, max(p, 1.0 / BOOT), p < 1.0 / BOOT


def main() -> int:
    # Alle Zellen auf 40 Seeds beschneiden -- der Vergleich soll nicht daran
    # haengen, dass eine Zelle die doppelte Datenbasis hat.
    NS = 40
    D = {}
    for name, ns, kf, praefixe, _ in ZELLEN:
        V = {a: lade(praefixe[a], NS) for a in ARME}
        keys = sorted(set.intersection(*[set(V[a]) for a in ARME]))
        cids = sorted({c for c, _ in keys})
        seeds = sorted({s for _, s in keys})
        if len(seeds) != NS:
            print(f"[WARN] {name}: {len(seeds)} Seeds")
        D[name] = {"cids": cids, "seeds": seeds,
                   "ohne": {a: matrix(V[a], cids, seeds, 0) for a in ARME},
                   "mit": {a: matrix(V[a], cids, seeds, 1) for a in ARME}}
        print(f"{name:<20} {len(cids)} Komplexe x {len(seeds)} Seeds")
    print()

    for schluessel, titel in (("ohne", "PB-valid OHNE Protein"),
                              ("mit", "PB-valid MIT Protein")):
        print("=" * 74)
        print(f"  {titel}")
        print("=" * 74)
        print(f"{'Zelle':<20}" + "".join(f"{a:>13}" for a in ARME))
        for name, *_ in ZELLEN:
            z = f"{name:<20}"
            for a in ARME:
                z += f"{100 * D[name][schluessel][a].mean():>11.2f} %"
            print(z)

        print(f"\n  Abfall von 25 auf 5 Schritte, gepaart ueber die 209 Komplexe")
        print(f"{'Konformer':<12}{'Arm':<12}{'25 Schr.':>10}{'5 Schr.':>10}"
              f"{'Differenz':>12}{'Faktor':>9}   p")
        abfall = {}
        for kf in ("bound", "sampled"):
            n25 = next(n for n, _, k, *_ in ZELLEN if k == kf and n.startswith("25"))
            n5 = next(n for n, _, k, *_ in ZELLEN if k == kf and n.startswith("5"))
            for a in ARME:
                A25 = D[n25][schluessel][a]
                A5 = D[n5][schluessel][a]
                d, p, grenze = paar(A25, A5)
                v25, v5 = 100 * A25.mean(), 100 * A5.mean()
                abfall[(kf, a)] = (v25, v5, d)
                ptxt = f"< {1.0/BOOT:.5f}" if grenze else f"= {p:.4f}"
                fak = v25 / v5 if v5 > 0 else float("inf")
                print(f"{kf:<12}{a:<12}{v25:>9.2f} %{v5:>9.2f} %"
                      f"{d:>+11.2f} pp{fak:>8.2f}x   p {ptxt}")

        print(f"\n  WECHSELWIRKUNG: ist der Abfall in beiden Konformerquellen gleich?")
        for a in ARME:
            db = abfall[("bound", a)][2]
            ds = abfall[("sampled", a)][2]
            print(f"    {a:<12} bound {db:+7.2f} pp   sampled {ds:+7.2f} pp"
                  f"   Unterschied {ds - db:+7.2f} pp")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
