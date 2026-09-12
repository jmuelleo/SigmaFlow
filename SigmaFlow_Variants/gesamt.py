"""Die vollstaendige Modenanalyse eines Benchmarks, mit EINER Clusterdefinition.

WARUM EIN SKRIPT STATT FUENF
    Modenzahl, Konzentration, Vertrauensmass, Uebereinstimmungstabelle und
    Ursachenzerlegung haengen alle an derselben Clusterung. Lagen sie in
    getrennten Skripten, muesste ein Wechsel der Verkettungsregel an fuenf
    Stellen nachgezogen werden -- und die eine, die man vergisst, faellt nicht
    auf, weil die Zahlen weiter plausibel aussehen.

DIE VERKETTUNGSREGEL
    Vorgabe ist VOLLSTAENDIGE Verkettung: zwei Cluster verschmelzen erst, wenn
    ihr GROESSTER Kreuzabstand unter der Schwelle liegt. Daraus folgt die
    Garantie, die man beim Ansehen erwartet: **jedes Paar innerhalb eines
    Modus liegt unter der Schwelle.** Bei Mittelwert-Verkettung gilt das
    nicht -- dort wurden Paare bis 2,33 A im selben Cluster gemessen.

    Der Preis sind kleinere, zahlreichere Cluster. Der Unterschied zwischen
    den Armen wird dadurch nicht kleiner, sondern groesser.

Aufruf:
    python SigmaFlow_Variants/gesamt.py --satz pb308
    python SigmaFlow_Variants/gesamt.py --satz astex --methode average
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ARME = ["SigmaDock", "Minimal", "Separate"]
KURZ = {"SigmaDock": "SD", "Minimal": "MIN", "Separate": "SEP"}

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--methode", default="complete",
               choices=["complete", "average", "single"])
p.add_argument("--schwelle", type=float, default=2.0)
p.add_argument("--ziel", default="beides", choices=["acc", "beides"])
p.add_argument("--min-modus", type=int, default=2)
p.add_argument("--zieh", type=int, default=8000)
p.add_argument("--seed", type=int, default=20260908)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
S = a.schwelle


def auc(x, y):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(x, float)).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    m = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (100 * (m - h), 100 * (m + h))


def boot(diff):
    b = np.array([rng.choice(diff, len(diff)).mean() for _ in range(a.zieh)])
    pv = min(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0)
    return diff.mean(), pv, np.percentile(b, [2.5, 97.5])


def stern(pv):
    return ("***" if pv < 0.001 else "**" if pv < 0.01
            else "*" if pv < 0.05 else "n.s.")


# --- je Komplex und Arm alles einmal berechnen ---------------------------
zeilen = []
for arm in ARME:
    for code, v in posencache.hole(a.satz, arm, a.nfe, leise=True).items():
        d = v["d"].astype(float)
        n = len(d)
        ok = np.asarray(v[a.ziel], bool)
        heur = v["heur"]
        lab = fcluster(linkage(squareform(d, checks=False), method=a.methode),
                       t=S, criterion="distance")
        g = np.bincount(lab)[1:]
        gross = int(np.argmax(g)) + 1
        s = int(np.argmax(heur))
        pk = g / n
        iu = np.triu_indices(n, 1)
        # Groesster Paarabstand im selben Cluster -- die Probe auf die
        # Garantie der vollstaendigen Verkettung.
        dmax = max((d[np.ix_(np.where(lab == c)[0], np.where(lab == c)[0])].max()
                    if (lab == c).sum() > 1 else 0.0) for c in np.unique(lab))
        zeilen.append({
            "arm": arm, "complex": code, "n": n,
            "moden": int(len(g)),
            "moden_echt": int((g >= a.min_modus).sum()),
            "gross": float(g.max() / n),
            "eff_moden": float(np.exp(-(pk * np.log(pk)).sum())),
            "p_agree": float((d[iu] < S).mean()),
            "dmax_intra": dmax,
            "hat_korrekte": bool(ok.any()),
            "n_korrekt": int(ok.sum()),
            "gross_richtig": bool(ok[lab == gross].mean() > 0.5),
            "wahl_gross": bool(lab[s] == gross),
            "wahl_anteil": float((lab == lab[s]).sum() / n),
            "trifft": bool(ok[s]),
        })
t = pd.DataFrame(zeilen)
AUS = os.path.join(_HIER, f"gesamt_{a.satz}_nfe{a.nfe}_{a.methode}.csv")
t.to_csv(AUS, index=False)

W = {arm: t[t.arm == arm].set_index("complex") for arm in ARME}
idx = W[ARME[0]].index
for arm in ARME[1:]:
    idx = idx.intersection(W[arm].index)
N = len(idx)
KRIT = "RMSD < 2 A" + (" und PB-valide" if a.ziel == "beides" else "")
print(f"########## {a.satz.upper()}, {a.nfe} Schritte, {N} Komplexe, "
      f"{a.methode}-Verkettung bei {S} A, Ziel {KRIT} ##########")

# --- A -------------------------------------------------------------------
print("\n=== A. Struktur der Verteilungen ===")
print(f"  {'Arm':<11}{'Posen':>6}{'Cluster':>9}{'Moden>=2':>10}{'groesster':>11}"
      f"{'eff.Moden':>11}{'p_agree':>9}{'max intra':>11}")
for arm in ARME:
    d = W[arm].loc[idx]
    print(f"  {arm:<11}{d.n.mean():6.0f}{d.moden.mean():9.1f}"
          f"{d.moden_echt.mean():10.1f}{100 * d.gross.mean():10.1f}%"
          f"{d.eff_moden.mean():11.2f}{100 * d.p_agree.mean():8.1f}%"
          f"{d.dmax_intra.mean():10.2f} A")
print(f"\n  'max intra' ist der groesste Paarabstand INNERHALB eines Clusters.")
print(f"  Bei vollstaendiger Verkettung muss er unter {S} A liegen -- das ist")
print(f"  die Garantie, dass alle Posen eines Modus einander aehnlich sind.")

print("\n  Gepaart gegen SigmaDock:")
for arm in ARME[1:]:
    z = f"    {arm:<10}"
    for feld, lab_, sk, eh in (("gross", "groesster", 100, "pp"),
                               ("eff_moden", "eff.Moden", 1, ""),
                               ("p_agree", "p_agree", 100, "pp"),
                               ("moden_echt", "Moden", 1, "")):
        diff = ((W[arm][feld][idx] - W["SigmaDock"][feld][idx]).to_numpy() * sk)
        m, pv, _ = boot(diff)
        z += f"  {lab_} {m:+6.2f}{eh:<2} {stern(pv):<4}"
    print(z)

# --- B -------------------------------------------------------------------
print("\n=== B. Vertrauensmass: Anteil im GEWAEHLTEN Modus ===")
print(f"  {'Arm':<11}{'bei Treffer':>13}{'bei Fehlgriff':>15}{'Differenz':>11}{'AUC':>8}")
for arm in ARME:
    d = W[arm].loc[idx]
    r_, f_ = d.wahl_anteil[d.trifft], d.wahl_anteil[~d.trifft]
    print(f"  {arm:<11}{100 * r_.mean():12.1f}%{100 * f_.mean():14.1f}%"
          f"{100 * (r_.mean() - f_.mean()):+10.1f}"
          f"{auc(d.wahl_anteil, d.trifft):8.3f}")

pool = pd.concat([W[arm].loc[idx].assign(arm=arm) for arm in ARME])
print(f"\n  Kalibrierung, alle Arme zusammen ({len(pool)} Beobachtungen):")
print(f"  {'Quintil':<9}{'Konzentration':>16}{'n':>6}{'trifft':>9}{'95%-Intervall':>17}")
pool["q"] = pd.qcut(pool.wahl_anteil, 5, labels=False, duplicates="drop")
for q, gq in pool.groupby("q"):
    lo, hi = wilson(gq.trifft.sum(), len(gq))
    print(f"  {int(q) + 1:<9}{100 * gq.wahl_anteil.min():7.0f}-"
          f"{100 * gq.wahl_anteil.max():3.0f}%{len(gq):8d}"
          f"{100 * gq.trifft.mean():8.1f}%{lo:9.1f}-{hi:.1f}")

# --- C -------------------------------------------------------------------
muster = pd.DataFrame({arm: W[arm].trifft.loc[idx] for arm in ARME})
print("\n=== C. Wer trifft? ===")
print(f"  {'SD':>4}{'MIN':>5}{'SEP':>5}{'n':>6}{'Anteil':>9}   Lesart")
LES = {(1, 1, 1): "alle drei", (0, 0, 0): "keiner",
       (0, 1, 1): "nur die Flow-Arme", (1, 0, 0): "nur SigmaDock",
       (0, 1, 0): "nur Minimal", (0, 0, 1): "nur Separate",
       (1, 1, 0): "SigmaDock und Minimal", (1, 0, 1): "SigmaDock und Separate"}
z = muster.groupby(ARME).size()
paare = sorted(LES, key=lambda k: -int(z.get(tuple(bool(x) for x in k), 0)))
for schl in paare:
    n_ = int(z.get(tuple(bool(x) for x in schl), 0))
    print(f"  {schl[0]:>4}{schl[1]:>5}{schl[2]:>5}{n_:6d}"
          f"{100 * n_ / N:8.1f}%   {LES[schl]}")

print("\n  Gepaarter Test (McNemar) auf Top-1:")
for arm in ARME[1:]:
    x, y = muster.SigmaDock.to_numpy(), muster[arm].to_numpy()
    na, nb = int((x & ~y).sum()), int((y & ~x).sum())
    pv = binomtest(nb, na + nb, 0.5).pvalue if na + nb else 1.0
    print(f"    {arm:<10}{100 * y.mean():5.1f}% gegen {100 * x.mean():5.1f}%   "
          f"diskordant {nb}:{na}   p = {pv:.4f}  {stern(pv)}")


# --- D -------------------------------------------------------------------
def ursache(r):
    if r.trifft:
        return "richtig: aus dem Hauptmodus" if r.wahl_gross \
            else "richtig: aus einem NEBENMODUS"
    if not r.hat_korrekte:
        return "falsch: keine korrekte Pose vorhanden"
    if not r.gross_richtig and r.wahl_gross:
        return "falsch: der eigenen falschen Dichte gefolgt"
    if r.gross_richtig and not r.wahl_gross:
        return "falsch: Hauptmodus war richtig, Wahl daneben"
    if r.gross_richtig and r.wahl_gross:
        return "falsch: Hauptmodus richtig, Wahl darin falsch"
    return "falsch: Dichte falsch UND Wahl ausserhalb"


REIHE = ["richtig: aus dem Hauptmodus", "richtig: aus einem NEBENMODUS",
         "falsch: keine korrekte Pose vorhanden",
         "falsch: der eigenen falschen Dichte gefolgt",
         "falsch: Hauptmodus war richtig, Wahl daneben",
         "falsch: Hauptmodus richtig, Wahl darin falsch",
         "falsch: Dichte falsch UND Wahl ausserhalb"]

print("\n=== D. Woran liegt es, wo die Arme uneinig sind? ===")
for schl in [(0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]:
    k = idx[(muster.SigmaDock == bool(schl[0]))
            & (muster.Minimal == bool(schl[1]))
            & (muster.Separate == bool(schl[2]))]
    if len(k) < 3:
        continue
    print(f"\n  --- {LES[schl]}  (n = {len(k)}) ---")
    tab = pd.DataFrame({KURZ[arm]: W[arm].loc[k].apply(ursache, axis=1)
                        .value_counts() for arm in ARME})
    tab = tab.reindex(REIHE).fillna(0).astype(int)
    tab = tab[tab.sum(axis=1) > 0]
    print(f"    {'':<44}" + "".join(f"{c:>6}" for c in tab.columns))
    for zl, wv in tab.iterrows():
        print(f"    {zl:<44}" + "".join(f"{int(x):6d}" for x in wv))
    for arm in ARME:
        d = W[arm].loc[k]
        print(f"    [{KURZ[arm]}] korrekte Posen {d.n_korrekt.mean():5.1f}/"
              f"{int(d.n.mean())}   groesster Modus {100 * d.gross.mean():4.0f} %"
              f"   davon richtig {100 * d.gross_richtig.mean():4.0f} %")

print(f"\n{len(t)} Zeilen nach {AUS}")
