"""Wie entwickelt sich die Unsicherheit der Flow-Arme mit der Ziehungszahl?

DIE FRAGE
    Der Anteil der Ziehungen im groessten Cluster ist ein brauchbares
    Vertrauensmass (siehe konfidenz.py): bei K = 40 trennt er richtig von
    falsch mit AUC 0,79 bis 0,84. Offen ist, wie sich das mit K aendert.

    Drei Groessen, die getrennt zu halten sind:

      NIVEAU   Der mittlere Anteil im groessten Cluster. Er MUSS mit K fallen:
               mehr Ziehungen finden Nebenmoden, die bei wenigen Ziehungen
               unbesetzt bleiben. Ein Rueckgang ist also kein Befund, sondern
               Arithmetik. Interessant ist, wie schnell er faellt und wo er
               sich beruhigt -- das sagt, ab wann die Verteilung ausgeschoepft
               ist.

      TRENNUNG Niveau bei Treffern gegen Niveau bei Fehlgriffen. Die Differenz
               ist der eigentliche Inhalt: liegt das Modell daneben, wenn es
               sich nicht festgelegt hat?

      AUC      Dieselbe Trennung, rangbasiert und damit gegen das fallende
               Niveau immun. Das ist die einzige der drei Groessen, die man
               ueber verschiedene K hinweg vergleichen darf.

WARUM NUR DIE FLOW-ARME BEI FUENF SCHRITTEN
    Nur dort liegen 200 Seeds lokal. SigmaDock hat bei beiden Schrittzahlen
    40, die Flow-Arme bei 25 Schritten ebenfalls 40. Die Kurve endet also
    dort, wo die Zelle endet -- es wird nicht extrapoliert.

METHODIK, WIE UEBERALL SONST
    Je Komplex und je K werden 1000 Teilmengen der Groesse K gezogen. In jeder
    wird neu geclustert und neu gerankt; der Mittelwert ueber die Ziehungen ist
    der Wert des Komplexes. Erst darueber wird ueber die Komplexe gemittelt.

    Die AUC dagegen wird JE ZIEHUNG ueber die 67 Komplexe gerechnet und dann
    ueber die Ziehungen gemittelt. Ueber gemittelte Konzentrationen zu ranken
    waere etwas anderes -- es verwischt genau die Streuung, um die es geht.

WAS TEUER IST UND WAS NICHT
    Teuer ist das Einlesen der 13 400 SDF je Arm und die 200x200-RMSD-Matrix
    je Komplex. Beides haengt nicht von K ab und wird deshalb einmal gerechnet
    und als .npz abgelegt. Das Clustern einer Teilmenge ist danach billig: es
    liest nur einen Ausschnitt der fertigen Matrix.

    Diese Trennung -- teure, K-unabhaengige Vorstufe cachen, billige Schleife
    darueber -- ist der uebliche Weg, aus einer Einzelmessung eine Kurve zu
    machen, ohne die Messung K-mal zu wiederholen.

Aufruf:
    python SigmaFlow_Variants/unsicherheit_k.py --nur-cache
    python SigmaFlow_Variants/unsicherheit_k.py
    python SigmaFlow_Variants/unsicherheit_k.py --schwelle 1.5 --ziehungen 200
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, lade_zelle, posenordner  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except ImportError:
    sys.exit("ABBRUCH: scipy fehlt. pip install scipy")

KS = [2, 3, 5, 10, 20, 40, 60, 100, 140, 200]
ZELLEN = [("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwelle", type=float, default=2.0)
p.add_argument("--ziehungen", type=int, default=1000)
p.add_argument("--seed", type=int, default=20260908)
p.add_argument("--nur-cache", action="store_true")
p.add_argument("--out", default=os.path.join(_HIER, "unsicherheit_k.csv"))
a = p.parse_args()


def lade(pfad):
    m = Chem.MolFromMolFile(pfad, sanitize=False, removeHs=False)
    if m is None or m.GetNumConformers() == 0:
        return None, None
    return (m.GetConformer().GetPositions(),
            tuple(at.GetSymbol() for at in m.GetAtoms()))


def cache_bauen(arm, nfe):
    """RMSD-Matrix, Trefferflags und Scores je Komplex -- einmal, dann auf Platte.

    Die Matrix ist symmetrisch und wird als float32 abgelegt: 200x200 sind so
    160 kB je Komplex, fuer 67 Komplexe rund 11 MB. In float64 waere es das
    Doppelte ohne jeden Gewinn -- RMSD-Werte tragen keine 15 Stellen.
    """
    pfad = os.path.join(_HIER, f"cache_unsicherheit_{a.satz}_{arm}_nfe{nfe}.npz")
    if os.path.isfile(pfad):
        print(f"  {arm} nfe{nfe}: Cache vorhanden")
        return pfad
    z = next(z for z in SAETZE[a.satz]["zellen"]
             if z["arm"] == arm and z["nfe"] == nfe)
    w = posenordner(a.satz, arm, nfe)
    if not os.path.isdir(w):
        sys.exit(f"ABBRUCH: {w} fehlt.")
    tab = lade_zelle(z, leise=True)

    ablage = {}
    codes = sorted(tab["complex"].unique())
    for i, code in enumerate(codes, 1):
        g = tab[tab["complex"] == code]
        acc_von = dict(zip(g["seed"], g["acc"]))
        heur_von = dict(zip(g["seed"], g["heur"]))
        dateien = sorted(glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")),
                         key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        x, acc, heur, ref = [], [], [], None
        for f in dateien:
            s = int(re.search(r"seed_(\d+)", f).group(1))
            if s not in acc_von:
                continue
            k, el = lade(f)
            if k is None:
                continue
            if ref is None:
                ref = el
            elif el != ref:
                # Andere Atomfolge heisst: paarweise RMSD vergleichen Atom i
                # mit Atom i, aber i bedeutet in beiden Posen etwas anderes.
                # Das waere still falsch, deshalb Abbruch statt Ueberspringen.
                sys.exit(f"ABBRUCH: {f} hat eine andere Atomfolge.")
            x.append(k)
            acc.append(bool(acc_von[s]))
            heur.append(float(heur_von[s]))
        if len(x) < 3:
            continue
        x = np.asarray(x, float)
        d = np.sqrt(((x[:, None, :, :] - x[None, :, :, :]) ** 2).sum(-1).mean(-1))
        np.fill_diagonal(d, 0.0)
        ablage[f"{code}__d"] = d.astype(np.float32)
        ablage[f"{code}__acc"] = np.asarray(acc, bool)
        ablage[f"{code}__heur"] = np.asarray(heur, float)
        if i % 10 == 0 or i == len(codes):
            print(f"    {arm} nfe{nfe}: {i}/{len(codes)} Komplexe", flush=True)
    np.savez_compressed(pfad, **ablage)
    print(f"  -> {pfad}")
    return pfad


def kennzahlen(d, teil, schwelle):
    """Konzentration, Modenzahl und normierte Entropie einer Teilmenge."""
    n = len(teil)
    if n < 2:
        return 1.0, 1, 0.0
    sub = d[np.ix_(teil, teil)].astype(float)
    Z = linkage(squareform(sub, checks=False), method="average")
    lab = fcluster(Z, t=schwelle, criterion="distance")
    g = np.bincount(lab)[1:]
    pk = g / n
    ent = float(-(pk * np.log(pk)).sum() / np.log(n)) if n > 1 else 0.0
    return float(g.max() / n), int(len(g)), ent


def auc(x, y):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(x, float)).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print(f"Cache ({a.satz}) ...")
cache = {(arm, nfe): cache_bauen(arm, nfe) for arm, nfe in ZELLEN}
if a.nur_cache:
    sys.exit(0)

rng = np.random.default_rng(a.seed)
zeilen = []
for arm, nfe in ZELLEN:
    z = np.load(cache[(arm, nfe)])
    codes = sorted({k.split("__")[0] for k in z.files})
    D = {c: z[f"{c}__d"] for c in codes}
    ACC = {c: z[f"{c}__acc"] for c in codes}
    HEUR = {c: z[f"{c}__heur"] for c in codes}
    n_max = min(len(ACC[c]) for c in codes)
    print(f"\n{arm}, {nfe} Schritte: {len(codes)} Komplexe, "
          f"mindestens {n_max} Posen je Komplex")

    for K in KS:
        if K > n_max:
            continue
        R = 1 if K == n_max else a.ziehungen
        # [Ziehung, Komplex] -- so laesst sich hinterher sowohl ueber die
        # Komplexe (fuer die AUC je Ziehung) als auch ueber die Ziehungen
        # (fuer den Wert je Komplex) mitteln.
        kon = np.empty((R, len(codes)))
        mod = np.empty((R, len(codes)))
        ent = np.empty((R, len(codes)))
        tr = np.empty((R, len(codes)), bool)
        for j, c in enumerate(codes):
            n = len(ACC[c])
            acc, heur, d = ACC[c], HEUR[c], D[c]
            for r in range(R):
                teil = (np.arange(n) if K == n
                        else rng.choice(n, K, replace=False))
                kon[r, j], mod[r, j], ent[r, j] = kennzahlen(d, teil, a.schwelle)
                tr[r, j] = bool(acc[teil[np.argmax(heur[teil])]])
        # Werte je Komplex: erst ueber die Ziehungen mitteln.
        t_je_komplex = tr.mean(axis=0)
        # AUC dagegen je Ziehung ueber die Komplexe, dann gemittelt.
        aucs = [auc(kon[r], tr[r]) for r in range(R)]
        aucs = [v for v in aucs if not np.isnan(v)]
        # Niveau getrennt nach Ausgang, ueber alle (Ziehung, Komplex)-Paare.
        flach_k, flach_t = kon.ravel(), tr.ravel()
        zeilen.append({
            "arm": arm, "nfe": nfe, "K": K, "ziehungen": R,
            "konz": round(float(flach_k.mean()), 4),
            "konz_trifft": round(float(flach_k[flach_t].mean()), 4)
            if flach_t.any() else np.nan,
            "konz_daneben": round(float(flach_k[~flach_t].mean()), 4)
            if (~flach_t).any() else np.nan,
            "moden": round(float(mod.mean()), 2),
            "entropie": round(float(ent.mean()), 4),
            "auc": round(float(np.mean(aucs)), 4) if aucs else np.nan,
            "auc_sd": round(float(np.std(aucs)), 4) if len(aucs) > 1 else np.nan,
            "top1": round(100 * float(t_je_komplex.mean()), 2),
        })
        w = zeilen[-1]
        print(f"  K={K:3d} ({R:4d} Ziehungen)  konz {100*w['konz']:5.1f}%   "
              f"trifft {100*w['konz_trifft']:5.1f}%  daneben "
              f"{100*w['konz_daneben']:5.1f}%   AUC {w['auc']:.3f}   "
              f"Top-1 {w['top1']:5.1f}%", flush=True)

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")
