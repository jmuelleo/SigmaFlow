"""Teil N -- Clusterkonzentration oder Mixed Score? Der direkte Vergleich.

DIE FRAGE
    "Vertrauen wir dieser Pose?" -- welche Groesse beantwortet das besser?

DIE KANDIDATEN
    score        der Mixed Score der ausgelieferten Pose
    rueckhalt    Anteil aller Ziehungen im SELBEN Cluster wie die
                 ausgelieferte Pose. Das ist die natuerlichste Fassung der
                 Frage: wie viele meiner Ziehungen stimmen mit dem ueberein,
                 was ich gerade abgebe? Bisher gar nicht geprueft.
    groesster    Anteil des groessten Clusters (unabhaengig davon, ob die
                 ausgelieferte Pose darin liegt)
    p_agree      Anteil ALLER Posenpaare unter der Schwelle. Kommt ohne
                 Clusterdefinition aus und ist damit robuster gegen die
                 willkuerliche Wahl von Verkettung und Schnitt.
    entropie     normierte Entropie der Clustermassen
    n_cluster    Zahl der Cluster

    p_agree wird aus den Abstandsmatrizen direkt gerechnet, nicht aus den
    Clustergroessen geschaetzt: bei vollstaendiger Verkettung liegen zwar alle
    Paare INNERHALB eines Clusters unter der Schwelle, aber es kann auch
    Paare zwischen zwei Clustern geben, die darunter liegen. Die Schaetzung
    aus Clustergroessen waere also systematisch zu klein.

DREI GESTAFFELTE ZIELE, WEIL SIE VERSCHIEDENE DINGE MESSEN
    loesbar    Wurde ueberhaupt eine korrekte Pose gezogen?  (leicht)
    treffer    Ist die ausgelieferte Pose korrekt?           (operativ)
    gewaehlt   Richtig gewaehlt, GEGEBEN es lag etwas vor?   (schwer)

    Es ist gut moeglich, dass die Konzentration bei einer anderen Stufe
    traegt als der Score. Genau das waere der Grund, beide zu benutzen.

Aufruf:
    python analysis/cluster_selection/n_konfidenzwahl.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HIER = os.path.dirname(os.path.abspath(__file__))
_VAR = os.path.join(os.path.dirname(os.path.dirname(_HIER)), "SigmaFlow_Variants")
sys.path.insert(0, _VAR)
import posencache  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
SCHL = ["arm", "nfe", "complex"]
S = 2.0
rng = np.random.default_rng(20260909)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tabelle(satz):
    C = pd.read_csv(os.path.join(_HIER, f"cluster_{satz}.csv"))
    C = C[C.S == S]
    g = C.groupby(SCHL)
    w = C[C.selected_by_ranker].set_index(SCHL)
    V = pd.DataFrame({
        "score": g["s_h_max"].max(),
        "rueckhalt": w["g_relgr"],
        "groesster": g["g_relgr"].max(),
        "entropie": g["a_masse_ent_norm"].first(),
        "n_cluster": g["a_n_cluster"].first(),
        "loesbar": g["best_pose_correct"].max().astype(bool),
        "treffer": w["best_pose_correct"].astype(bool),
    })
    # p_agree direkt aus den Abstandsmatrizen
    pa = {}
    for arm, nfe in ZELLEN:
        try:
            dat = posencache.hole(satz, arm, nfe, leise=True)
        except Exception:
            continue
        for code, v in dat.items():
            d = v["d"].astype(float)
            n = d.shape[0]
            iu = np.triu_indices(n, 1)
            pa[(arm, nfe, code)] = float((d[iu] < S).mean())
    V["p_agree"] = pd.Series(pa).reindex(V.index)
    return V.reset_index()


MASSE = [("Mixed Score", "score", +1),
         ("Rueckhalt der Pose", "rueckhalt", +1),
         ("groesster Cluster", "groesster", +1),
         ("p_agree unter 2 A", "p_agree", +1),
         ("Entropie der Massen", "entropie", -1),
         ("Zahl der Cluster", "n_cluster", -1)]

print("#" * 96)
print("  Clusterkonzentration oder Mixed Score -- was traegt die Frage")
print("  'vertrauen wir dieser Pose?'")
print("#" * 96)
print("\n  Vorzeichen sind so gedreht, dass hoehere Werte 'vertrauenswuerdig'")
print("  bedeuten sollen. AUC unter 0,5 hiesse: das Mass zeigt in die")
print("  Gegenrichtung.\n")

DATEN = {}
for satz in ("pb308", "astex"):
    V = tabelle(satz)
    DATEN[satz] = V
    L = V[V.loesbar]
    print(f"  === {satz.upper()} ({len(V)} Faelle, "
          f"{int((~V.loesbar).sum())} unloesbar) ===")
    kopf = "Mass"
    print(f"  {kopf:<24}{'treffer':>10}{'loesbar':>10}{'gewaehlt':>11}"
          f"{'(operativ)':>13}")
    for nm, sp, vz in MASSE:
        a1 = auc(vz * V[sp], V.treffer)
        a2 = auc(vz * V[sp], V.loesbar)
        a3 = auc(vz * L[sp], L.treffer)
        print(f"  {nm:<24}{a1:10.3f}{a2:10.3f}{a3:11.3f}")
    print()

print("  Spalte 'treffer' ist die operative: ist die ausgelieferte Pose korrekt?")
print("  'loesbar' und 'gewaehlt' zerlegen sie in die leichte und die schwere")
print("  Teilaufgabe.\n")

# ---- Kombination -------------------------------------------------------
print("=" * 96)
print("  Bringt die Kombination mehr als der Mixed Score allein?")
print("  Logistische Regression, fuenf Falten gruppiert nach Komplex.")
print("=" * 96)


def kreuz(V, spalten, ziel="treffer", falten=5):
    D = V.dropna(subset=spalten).copy()
    kx = np.array(sorted(D["complex"].unique()))
    np.random.default_rng(20260909).shuffle(kx)
    f = D["complex"].map({c: i % falten for i, c in enumerate(kx)}).to_numpy()
    y = D[ziel].to_numpy(bool)
    pkt = np.zeros(len(D))
    for k in range(falten):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(D.loc[tr, spalten])
        m = LogisticRegression(max_iter=2000).fit(
            sc.transform(D.loc[tr, spalten]), y[tr])
        pkt[te] = m.predict_proba(sc.transform(D.loc[te, spalten]))[:, 1]
    return D, pkt, auc(pkt, y)


for satz in ("pb308", "astex"):
    V = DATEN[satz]
    D0, p0, a0 = kreuz(V, ["score"])
    print(f"\n  {satz.upper()}    nur Mixed Score: AUC {a0:.3f}")
    kod = D0["complex"].to_numpy()
    kx = np.unique(kod)
    idx = {c: np.where(kod == c)[0] for c in kx}
    y = D0.treffer.to_numpy(bool)
    kopf2 = "zusaetzlich"
    print(f"    {kopf2:<26}{'AUC':>8}{'Differenz':>12}{'95%-KI':>18}")
    for nm, sp, _ in MASSE[1:]:
        D1, p1, a1 = kreuz(V, ["score", sp])
        if len(D1) != len(D0):
            continue
        d = np.empty(2000)
        for i in range(2000):
            j = np.concatenate([idx[c] for c in rng.choice(kx, len(kx), True)])
            d[i] = auc(p1[j], y[j]) - auc(p0[j], y[j])
        lo, hi = np.percentile(d, [2.5, 97.5])
        st = " *" if lo > 0 else ""
        print(f"    {nm:<26}{a1:8.3f}{a1 - a0:+12.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{st}")
    Dv, pv, av = kreuz(V, ["score", "rueckhalt", "p_agree", "n_cluster"])
    print(f"    {'alles zusammen':<26}{av:8.3f}{av - a0:+12.3f}")

# ---- Wie stark haengen sie zusammen? ----------------------------------
print("\n" + "=" * 96)
print("  Sind die Masse ueberhaupt komplementaer? (Spearman auf PB308)")
print("=" * 96)
V = DATEN["pb308"]
sp = [m[1] for m in MASSE]
K = V[sp].corr(method="spearman")
kopf3 = ""
print(f"\n  {kopf3:<20}" + "".join(f"{m[0][:11]:>13}" for m in MASSE))
for nm, s_, _ in MASSE:
    print(f"  {nm:<20}" + "".join(f"{K.loc[s_, m[1]]:13.2f}" for m in MASSE))
print("\n  Werte nahe 0 zwischen Score und den Clustermassen heissen: die")
print("  Information ist weitgehend unabhaengig, eine Kombination lohnt sich.")

# ---- Praktische Markierung --------------------------------------------
print("\n" + "=" * 96)
print("  Praktisch: die schwaechsten x Prozent je Zelle als unsicher markieren")
print("=" * 96)
for satz in ("pb308", "astex"):
    V = DATEN[satz]
    D, pk, ak = kreuz(V, ["score", "rueckhalt", "p_agree", "n_cluster"])
    D = D.copy()
    D["_komb"] = pk
    print(f"\n  {satz.upper()}, Basisrate {100*D.treffer.mean():.1f} % Treffer")
    kopf4 = "markiert"
    print(f"  {kopf4:>9}" + "".join(f"{nm[:20]:>22}"
                                    for nm in ("nur Mixed Score",
                                               "nur Rueckhalt",
                                               "kombiniert")))
    for q in (0.10, 0.20, 0.30):
        aus = f"  {100*q:8.0f}%"
        for sp_ in ("score", "rueckhalt", "_komb"):
            fang, rest = [], []
            for arm, nfe in ZELLEN:
                s = D[(D.arm == arm) & (D.nfe == nfe)]
                if not len(s):
                    continue
                mark = s[sp_] <= s[sp_].quantile(q)
                fang.append((mark & ~s.treffer).sum()
                            / max((~s.treffer).sum(), 1))
                rest.append(s.loc[~mark, "treffer"].mean())
            aus += f"{100*np.mean(fang):11.1f}% /{100*np.mean(rest):8.1f}%"
        print(aus)
    print("  je Spalte: Anteil der gefangenen Fehlgriffe / Trefferquote im Rest")
