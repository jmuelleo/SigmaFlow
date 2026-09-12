"""Teil K -- taugt die Clustergroesse als Konfidenzmass? Sauber gerechnet.

DER STOERFAKTOR, DER DIE GEPOOLTE ZAHL AUFBLAEHT
    In Teil I wurde ueber alle fuenf Zellen gepoolt. Das ist hier nicht
    harmlos: die 5-Schritt-Zellen ziehen 200 Posen, die 25-Schritt-Zellen 40.
    Der groesste Cluster ist damit in absoluten Posen um ein Vielfaches
    groesser -- und ausgerechnet die 200er-Zellen haben auch die hoehere
    Trefferquote (72,3 und 70,4 gegen 64,5 bis 68,1 Prozent). Ein Teil der
    gepoolten AUC misst deshalb nur, aus welcher Zelle die Zeile stammt.

    Die Abhilfe ist zweifach: JE ZELLE rechnen, und zusaetzlich den ANTEIL
    am Gesamtzug statt der absoluten Posenzahl verwenden.

DIE ZWEI ZIELE AUSEINANDERHALTEN
    loesbar  Enthaelt irgendein Cluster eine korrekte Pose? Das ist eine
             Eigenschaft des Ziehvorgangs, nicht der Auslieferung.
    Treffer  Ist die tatsaechlich ausgelieferte Pose korrekt? Das ist die
             Frage, die ein Konfidenzmass beantworten soll.

DIE FRAGE, DIE ALLEIN ZAEHLT: BRINGT ES ETWAS OBENDRAUF?
    Der Mixed Score selbst erreicht als Konfidenzmass 0,87. Ein zweites
    Merkmal ist nur dann etwas wert, wenn es die Vorhersage darueber hinaus
    verbessert. Gemessen wird das dreifach:
      1. AUC einer logistischen Regression mit beiden Merkmalen gegen die mit
         dem Mixed Score allein, gruppierte Kreuzvalidierung nach Komplex.
      2. Die Teilkorrelation: AUC der Clustergroesse INNERHALB von Schichten
         gleichen Mixed Scores (Quintile). Bleibt dort Signal, ist es echt.
      3. Gepaarter Bootstrap ueber Komplexe auf die AUC-Differenz.

Aufruf:
    python analysis/cluster_selection/k_konfidenz.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
SCHL = ["arm", "nfe", "complex"]
ZIEL = "best_pose_correct"
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


def komplexebene(name):
    """Eine Zeile je Komplex und Zelle, mit den Kandidatenmerkmalen."""
    t = pd.read_csv(os.path.join(_HIER, f"cluster_{name}.csv"))
    t = t[t.S == S]
    gk = t.groupby(SCHL)
    d = pd.DataFrame({
        "gr_abs": gk["g_gr"].max(),               # groesster Cluster, Posen
        "gr_anteil": gk["g_relgr"].max(),         # derselbe, als Anteil an K
        "n_cluster": gk["a_n_cluster"].first(),
        "entropie": gk["a_masse_ent_norm"].first(),
        "mixed": gk["s_h_max"].max(),
        "K": gk["K"].first(),
        "loesbar": gk[ZIEL].max().astype(bool),
    })
    d["treffer"] = t[t.selected_by_ranker].set_index(SCHL)[ZIEL] \
        .reindex(d.index).astype(bool)
    return d.reset_index()


P, A = komplexebene("pb308"), komplexebene("astex")
MERK = [("groesster Cluster, Posen", "gr_abs"),
        ("groesster Cluster, Anteil", "gr_anteil"),
        ("Zahl der Cluster", "n_cluster"),
        ("Entropie, normiert", "entropie"),
        ("bester Mixed Score", "mixed")]

print("#" * 100)
print("  Taugt die Clusterstruktur als Konfidenzmass?  Ziel: ist die "
      "AUSGELIEFERTE Pose korrekt?")
print("#" * 100)
print("\n  AUC unter 0,5 heisst: das Merkmal zeigt in die Gegenrichtung und ist")
print("  nach Vorzeichenwechsel gleich informativ. Ausschlaggebend ist der")
print("  Abstand zu 0,5, nicht die Seite.\n")

for nm_satz, D in (("POSEBUSTERS", P), ("ASTEX", A)):
    print(f"  === {nm_satz} ===")
    kopf = "Merkmal"
    print(f"  {kopf:<28}{'gepoolt':>10}" +
          "".join(f"{LANG[x][:9] + '/' + str(y):>14}" for x, y in ZELLEN))
    for nm, sp in MERK:
        z = [auc(D[sp], D.treffer)]
        for x, y in ZELLEN:
            s = D[(D.arm == x) & (D.nfe == y)]
            z.append(auc(s[sp], s.treffer) if len(s) else np.nan)
        warn = ""
        # Ausschlag: gepoolt deutlich weiter von 0,5 weg als jede Zelle?
        if abs(z[0] - .5) > max(abs(q - .5) for q in z[1:]) + 0.02:
            warn = "  <- gepoolt uebertrieben"
        print(f"  {nm:<28}{z[0]:10.3f}" +
              "".join(f"{q:14.3f}" for q in z[1:]) + warn)
    print()

# ---- Bringt die Clustergroesse etwas OBENDRAUF? ------------------------
print("=" * 100)
print("  Bringt die Clustergroesse etwas ueber den Mixed Score hinaus?")
print("=" * 100)


def kreuz_auc(D, spalten, ziel="treffer", falten=5, boot=2000):
    """Gruppierte Kreuzvalidierung nach Komplex; AUC mit Bootstrap-KI."""
    kx = np.array(sorted(D["complex"].unique()))
    np.random.default_rng(20260909).shuffle(kx)
    f = D["complex"].map({c: i % falten for i, c in enumerate(kx)}).to_numpy()
    y = D[ziel].to_numpy(bool)
    pkt = np.zeros(len(D))
    for k in range(falten):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(D.loc[tr, spalten])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(
            D.loc[tr, spalten]), y[tr])
        pkt[te] = m.predict_proba(sc.transform(D.loc[te, spalten]))[:, 1]
    return pkt, auc(pkt, y)


for nm_satz, D in (("POSEBUSTERS", P), ("ASTEX", A)):
    p1, a1 = kreuz_auc(D, ["mixed"])
    p2, a2 = kreuz_auc(D, ["mixed", "gr_anteil"])
    p3, a3 = kreuz_auc(D, ["mixed", "gr_anteil", "n_cluster", "entropie"])
    # Gepaarter Bootstrap ueber Komplexe auf die AUC-Differenz
    kod = D["complex"].to_numpy()
    kx = np.unique(kod)
    idx = {c: np.where(kod == c)[0] for c in kx}
    y = D.treffer.to_numpy(bool)
    d = np.empty(2000)
    for i in range(2000):
        j = np.concatenate([idx[c] for c in rng.choice(kx, len(kx), True)])
        d[i] = auc(p2[j], y[j]) - auc(p1[j], y[j])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  {nm_satz}")
    print(f"    nur Mixed Score                       AUC {a1:.3f}")
    print(f"    + Anteil des groessten Clusters       AUC {a2:.3f}   "
          f"Differenz {a2 - a1:+.3f}  [{lo:+.3f},{hi:+.3f}]"
          + ("  *" if lo > 0 else ""))
    print(f"    + Anteil, Clusterzahl, Entropie       AUC {a3:.3f}   "
          f"Differenz {a3 - a1:+.3f}")

# ---- Teilkorrelation: Signal innerhalb gleicher Score-Schichten --------
print("\n" + "=" * 100)
print("  Bleibt Signal INNERHALB von Schichten gleichen Mixed Scores?")
print("  Wenn ja, ist es eigenstaendige Information und nicht nur eine")
print("  Umschreibung des Scores. Quintile je Zelle gebildet.")
print("=" * 100)
kopf2 = "Schicht (Quintil des Mixed Scores)"
print(f"\n  {kopf2:<36}{'PB308 n':>9}{'PB308 AUC':>11}"
      f"{'ASTEX n':>9}{'ASTEX AUC':>11}")
for q in range(5):
    z = []
    for D in (P, A):
        sub = []
        for x, y in ZELLEN:
            s = D[(D.arm == x) & (D.nfe == y)].copy()
            if len(s) < 10:
                continue
            s["_q"] = pd.qcut(s["mixed"], 5, labels=False, duplicates="drop")
            sub.append(s[s._q == q])
        sub = pd.concat(sub) if sub else pd.DataFrame()
        z += [len(sub), auc(sub["gr_anteil"], sub.treffer)
              if len(sub) else np.nan]
    print(f"  {q + 1}. Quintil{'':<26}{z[0]:9d}{z[1]:11.3f}"
          f"{z[2]:9d}{z[3]:11.3f}")
alle = []
for D, nm in ((P, "PB308"), (A, "ASTEX")):
    sub = []
    for x, y in ZELLEN:
        s = D[(D.arm == x) & (D.nfe == y)].copy()
        if len(s) < 10:
            continue
        s["_q"] = pd.qcut(s["mixed"], 5, labels=False, duplicates="drop")
        sub.append(s)
    s = pd.concat(sub)
    # Mittel der Schicht-AUCs, gewichtet nach Schichtgroesse
    ws, vs = [], []
    for q in range(5):
        t_ = s[s._q == q]
        v = auc(t_["gr_anteil"], t_.treffer)
        if np.isfinite(v):
            ws.append(len(t_))
            vs.append(v)
    alle.append(np.average(vs, weights=ws))
print(f"\n  Gewichtetes Mittel ueber die Schichten:  "
      f"PB308 {alle[0]:.3f}   ASTEX {alle[1]:.3f}")
print("  (0,5 hiesse: die Clustergroesse sagt bei gleichem Mixed Score nichts mehr)")
