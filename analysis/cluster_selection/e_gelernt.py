"""Teil E/F -- gelernte Clusterselektoren unter strenger Kreuzvalidierung.

DIE ZIELGROESSE, DIE ZUR ENTSCHEIDUNG PASST
    Eine Regel waehlt einen Cluster und nimmt darin die bestbewertete Pose.
    Ihr Erfolg ist damit genau `best_pose_correct` des gewaehlten Clusters --
    nicht `contains_correct`. Ein Cluster kann eine korrekte Pose enthalten und
    trotzdem die falsche herausgeben. Trainiert wird deshalb primaer auf
    `best_pose_correct`; `contains_correct` laeuft als Kontrolle mit.

DIE AUFTEILUNG
    Fuenf Falten, gruppiert nach KOMPLEX. Ein Komplex steckt in allen fuenf
    Zellen; die Gruppierung erfolgt darum ueber die Zellen hinweg gemeinsam.
    Standardisierung und Merkmalsauswahl passieren INNERHALB des
    Trainingsfaltes -- sonst sieht das Modell die Teststreuung.

DIE MERKMALSMENGEN (Ablation)
    Die entscheidende Frage ist nicht "erreicht ein Modell 70 Prozent", sondern
    "gibt es Signal, das NICHT schon im Mixed Score steckt". Deshalb laeuft
    jedes Modell auf gestaffelten Mengen: nur der Score; alles; und alles OHNE
    Score-Merkmale. Wenn die letzte Menge nichts kann, ist die Antwort klar.

DER VERGLEICH
    Gepaarter Bootstrap auf KOMPLEXebene (nicht auf Clusterebene -- Cluster
    desselben Komplexes sind nicht unabhaengig). 2000 Ziehungen, 95-Prozent-
    Intervall auf die DIFFERENZ zur Basis.

Aufruf:
    python analysis/cluster_selection/e_gelernt.py --satz pb308 --S 2.0
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--S", type=float, default=2.0)
p.add_argument("--ziel", default="best_pose_correct")
p.add_argument("--falten", type=int, default=5)
p.add_argument("--boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=20260909)
a = p.parse_args()

t = pd.read_csv(os.path.join(_HIER, f"cluster_{a.satz}.csv"))
t = t[t.S == a.S].copy().reset_index(drop=True)
rng = np.random.default_rng(a.seed)

# Nur die rangnormierten Merkmale: absolute Werte sind zwischen Komplexen
# nicht vergleichbar, und das Modell entscheidet immer innerhalb eines.
MERK = [c for c in t.columns if c.endswith("__r")]
MERK = [c for c in MERK if t[c].notna().mean() > 0.9]
t[MERK] = t[MERK].fillna(0.5)

SCORE = [c for c in MERK if c.startswith("s_")]
PB = [c for c in MERK if c.startswith(("p_", "pc_"))]
GEO = [c for c in MERK if c.startswith("g_")]
KON = [c for c in MERK if c.startswith("k_")]
AMB = [c for c in MERK if c.startswith("a_")]
MENGEN = {
    "nur Mixed-Score-Max": ["s_h_max__r"],
    "nur Score-Familie": SCORE,
    "nur Geometrie": GEO,
    "nur Kontakte": KON,
    "nur PoseBusters": PB,
    "OHNE Score (G+K+P+A)": GEO + KON + PB + AMB,
    "alles": MERK,
}

print(f"########## {a.satz.upper()}, Schwelle {a.S} A, Ziel {a.ziel} ##########")
print(f"{len(t)} Cluster, {len(MERK)} Merkmale "
      f"(Score {len(SCORE)}, PB {len(PB)}, Geo {len(GEO)}, "
      f"Kontakte {len(KON)}, Ambig {len(AMB)})\n")

# ---- Falten nach Komplex, ueber alle Zellen gemeinsam -------------------
kx = np.array(sorted(t["complex"].unique()))
rng.shuffle(kx)
falte = {c: i % a.falten for i, c in enumerate(kx)}
t["_f"] = t["complex"].map(falte)
Y = t[a.ziel].to_numpy(bool)
SCHL = ["arm", "nfe", "complex"]


def auswahl(punkte):
    """Je Komplex und Zelle den hoechstbewerteten Cluster; Erfolg nachschlagen."""
    tmp = t[SCHL].copy()
    tmp["_p"] = punkte
    tmp["_y"] = Y
    i = tmp.groupby(SCHL, sort=False)["_p"].idxmax()
    return tmp.loc[i].set_index(SCHL)["_y"]


basis = auswahl(t["s_h_max"].to_numpy(float))
orakel = t.groupby(SCHL)[a.ziel].max()
print(f"Basis {100*basis.mean():.2f} %, Cluster-Orakel {100*orakel.mean():.2f} %, "
      f"Spielraum {100*(orakel.mean()-basis.mean()):+.2f} Punkte\n")

MODELLE = {
    "Logistisch": lambda: LogisticRegression(C=0.1, max_iter=2000),
    "Baum(2)": lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=200,
                                              random_state=0),
    "Baum(3)": lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=100,
                                              random_state=0),
    # Histogramm-Boosting: bindet die Merkmale in 255 Faecher und rechnet
    # damit dasselbe wie das klassische Boosting, aber in Sekunden statt
    # Minuten. Flach gehalten (Tiefe 3, 30 Blaetter), weil ein tiefes Modell
    # auf 140.000 Clustern die Komplexidentitaet auswendig lernen wuerde.
    "Boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, max_leaf_nodes=30, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=False, random_state=0),
}


# Die Bootstrapgruppen haengen nur an der Zeilenreihenfolge, die ueber alle
# Modelle gleich ist -- einmal bilden statt 28-mal.
_kx = None
_idx = None


def bootstrap(neu, alt):
    """Gepaarte Differenz, Bootstrap ueber KOMPLEXE (nicht ueber Cluster).

    Cluster und Zellen desselben Komplexes sind nicht unabhaengig; Ziehen auf
    Zeilenebene wuerde das Intervall kuenstlich verengen.
    """
    global _kx, _idx
    kod = neu.index.get_level_values("complex").to_numpy()
    if _kx is None:
        _kx = np.unique(kod)
        _idx = {c: np.where(kod == c)[0] for c in _kx}
    n_, a_ = neu.to_numpy(float), alt.to_numpy(float)
    d = np.empty(a.boot)
    for i in range(a.boot):
        j = np.concatenate([_idx[c] for c in rng.choice(_kx, len(_kx), True)])
        d[i] = 100 * (n_[j].mean() - a_[j].mean())
    return np.percentile(d, [2.5, 97.5])


erg = []
baeume = {}
for mname, bau in MODELLE.items():
    for sname, sp in MENGEN.items():
        if not sp:
            continue
        pkt = np.zeros(len(t))
        for f in range(a.falten):
            tr, te = t["_f"] != f, t["_f"] == f
            sc = StandardScaler().fit(t.loc[tr, sp])
            m = bau()
            m.fit(sc.transform(t.loc[tr, sp]), Y[tr.to_numpy()])
            pkt[te.to_numpy()] = m.predict_proba(sc.transform(t.loc[te, sp]))[:, 1]
            if f == 0 and mname == "Baum(2)" and sname == "alles":
                baeume[sname] = export_text(m, feature_names=list(sp),
                                            max_depth=2)
        r = auswahl(pkt)
        lo, hi = bootstrap(r, basis.reindex(r.index))
        gew = 100 * (r.mean() - basis.mean())
        erg.append({"modell": mname, "menge": sname, "erfolg": 100 * r.mean(),
                    "gewinn": gew, "lo": lo, "hi": hi,
                    "gerettet": int((r & ~basis.reindex(r.index)).sum()),
                    "zerstoert": int((~r & basis.reindex(r.index)).sum()),
                    "luecke": 100 * gew / (100 * (orakel.mean() - basis.mean()))})

E = pd.DataFrame(erg)
E.to_csv(os.path.join(_HIER, f"e_gelernt_{a.satz}_{a.S}.csv"), index=False)
print(f"  {'Modell':<12}{'Merkmalsmenge':<24}{'Erfolg':>9}{'vs Basis':>10}"
      f"{'95%-KI':>18}{'gerettet':>10}{'zerstoert':>11}{'Luecke':>9}")
for mname in MODELLE:
    for _, r in E[E.modell == mname].iterrows():
        stern = " *" if r.lo > 0 else ("  " if r.hi > 0 else " -")
        print(f"  {r.modell:<12}{r.menge:<24}{r.erfolg:8.2f}%{r.gewinn:+10.2f}"
              f"   [{r.lo:+6.2f},{r.hi:+6.2f}]{r.gerettet:10d}{r.zerstoert:11d}"
              f"{r.luecke:8.1f}%{stern}")
    print()

print("* = 95-Prozent-Intervall der Differenz liegt vollstaendig ueber null\n")
if baeume:
    print("=== Der flache Baum auf allen Merkmalen (Falte 1) ===")
    print(baeume["alles"])
