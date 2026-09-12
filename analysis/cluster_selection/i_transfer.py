"""Teil I -- Uebertragung auf einen Benchmark, der nie mitgerechnet hat.

WARUM DAS DER EIGENTLICHE TEST IST
    Die gruppierte Kreuzvalidierung auf PoseBusters teilt nach Komplex, aber
    alle Falten stammen aus derselben Sammlung, mit derselben Aufbereitung,
    denselben Trefferquoten und teils verwandten Proteinen. Ein Modell kann
    darin ehrlich +2 Punkte erreichen und trotzdem nur die Eigenheiten dieser
    Sammlung gelernt haben. Astex ist unabhaengig aufgebaut, hat eine deutlich
    hoehere Basis und wurde in dieser Untersuchung nie angefasst.

DIE GEMEINSAME MERKMALSMENGE
    Astex hat lokal keine Proteinstrukturen, also keine Kontaktmerkmale.
    Trainiert und geprueft wird deshalb auf dem Durchschnitt beider Saetze.
    Damit die Zahl einzuordnen ist, laeuft dieselbe Menge auch noch einmal in
    der Kreuzvalidierung auf PoseBusters -- so ist sichtbar, wie viel der
    Verlust dem fehlenden Kontaktanteil und wie viel der Uebertragung
    zuzuschreiben ist.

ZWEI RICHTUNGEN
    Auch der umgekehrte Weg wird gerechnet (Astex trainieren, PoseBusters
    pruefen). Er ist mit 85 Komplexen schwach, aber wenn ein Effekt in beide
    Richtungen traegt, ist er wahrscheinlich echt; traegt er in keine, war die
    Kreuzvalidierung optimistisch.

DAZU EIN NEBENBEFUND, DER GEPRUEFT WERDEN MUSS
    Teil G zeigt: unloesbare Komplexe haben im Mittel 47 Cluster, getroffene
    19. Die Clusterzahl sagt also moeglicherweise vorher, OB ein Komplex
    loesbar ist -- eine Konfidenzaussage, keine Auswahlregel. Das wird hier
    sauber als AUC auf Komplexebene gemessen, auf beiden Saetzen.

Aufruf:
    python analysis/cluster_selection/i_transfer.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
SCHL = ["arm", "nfe", "complex"]
ZIEL = "best_pose_correct"
S = 2.0
rng = np.random.default_rng(20260909)


def bau():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, max_leaf_nodes=30, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def laden(name):
    t = pd.read_csv(os.path.join(_HIER, f"cluster_{name}.csv"))
    return t[t.S == S].copy().reset_index(drop=True)


tp, ta = laden("pb308"), laden("astex")
GEM = sorted(set(c for c in tp.columns if c.endswith("__r"))
             & set(c for c in ta.columns if c.endswith("__r")))
NUR_PB = sorted(set(c for c in tp.columns if c.endswith("__r")) - set(GEM))
for t in (tp, ta):
    t[GEM] = t[GEM].fillna(0.5)

print("#" * 96)
print(f"  Gemeinsame Merkmale beider Saetze: {len(GEM)}")
print(f"  Nur auf PoseBusters vorhanden    : {len(NUR_PB)} "
      f"(die Kontaktfamilie -- Astex hat lokal keine Proteine)")
print("#" * 96)


def auswahl(t, punkte):
    tmp = t[SCHL].copy()
    tmp["_p"] = punkte
    tmp["_y"] = t[ZIEL].to_numpy(bool)
    return tmp.loc[tmp.groupby(SCHL, sort=False)["_p"].idxmax()] \
              .set_index(SCHL)["_y"]


def basis_von(t):
    return t[t.selected_by_ranker].set_index(SCHL)[ZIEL].astype(bool)


def bericht(name, t, punkte, boot=4000):
    r = auswahl(t, punkte)
    b = basis_von(t).reindex(r.index)
    o = t.groupby(SCHL)[ZIEL].max().reindex(r.index)
    kod = r.index.get_level_values("complex").to_numpy()
    kx = np.unique(kod)
    idx = {c: np.where(kod == c)[0] for c in kx}
    r_, b_ = r.to_numpy(float), b.to_numpy(float)
    d = np.empty(boot)
    for i in range(boot):
        j = np.concatenate([idx[c] for c in rng.choice(kx, len(kx), True)])
        d[i] = 100 * (r_[j].mean() - b_[j].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    g = 100 * (r.mean() - b.mean())
    sp = 100 * (o.mean() - b.mean())
    print(f"  {name:<42}{100*b.mean():7.2f}%{100*r.mean():8.2f}%{g:+8.2f}"
          f"  [{lo:+6.2f},{hi:+6.2f}]{int((r & ~b).sum()):7d}"
          f"{int((~r & b).sum()):7d}{100*g/sp if sp > 0 else np.nan:8.1f}%")
    return g, lo, hi


def kreuz(t, sp_, falten=5):
    """Gruppierte Kreuzvalidierung nach Komplex, Skalierung im Falt."""
    kx = np.array(sorted(t["complex"].unique()))
    np.random.default_rng(20260909).shuffle(kx)
    f = t["complex"].map({c: i % falten for i, c in enumerate(kx)})
    y = t[ZIEL].to_numpy(bool)
    pkt = np.zeros(len(t))
    for k in range(falten):
        tr, te = (f != k).to_numpy(), (f == k).to_numpy()
        sc = StandardScaler().fit(t.loc[tr, sp_])
        m = bau().fit(sc.transform(t.loc[tr, sp_]), y[tr])
        pkt[te] = m.predict_proba(sc.transform(t.loc[te, sp_]))[:, 1]
    return pkt


def transfer(t_tr, t_te, sp_):
    sc = StandardScaler().fit(t_tr[sp_])
    m = bau().fit(sc.transform(t_tr[sp_]), t_tr[ZIEL].to_numpy(bool))
    return m.predict_proba(sc.transform(t_te[sp_]))[:, 1]


kopf = "Prueffall"
print(f"\n  {kopf:<42}{'Basis':>7}{'Modell':>8}{'Gewinn':>8}{'95%-KI':>17}"
      f"{'rett.':>7}{'zerst.':>7}{'Luecke':>9}")

print("\n  --- Bezugswerte: Kreuzvalidierung im eigenen Satz ---")
bericht("PB308, alle Merkmale (aus Teil E)", tp,
        kreuz(tp, sorted(set(GEM) | set(NUR_PB))))
bericht("PB308, nur gemeinsame Merkmale", tp, kreuz(tp, GEM))
bericht("ASTEX, nur gemeinsame Merkmale", ta, kreuz(ta, GEM))

print("\n  --- Uebertragung: trainiert auf dem einen, geprueft auf dem anderen ---")
bericht("PB308 trainiert -> ASTEX geprueft", ta, transfer(tp, ta, GEM))
bericht("ASTEX trainiert -> PB308 geprueft", tp, transfer(ta, tp, GEM))

print("\n  --- Uebertragung je Zelle, PB308 -> ASTEX ---")
pkt = transfer(tp, ta, GEM)
for arm, nfe in ZELLEN:
    m = ((ta.arm == arm) & (ta.nfe == nfe)).to_numpy()
    if m.sum():
        bericht(f"{arm}, {nfe} Schritte", ta[m], pkt[m], boot=2000)

# ---------------------------------------------------------------- Nebenbefund
print("\n" + "=" * 96)
print("  NEBENBEFUND: sagt die Clusterzahl vorher, OB ein Komplex loesbar ist?")
print("  Das waere eine Konfidenzaussage, keine Auswahlregel -- eine andere,")
print("  aber ebenfalls nuetzliche Groesse.")
print("=" * 96)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


kopf2 = "Groesse"
print(f"\n  {kopf2:<26}{'PB308 loesbar':>15}{'PB308 Treffer':>15}"
      f"{'ASTEX loesbar':>15}{'ASTEX Treffer':>15}")
for nm, sp_ in (("Zahl der Cluster", "a_n_cluster"),
                ("Entropie der Massen", "a_masse_ent"),
                ("Groesse des groessten", "g_gr"),
                ("Konkurrenten im Band 0,5", "a_n_im_band05"),
                ("bester Mixed Score", "s_h_max")):
    z = []
    for t in (tp, ta):
        gk = t.groupby(SCHL)
        x = gk[sp_].max() if sp_ != "a_n_cluster" else gk[sp_].first()
        loesbar = gk[ZIEL].max().astype(bool)
        treffer = basis_von(t).reindex(x.index).astype(bool)
        # Vorzeichen so drehen, dass hoehere Werte "loesbar" bedeuten sollen
        z += [auc(x, loesbar), auc(x, treffer)]
    print(f"  {nm:<26}{z[0]:15.3f}{z[1]:15.3f}{z[2]:15.3f}{z[3]:15.3f}")
print("\n  AUC unter 0,5 heisst: das Merkmal zeigt in die Gegenrichtung, ist")
print("  also nach Vorzeichenwechsel ebenso informativ. 0,5 heisst wertlos.")
