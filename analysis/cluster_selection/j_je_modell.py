"""Teil J -- wirkt die Methode auf allen fuenf Zellen gleich?

WARUM DAS NICHT NEBENSAECHLICH IST
    Ein Gesamtgewinn von +2 Punkten kann zwei voellig verschiedene Dinge
    bedeuten: entweder wirkt die Methode ueberall etwa gleich, dann ist sie
    ein Mechanismus; oder sie wirkt bei einer Zelle stark und bei den anderen
    gar nicht, dann beschreibt sie eine Eigenheit dieses einen Laufs. Der
    Mittelwert sieht in beiden Faellen identisch aus.

DIE PAARUNG, DIE DEN TEST SCHARF MACHT
    Alle fuenf Zellen laufen auf DENSELBEN 307 Komplexen. Der Vergleich
    zwischen zwei Zellen ist damit gepaart: fuer jeden Komplex liegen beide
    Gewinne vor. Der Bootstrap zieht deshalb ueber Komplexe und bewertet die
    DIFFERENZ der Gewinne -- das ist deutlich empfindlicher, als zwei
    unabhaengige Intervalle nebeneinanderzulegen und zu schauen, ob sie sich
    ueberlappen (was ein bekanntermassen zu konservativer Test ist).

DREI FRAGEN
    1. Wie gross ist der Gewinn je Zelle, fuer beide Methoden?
    2. Ist die Streuung zwischen den Zellen groesser als Zufall? Gemessen an
       der Spannweite max-min unter Bootstrap, und an den zehn paarweisen
       Kontrasten.
    3. Haengt der Unterschied am MODELL oder an der Ziehungszahl? Die beiden
       Achsen sind trennbar: zwei Arme laufen mit 25 und mit 5 Schritten.

Aufruf:
    python analysis/cluster_selection/j_je_modell.py
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
LANG = {"SigmaDock": "SigmaDock", "Minimal": "SF-Minimal",
        "Separate": "SF-Separate"}
SCHL = ["arm", "nfe", "complex"]
ZIEL = "best_pose_correct"
S = 2.0
DELTA = 0.5
BOOT = 4000
rng = np.random.default_rng(20260909)

t = pd.read_csv(os.path.join(_HIER, "cluster_pb308.csv"))
t = t[t.S == S].copy().reset_index(drop=True)
MERK = [c for c in t.columns if c.endswith("__r")]
MERK = [c for c in MERK if t[c].notna().mean() > 0.9]
t[MERK] = t[MERK].fillna(0.5)
Y = t[ZIEL].to_numpy(bool)


def auswahl(punkte):
    tmp = t[SCHL].copy()
    tmp["_p"] = punkte
    tmp["_y"] = Y
    return tmp.loc[tmp.groupby(SCHL, sort=False)["_p"].idxmax()] \
              .set_index(SCHL)["_y"]


# ---- die beiden Methoden -----------------------------------------------
basis = t[t.selected_by_ranker].set_index(SCHL)[ZIEL].astype(bool)
orakel = t.groupby(SCHL)[ZIEL].max()

best = t.groupby(SCHL)["s_h_max"].transform("max")
im_band = (best - t["s_h_max"]) <= DELTA
regel = auswahl(np.where(im_band,
                         100 + t["g_gr__r"] + 1e-3 * t["s_h_max__r"],
                         t["s_h_max__r"]))

kx = np.array(sorted(t["complex"].unique()))
np.random.default_rng(20260909).shuffle(kx)
f = t["complex"].map({c: i % 5 for i, c in enumerate(kx)})
pkt = np.zeros(len(t))
for k in range(5):
    tr, te = (f != k).to_numpy(), (f == k).to_numpy()
    sc = StandardScaler().fit(t.loc[tr, MERK])
    m = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, max_leaf_nodes=30, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=False,
        random_state=0).fit(sc.transform(t.loc[tr, MERK]), Y[tr])
    pkt[te] = m.predict_proba(sc.transform(t.loc[te, MERK]))[:, 1]
modell = auswahl(pkt)

# ---- Gewinn je Komplex und Zelle, als Matrix ---------------------------
# Zeilen = Komplexe, Spalten = Zellen. Die Paarung ueber Komplexe ist damit
# explizit und der Bootstrap zieht ganze Zeilen.
def matrix(reihe):
    d = (reihe.astype(float) - basis.reindex(reihe.index).astype(float))
    d = d.reset_index()
    d.columns = list(SCHL) + ["g"]
    d["zelle"] = d.arm + "|" + d.nfe.astype(str)
    return d.pivot(index="complex", columns="zelle", values="g")


SP = [f"{x}|{y}" for x, y in ZELLEN]
M_reg, M_mod = matrix(regel)[SP], matrix(modell)[SP]
B = matrix(basis)[SP] * 0 + basis.reset_index().pivot(
    index="complex", columns=[], values=ZIEL).to_numpy() if False else None
bas = basis.reset_index()
bas["zelle"] = bas.arm + "|" + bas.nfe.astype(str)
B = bas.pivot(index="complex", columns="zelle", values=ZIEL)[SP].astype(float)
ora = orakel.reset_index()
ora["zelle"] = ora.arm + "|" + ora.nfe.astype(str)
O = ora.pivot(index="complex", columns="zelle", values=ZIEL)[SP].astype(float)

# Bootstrapziehungen einmal bilden, fuer beide Methoden dieselben
n = len(M_reg)
zieh = rng.integers(0, n, size=(BOOT, n))


def je_zelle(M):
    """Gewinn je Zelle mit Intervall, plus die Bootstrapmatrix fuer spaeter."""
    a = M.to_numpy(float)
    punkt = 100 * np.nanmean(a, axis=0)
    bs = np.stack([100 * np.nanmean(a[z], axis=0) for z in zieh])
    return punkt, bs


p_reg, bs_reg = je_zelle(M_reg)
p_mod, bs_mod = je_zelle(M_mod)
p_bas = 100 * B.to_numpy(float).mean(axis=0)
p_spiel = 100 * (O.to_numpy(float).mean(axis=0) - B.to_numpy(float).mean(axis=0))

print("#" * 98)
print("  Wirkt die Clusterwahl auf allen fuenf Zellen gleich?   "
      "PoseBusters, 307 Komplexe, Schwelle 2,0 A")
print("#" * 98)
kopf = "Zelle"
print(f"\n  {kopf:<24}{'Basis':>8}{'Spielraum':>11}"
      f"{'Regel':>9}{'95%-KI':>17}{'Modell':>9}{'95%-KI':>17}")
for i, (arm, nfe) in enumerate(ZELLEN):
    lr = np.percentile(bs_reg[:, i], [2.5, 97.5])
    lm = np.percentile(bs_mod[:, i], [2.5, 97.5])
    print(f"  {LANG[arm] + ', ' + str(nfe):<24}{p_bas[i]:7.2f}%"
          f"{p_spiel[i]:+10.2f}{p_reg[i]:+9.2f}  [{lr[0]:+6.2f},{lr[1]:+6.2f}]"
          f"{p_mod[i]:+9.2f}  [{lm[0]:+6.2f},{lm[1]:+6.2f}]")
g_reg = np.percentile(bs_reg.mean(axis=1), [2.5, 97.5])
g_mod = np.percentile(bs_mod.mean(axis=1), [2.5, 97.5])
print(f"  {'GEPOOLT':<24}{p_bas.mean():7.2f}%{p_spiel.mean():+10.2f}"
      f"{p_reg.mean():+9.2f}  [{g_reg[0]:+6.2f},{g_reg[1]:+6.2f}]"
      f"{p_mod.mean():+9.2f}  [{g_mod[0]:+6.2f},{g_mod[1]:+6.2f}]")

# ---- 2. Ist die Streuung groesser als Zufall? --------------------------
print("\n=== Ist die Streuung zwischen den Zellen echt? ===")
print("  Spannweite = groesster minus kleinster Zellgewinn. Der Bootstrap")
print("  liefert das Intervall, in dem sie unter Wiederholung des Experiments")
print("  laege. Ein Intervall, das die Null erreicht, heisst: mit einheitlicher")
print("  Wirkung vereinbar.\n")
for nm, bs, p in (("Regel", bs_reg, p_reg), ("Modell", bs_mod, p_mod)):
    sp = bs.max(axis=1) - bs.min(axis=1)
    # Nullverteilung: dieselbe Spannweite, nachdem der Zelleffekt entfernt
    # wurde (jede Spalte auf ihren Gesamtmittelwert zentriert).
    # Nullverteilung: jede Bootstrapziehung auf ihren eigenen Zellmittelwert
    # zentriert. Damit ist der Zelleffekt entfernt, die Rauschstruktur und die
    # Korrelation zwischen den Zellen aber erhalten. Die Spannweite, die dabei
    # uebrig bleibt, ist die, die reines Rauschen erzeugt.
    zentriert = bs - bs.mean(axis=1, keepdims=True)
    n_sp = zentriert.max(axis=1) - zentriert.min(axis=1)
    beob = p.max() - p.min()
    pw = float((n_sp >= beob).mean())
    print(f"  {nm:<8} beobachtet {beob:5.2f} Punkte | reines Rauschen erzeugt "
          f"im Median {np.median(n_sp):.2f}, in 95 % der Faelle unter "
          f"{np.percentile(n_sp, 95):.2f}")
    print(f"  {'':<8} -> p = {pw:.3f} gegen die Annahme einheitlicher Wirkung")

print("\n=== Die zehn paarweisen Kontraste (gepaart ueber dieselben Komplexe) ===")
print(f"  {'Vergleich':<40}{'Regel':>19}{'Modell':>19}")
for i in range(5):
    for j in range(i + 1, 5):
        nm = (f"{LANG[ZELLEN[i][0]]},{ZELLEN[i][1]} gegen "
              f"{LANG[ZELLEN[j][0]]},{ZELLEN[j][1]}")
        aus = f"  {nm:<40}"
        for bs, p in ((bs_reg, p_reg), (bs_mod, p_mod)):
            d = bs[:, i] - bs[:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            st = "*" if lo > 0 or hi < 0 else " "
            aus += f"{p[i] - p[j]:+7.2f} [{lo:+5.2f},{hi:+5.2f}]{st}"
        print(aus)
print("  * = Intervall der Differenz schliesst die Null aus")

# ---- 3. Modell oder Ziehungszahl? --------------------------------------
print("\n=== Liegt der Unterschied am Modell oder an der Ziehungszahl? ===")
print("  Zwei Arme laufen mit beiden Schrittzahlen, die Achsen sind trennbar.\n")
i25 = [SP.index(f"{a}|25") for a in ("Minimal", "Separate")]
i5 = [SP.index(f"{a}|5") for a in ("Minimal", "Separate")]
imin = [SP.index(f"Minimal|{n_}") for n_ in (25, 5)]
isep = [SP.index(f"Separate|{n_}") for n_ in (25, 5)]
for nm, bs, p in (("Regel", bs_reg, p_reg), ("Modell", bs_mod, p_mod)):
    d1 = bs[:, i25].mean(axis=1) - bs[:, i5].mean(axis=1)
    d2 = bs[:, imin].mean(axis=1) - bs[:, isep].mean(axis=1)
    print(f"  {nm}")
    print(f"    25 Schritte minus 5 Schritte   "
          f"{p[i25].mean() - p[i5].mean():+6.2f}  "
          f"[{np.percentile(d1, 2.5):+5.2f},{np.percentile(d1, 97.5):+5.2f}]")
    print(f"    SF-Minimal minus SF-Separate   "
          f"{p[imin].mean() - p[isep].mean():+6.2f}  "
          f"[{np.percentile(d2, 2.5):+5.2f},{np.percentile(d2, 97.5):+5.2f}]")

# ---- Eingriffshaeufigkeit als Erklaerung -------------------------------
print("\n=== Wie oft kann die Regel je Zelle ueberhaupt eingreifen? ===")
print("  Anteil der Komplexe mit mehr als einem Cluster im Band, und die")
print("  mittlere Clusterzahl -- bei 200 Ziehungen entsteht eine andere")
print("  Clusterlandschaft als bei 40.\n")
kopf2 = "Zelle"
print(f"  {kopf2:<24}{'Ziehungen':>11}{'Cluster':>10}{'im Band>1':>12}"
      f"{'geaendert':>12}")
for i, (arm, nfe) in enumerate(ZELLEN):
    m = (t.arm == arm) & (t.nfe == nfe)
    sub = t[m]
    gb = sub.groupby("complex")
    n_band = float((im_band[m].groupby(sub["complex"]).sum() > 1).mean())
    r = regel[(regel.index.get_level_values(0) == arm)
              & (regel.index.get_level_values(1) == nfe)]
    b = basis.reindex(r.index)
    print(f"  {LANG[arm] + ', ' + str(nfe):<24}{int(sub.K.iloc[0]):11d}"
          f"{gb['a_n_cluster'].first().mean():10.1f}{100 * n_band:11.1f}%"
          f"{100 * float((r != b).mean()):11.1f}%")
