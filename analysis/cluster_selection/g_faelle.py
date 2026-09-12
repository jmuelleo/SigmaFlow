"""Teil G -- Einzelfallanalyse der rettbaren Fehlgriffe.

Nicht "gibt es ein Merkmal", sondern "wie sehen diese Faelle ueberhaupt aus".
Fuenf Fragen, die eine aggregierte Zahl nicht beantwortet:

  1. IST DER FEHLER EINE EIGENSCHAFT DES KOMPLEXES?
     Wenn dieselben Komplexe in allen fuenf Zellen scheitern, liegt es nicht am
     Sampling, sondern an der Bewertungsfunktion oder am Komplex selbst. Dann
     ist eine Selektionsregel aus denselben Groessen aussichtslos, und die
     Diagnose zeigt gleich, wo man stattdessen ansetzen muesste.

  2. WIE KNAPP IST DIE ENTSCHEIDUNG?
     Verteilung des Score-Abstands zwischen gewaehltem und bestem korrekten
     Cluster. Ist der Abstand gross, ist die Bewertungsfunktion nicht knapp
     danebengelegen, sondern ueberzeugt falsch -- ein qualitativ anderer
     Fehler, den kein Umsortieren im Band einfangen kann.

  3. IST DER KORREKTE CLUSTER DER KLEINERE?
     Groessenverhaeltnis korrekter zu gewaehltem Cluster. Falls der korrekte
     systematisch kleiner ist, waere jede groessenbasierte Regel schaedlich --
     und das erklaert die frueheren Fehlschlaege.

  4. HAENGT DAS SCHEITERN AN DER LIGANDGROESSE?
     Schweratome und drehbare Bindungen aus der Kristallpose gegen die
     Fehlerrate. Grosse, flexible Liganden haben mehr plausible Moden.

  5. WIE VIEL WAERE MIT EINEM PERFEKTEN BAND ZU HOLEN?
     Anteil der Faelle, in denen der beste korrekte Cluster ueberhaupt im
     Band delta unter dem Spitzenwert liegt -- die Obergrenze der Bandfamilie.

Aufruf:
    python analysis/cluster_selection/g_faelle.py --satz pb308 --S 2.0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_VAR = os.path.join(os.path.dirname(os.path.dirname(_HIER)), "SigmaFlow_Variants")
sys.path.insert(0, _VAR)
from zellen import SAETZE  # noqa: E402

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--S", type=float, default=2.0)
a = p.parse_args()

t = pd.read_csv(os.path.join(_HIER, f"cluster_{a.satz}.csv"))
t = t[t.S == a.S].copy()
SCHL = ["arm", "nfe", "complex"]
print(f"########## {a.satz.upper()}, Schwelle {a.S} A ##########\n")

# ---- Fallliste ---------------------------------------------------------
zeilen = []
for (arm, nfe, code), g in t.groupby(SCHL):
    w = g[g.selected_by_ranker]
    if len(w) != 1:
        continue
    w = w.iloc[0]
    korr = g[g.best_pose_correct]
    typ = ("Treffer" if w.best_pose_correct else
           ("rettbar" if len(korr) else "unrettbar"))
    b = korr.sort_values("s_h_max", ascending=False).iloc[0] if len(korr) else None
    zeilen.append({
        "arm": arm, "nfe": nfe, "complex": code, "typ": typ,
        "n_cluster": int(g.a_n_cluster.iloc[0]), "K": int(g.K.iloc[0]),
        "gew_gr": int(w.g_gr), "gew_score": float(w.s_h_max),
        "gew_rmsd": float(w.best_rmsd),
        "korr_gr": int(b.g_gr) if b is not None else np.nan,
        "korr_score": float(b.s_h_max) if b is not None else np.nan,
        "korr_rmsd": float(b.best_rmsd) if b is not None else np.nan,
        "abstand": float(w.s_h_max - b.s_h_max) if b is not None else np.nan,
        "gr_verh": float(b.g_gr / w.g_gr) if b is not None else np.nan,
        "gew_valid": float(w.p_valid),
        "korr_valid": float(b.p_valid) if b is not None else np.nan})
F = pd.DataFrame(zeilen)
F.to_csv(os.path.join(_HIER, f"g_faelle_{a.satz}_{a.S}.csv"), index=False)
R = F[F.typ == "rettbar"]

print("=== Wie verteilen sich die Komplexe? ===")
for arm, nfe in ZELLEN:
    s = F[(F.arm == arm) & (F.nfe == nfe)]
    if not len(s):
        continue
    v = s.typ.value_counts()
    print(f"  {arm:<10} {nfe:>2} Schritte:  Treffer {v.get('Treffer', 0):>3}   "
          f"rettbar {v.get('rettbar', 0):>3}   "
          f"unrettbar {v.get('unrettbar', 0):>3}")

# ---- 1. Ist der Fehler eine Eigenschaft des Komplexes? -----------------
print("\n=== 1. Scheitern immer dieselben Komplexe? ===")
n_zellen = F.groupby(SCHL[:2]).ngroups
je = F.groupby("complex")["typ"].apply(
    lambda s: pd.Series({"treffer": (s == "Treffer").sum(),
                         "rettbar": (s == "rettbar").sum(),
                         "unrettbar": (s == "unrettbar").sum()})).unstack()
print(f"  Komplexe, die in ALLEN {n_zellen} Zellen treffen        : "
      f"{int((je.treffer == n_zellen).sum())}")
print(f"  Komplexe, die in ALLEN Zellen rettbar danebenliegen : "
      f"{int((je.rettbar == n_zellen).sum())}")
print(f"  Komplexe, die in KEINER Zelle treffen               : "
      f"{int((je.treffer == 0).sum())}")
print(f"  Komplexe, die mindestens einmal rettbar scheitern   : "
      f"{int((je.rettbar > 0).sum())} von {len(je)}")
q = (F.typ == "rettbar").mean()
erw = len(je) * q ** n_zellen
print(f"  ... davon in allen {n_zellen} Zellen: "
      f"{int((je.rettbar == n_zellen).sum())} beobachtet gegen {erw:.1f} "
      f"unter Unabhaengigkeit")
print("  -> deutlich mehr als erwartet heisst: der Fehler klebt am Komplex,")
print("     nicht an der Ziehung.")

# ---- 2. Wie knapp? ------------------------------------------------------
print("\n=== 2. Score-Abstand: gewaehlter minus bester korrekter Cluster ===")
qs = R.abstand.quantile([.1, .25, .5, .75, .9])
print("  Perzentile " + "  ".join(f"p{int(100 * k)} {v:.3f}"
                                  for k, v in qs.items()))
for d in (0.05, 0.1, 0.25, 0.5, 1.0):
    print(f"  Abstand unter {d:4.2f}: {100 * (R.abstand <= d).mean():5.1f} %  "
          f"({int((R.abstand <= d).sum())} Faelle)")
print("  -> das ist die Obergrenze der Bandfamilie: nur diese Faelle sind mit")
print("     einem Band der Breite delta ueberhaupt erreichbar.")

# ---- 3. Groessenverhaeltnis --------------------------------------------
print("\n=== 3. Ist der korrekte Cluster kleiner als der gewaehlte? ===")
print(f"  Median Groesse gewaehlt : {R.gew_gr.median():.0f} Posen")
print(f"  Median Groesse korrekt  : {R.korr_gr.median():.0f} Posen")
print(f"  korrekter Cluster kleiner in "
      f"{100 * (R.korr_gr < R.gew_gr).mean():.1f} % der Faelle")
print(f"  Median Verhaeltnis korrekt/gewaehlt: {R.gr_verh.median():.2f}")
print("  -> unter 1 heisst: eine Regel 'nimm den groesseren Cluster' wuerde")
print("     den Fehler in der Mehrheit der Faelle VERSTAERKEN.")
print(f"\n  PB-Validitaetsanteil gewaehlt {R.gew_valid.mean():.3f} gegen "
      f"korrekt {R.korr_valid.mean():.3f}")
print(f"  Zahl der Cluster: rettbare Faelle {R.n_cluster.mean():.2f}, "
      f"Treffer {F[F.typ == 'Treffer'].n_cluster.mean():.2f}, "
      f"unrettbar {F[F.typ == 'unrettbar'].n_cluster.mean():.2f}")

# ---- 4. Ligandgroesse ---------------------------------------------------
print("\n=== 4. Haengt das Scheitern an der Ligandgroesse? ===")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    ref = SAETZE[a.satz]["referenz"]
    props = {}
    for code in F["complex"].unique():
        f_ = os.path.join(ref, code, f"{code}_ligand.sdf")
        m = (Chem.MolFromMolFile(f_, sanitize=True, removeHs=True)
             if os.path.isfile(f_) else None)
        if m is None:
            continue
        props[code] = (m.GetNumHeavyAtoms(),
                       rdMolDescriptors.CalcNumRotatableBonds(m),
                       Descriptors.MolLogP(m))
    leer = (np.nan, np.nan, np.nan)
    F["atome"] = F["complex"].map(lambda c: props.get(c, leer)[0])
    F["drehbar"] = F["complex"].map(lambda c: props.get(c, leer)[1])
    F["logp"] = F["complex"].map(lambda c: props.get(c, leer)[2])
    kopf = "Gruppe"
    print(f"  {kopf:<14}{'Atome':>9}{'drehbar':>10}{'logP':>9}{'n':>7}")
    for typ in ("Treffer", "rettbar", "unrettbar"):
        s = F[F.typ == typ]
        print(f"  {typ:<14}{s.atome.median():9.0f}{s.drehbar.median():10.1f}"
              f"{s.logp.median():9.2f}{len(s):7d}")
    for sp in ("atome", "drehbar"):
        b = F[["typ", sp]].dropna()
        rk = b[b.typ != "unrettbar"]
        c = np.corrcoef((rk.typ == "rettbar").astype(float), rk[sp])[0, 1]
        print(f"  Korrelation {sp} mit 'rettbar statt Treffer': {c:+.3f}")
except ImportError:
    print("  (RDKit nicht verfuegbar)")

# ---- 5. Wiederkehrende Komplexe ----------------------------------------
print("\n=== 5. Die haeufigsten rettbaren Fehlgriffe ===")
top = R["complex"].value_counts().head(12)
kopf = "Komplex"
print(f"  {kopf:<12}{'Zellen':>8}{'D Abstand':>12}{'D Cluster':>11}"
      f"{'D RMSD gew.':>13}{'D RMSD korr.':>14}")
for code, n in top.items():
    s = R[R["complex"] == code]
    print(f"  {code:<12}{n:8d}{s.abstand.mean():12.3f}{s.n_cluster.mean():11.1f}"
          f"{s.gew_rmsd.mean():13.2f}{s.korr_rmsd.mean():14.2f}")
print(f"\nFallliste nach g_faelle_{a.satz}_{a.S}.csv")
