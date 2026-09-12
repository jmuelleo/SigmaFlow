"""Teil M -- wie kann ein hoher Score Richtigkeit anzeigen, wenn falsche Posen
hohe Scores bekommen?

DER SCHEINBARE WIDERSPRUCH
    Teil L: der groesste Fehlerposten ist, dass falsche Posen zu hoch bewertet
    werden. Teil K: der Siegerscore sagt mit AUC 0,81 vorher, ob die
    ausgelieferte Pose korrekt ist. Beides zugleich klingt unmoeglich.

DIE AUFLOESUNG, DIE GEPRUEFT WERDEN MUSS
    Es sind zwei verschiedene Vergleiche.
      INNERHALB eines Komplexes entscheidet der Score zwischen Posen. Dort
      versagt er: die falsche schlaegt die richtige.
      ZWISCHEN Komplexen wird nur das NIVEAU des Siegerscores verglichen.
    Die Vermutung: falsche Posen werden zwar hoch bewertet, aber nicht so
    hoch, wie eine wirklich richtige Pose bewertet wuerde. Die Ueberbewertung
    reicht, um die Reihenfolge zu kippen, nicht aber, um das Niveau eines
    echten Treffers zu erreichen.

    Pruefbar an den 95 Komplexen, die in beiden Zustaenden vorkommen: liegt
    der Siegerscore im Fehlzustand unter dem im Trefferzustand DESSELBEN
    Komplexes?

DIE ZWEITE FRAGE: WORAUS BESTEHT DIE AUC EIGENTLICH?
    Ein Score-Niveau haengt stark am Liganden -- grosse Liganden erreichen
    hoehere Affinitaeten, und grosse Liganden sind schwerer. Ein Teil der 0,81
    koennte also nur "wie gross ist der Ligand" sein, was man vorher weiss und
    wofuer man den Score nicht braucht. Deshalb:
      - AUC der Ligandgroesse allein,
      - AUC des Scores nach Herausrechnen der Ligandgroesse,
      - Zerlegung in einen Anteil ZWISCHEN Komplexen und einen INNERHALB.

DIE DRITTE FRAGE: TAUGT ES PRAKTISCH?
    Eine AUC ist kein Arbeitsmittel. Gerechnet wird deshalb, was passiert,
    wenn man die schwaechsten x Prozent der Laeufe als unsicher markiert:
    wie viele Fehlgriffe faengt man, wie viele Treffer wirft man weg.

Aufruf:
    python analysis/cluster_selection/m_warum_funktioniert.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_HIER = os.path.dirname(os.path.abspath(__file__))
_VAR = os.path.join(os.path.dirname(os.path.dirname(_HIER)), "SigmaFlow_Variants")
sys.path.insert(0, _VAR)
from zellen import SAETZE  # noqa: E402

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
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


T = pd.read_csv(os.path.join(_HIER, "l_score_pb308.csv"))
L = T[T.typ.isin(["Treffer", "Fehlgriff"])].copy()
# Der Siegerscore ist der Score der ausgelieferten Pose.
L["sieger"] = np.where(L.typ == "Treffer", L.s_ok, L.s_no)
L["treffer"] = L.typ == "Treffer"
# Auch die unloesbaren Komplexe gehoeren in die Konfidenzfrage: dort ist die
# ausgelieferte Pose immer falsch. Ihr Siegerscore fehlt in l_score, wird
# deshalb hier nicht mitgefuehrt -- ausdruecklich vermerkt.

print("#" * 92)
print("  Wie kann ein hoher Score Richtigkeit anzeigen, wenn falsche Posen")
print("  hohe Scores bekommen?")
print("#" * 92)

# ---- 1. Die Aufloesung -------------------------------------------------
print("\n=== 1. Der Siegerscore, aufgeschluesselt ===")
print("  Was der Score im Fehlfall auszeichnet, ist nicht, dass er niedrig")
print("  waere -- sondern dass er niedriger ist, als ein echter Treffer waere.\n")
H, F = L[L.treffer], L[~L.treffer]
kopf = "Groesse"
print(f"  {kopf:<40}{'Wert':>10}")
print(f"  {'Siegerscore im Trefferfall':<40}{H.sieger.mean():10.3f}")
print(f"  {'Siegerscore im Fehlfall':<40}{F.sieger.mean():10.3f}")
print(f"  {'bester RICHTIGER Score im Fehlfall':<40}{F.s_ok.mean():10.3f}")
print("\n  Der falsche Sieger (%.2f) liegt UEBER der richtigen Pose desselben"
      % F.sieger.mean())
print(f"  Ziehsatzes ({F.s_ok.mean():.2f}) -- deshalb gewinnt er. Er liegt aber")
print(f"  UNTER dem Niveau eines echten Treffers ({H.sieger.mean():.2f}).")

paare = []
for code, g in L.groupby("complex"):
    h, f = g[g.treffer], g[~g.treffer]
    if len(h) and len(f):
        paare.append({"complex": code, "H": h.sieger.mean(),
                      "F": f.sieger.mean()})
P = pd.DataFrame(paare)
d = P.F - P.H
print(f"\n  Gepaart, {len(P)} Komplexe in beiden Zustaenden:")
print(f"    Siegerscore Treffer {P.H.mean():.3f}, Fehlgriff {P.F.mean():.3f}, "
      f"Differenz {d.mean():+.3f}, p = {wilcoxon(d).pvalue:.5f}")
print("    -> derselbe Komplex, dieselbe Skala. Das Niveau faellt tatsaechlich.")

# ---- 2. Woraus besteht die AUC? ---------------------------------------
print("\n=== 2. Zerlegung: zwischen Komplexen oder innerhalb? ===")
gesamt = auc(L.sieger, L.treffer)
# innerhalb: nur Komplexe mit gemischtem Ausgang, Score komplexzentriert
L["_zentriert"] = L.sieger - L.groupby("complex")["sieger"].transform("mean")
gem = L.groupby("complex")["treffer"].transform("nunique") > 1
innen = auc(L.loc[gem, "_zentriert"], L.loc[gem, "treffer"])
# zwischen: je Komplex gemittelt
km = L.groupby("complex").agg(s=("sieger", "mean"), q=("treffer", "mean"))
zwischen = auc(km.s, km.q > 0.5)
print(f"  gepoolt, wie in Teil K berichtet          {gesamt:.3f}")
print(f"  ZWISCHEN Komplexen (je Komplex gemittelt) {zwischen:.3f}")
print(f"  INNERHALB eines Komplexes ({int(gem.sum())} Zeilen, "
      f"{int(L.loc[gem, 'complex'].nunique())} Komplexe) {innen:.3f}")
print("\n  Beide Anteile liegen ueber 0,5. Das Score-Niveau sagt also nicht nur,")
print("  welcher KOMPLEX schwer ist, sondern auch, welcher LAUF schiefging.")

# ---- 3. Ist es nur die Ligandgroesse? ---------------------------------
print("\n=== 3. Ist das Score-Niveau nur eine Umschreibung der Ligandgroesse? ===")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    ref = SAETZE["pb308"]["referenz"]
    pr = {}
    for code in L["complex"].unique():
        f_ = os.path.join(ref, code, f"{code}_ligand.sdf")
        m = (Chem.MolFromMolFile(f_, sanitize=True, removeHs=True)
             if os.path.isfile(f_) else None)
        if m is not None:
            pr[code] = (m.GetNumHeavyAtoms(),
                        rdMolDescriptors.CalcNumRotatableBonds(m))
    L["atome"] = L["complex"].map(lambda c: pr.get(c, (np.nan, np.nan))[0])
    L["drehbar"] = L["complex"].map(lambda c: pr.get(c, (np.nan, np.nan))[1])
    D = L.dropna(subset=["atome"]).copy()
    # Ligandeffizienz: Score je Schweratom -- die uebliche Groessenkorrektur
    D["effizienz"] = D.sieger / D.atome
    # Residuum nach linearer Anpassung an die Atomzahl
    A_ = np.vstack([D.atome.to_numpy(float), np.ones(len(D))]).T
    b = np.linalg.lstsq(A_, D.sieger.to_numpy(float), rcond=None)[0]
    D["residuum"] = D.sieger - A_ @ b
    print(f"  {'Groesse':<34}{'AUC':>8}")
    for nm, sp in (("Siegerscore roh", "sieger"),
                   ("Zahl der Schweratome", "atome"),
                   ("drehbare Bindungen", "drehbar"),
                   ("Score je Schweratom", "effizienz"),
                   ("Score nach Abzug der Groesse", "residuum")):
        print(f"  {nm:<34}{auc(D[sp], D.treffer):8.3f}")
    print("\n  Bleibt das Residuum deutlich ueber 0,5, traegt der Score eigene")
    print("  Information und ist nicht nur ein Groessenmass.")
except ImportError:
    print("  (RDKit nicht verfuegbar)")
    D = L.copy()

# ---- 4. Praktische Brauchbarkeit --------------------------------------
print("\n=== 4. Was bringt es, die schwaechsten Laeufe zu markieren? ===")
print("  Markiert wird komplexweise nach dem Siegerscore, je Zelle getrennt")
print("  (die Zellen haben verschiedene Trefferquoten).\n")
kopf2 = "markiert"
print(f"  {kopf2:>9}{'Fehlgriffe erwischt':>21}{'Treffer geopfert':>19}"
      f"{'Treffer im Rest':>18}{'Anreicherung':>14}")
for q in (0.05, 0.10, 0.20, 0.30, 0.50):
    fang, opfer, rest = [], [], []
    for arm, nfe in ZELLEN:
        s = L[(L.arm == arm) & (L.nfe == nfe)]
        if not len(s):
            continue
        gr = s.sieger.quantile(q)
        mark = s.sieger <= gr
        fang.append((mark & ~s.treffer).sum() / max((~s.treffer).sum(), 1))
        opfer.append((mark & s.treffer).sum() / max(s.treffer.sum(), 1))
        rest.append(s.loc[~mark, "treffer"].mean())
    basis = L.treffer.mean()
    print(f"  {100*q:8.0f}%{100*np.mean(fang):20.1f}%{100*np.mean(opfer):18.1f}%"
          f"{100*np.mean(rest):17.1f}%{100*(np.mean(rest)-basis):+13.1f}")
print(f"\n  Basisrate ohne Markierung: {100*L.treffer.mean():.1f} % Treffer.")
print("  'Anreicherung' ist der Gewinn an Trefferquote unter den nicht")
print("  markierten Laeufen -- das, was ein Anwender davon haette.")

# ---- 5. Dasselbe fuer ALLE Komplexe, auch die unloesbaren -------------
print("\n" + "=" * 92)
print("  5. Dieselbe Rechnung fuer ALLE 307 Komplexe")
print("=" * 92)
print("""
  Die Abschnitte 1 bis 4 betrachten nur LOESBARE Komplexe -- solche, in denen
  ueberhaupt eine korrekte Pose gezogen wurde. Dort beantwortet der Score die
  schwere Frage: "eine richtige Pose lag vor, habe ich sie genommen?"

  Zur Inferenzzeit weiss man das aber nicht. Da lautet die Frage: "ist die
  Pose, die ich gerade ausliefere, korrekt?" -- und in 300 der 1535 Faelle
  lautet die Antwort schon deshalb nein, weil nichts Richtiges gezogen wurde.
  Diese Faelle haben niedrige Scores und sind leicht zu erkennen. Deshalb ist
  die AUC auf der vollen Population hoeher; sie loest eine leichtere Aufgabe.

  Beide Zahlen sind richtig. Welche gilt, haengt an der Frage.
""")
C = pd.read_csv(os.path.join(_HIER, "cluster_pb308.csv"))
C = C[C.S == 2.0]
SCHL = ["arm", "nfe", "complex"]
V = pd.DataFrame({
    "sieger": C.groupby(SCHL)["s_h_max"].max(),
    "loesbar": C.groupby(SCHL)["best_pose_correct"].max().astype(bool),
})
V["treffer"] = C[C.selected_by_ranker].set_index(SCHL)["best_pose_correct"] \
    .reindex(V.index).astype(bool)
V = V.reset_index()
print(f"  {len(V)} Faelle, davon {int((~V.loesbar).sum())} unloesbar\n")
print(f"  {'Frage':<52}{'AUC':>8}")
print(f"  {'ist die ausgelieferte Pose korrekt? (alle)':<52}"
      f"{auc(V.sieger, V.treffer):8.3f}")
print(f"  {'wurde ueberhaupt etwas Richtiges gezogen?':<52}"
      f"{auc(V.sieger, V.loesbar):8.3f}")
print(f"  {'richtig gewaehlt, GEGEBEN es lag was vor?':<52}"
      f"{auc(V[V.loesbar].sieger, V[V.loesbar].treffer):8.3f}")
print("\n  Die erste Zahl ist eine Mischung aus der zweiten (leicht) und der")
print("  dritten (schwer). Sie ist die operativ richtige, aber der Score")
print("  verdankt ihr Niveau zum grossen Teil der leichten Teilaufgabe.\n")
kopf3 = "markiert"
print(f"  {kopf3:>9}{'Fehlgriffe erwischt':>21}{'Treffer geopfert':>19}"
      f"{'Treffer im Rest':>18}{'Anreicherung':>14}")
for q in (0.05, 0.10, 0.20, 0.30, 0.50):
    fang, opfer, rest = [], [], []
    for arm, nfe in ZELLEN:
        s = V[(V.arm == arm) & (V.nfe == nfe)]
        if not len(s):
            continue
        mark = s.sieger <= s.sieger.quantile(q)
        fang.append((mark & ~s.treffer).sum() / max((~s.treffer).sum(), 1))
        opfer.append((mark & s.treffer).sum() / max(s.treffer.sum(), 1))
        rest.append(s.loc[~mark, "treffer"].mean())
    print(f"  {100*q:8.0f}%{100*np.mean(fang):20.1f}%{100*np.mean(opfer):18.1f}%"
          f"{100*np.mean(rest):17.1f}%"
          f"{100*(np.mean(rest) - V.treffer.mean()):+13.1f}")
print(f"\n  Basisrate ohne Markierung: {100*V.treffer.mean():.1f} % Treffer.")
