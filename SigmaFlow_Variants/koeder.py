"""Sind SigmaDocks FALSCHE Posen besser getarnt als die von SigmaFlow?

DIE FRAGE, DIE HIERHER FUEHRT
    `rankbarkeit.py` zeigt: der Ranking-Hebel ist bei Minimal 1,66 gegen
    SigmaDocks 1,38, WAEHREND die AUC leicht fuer SigmaDock spricht
    (0,750 gegen 0,721). AUC misst die Gesamtordnung ueber alle K Posen; der
    Hebel misst die Spitze. Beides zugleich geht nur, wenn sich die
    Verteilungen OBEN unterscheiden -- also darin, wie die BESTBEWERTETEN
    falschen Posen aussehen.

DIE HYPOTHESE
    Eine falsche Pose ist ein guter KOEDER, wenn sie chemisch und geometrisch
    einwandfrei ist: dann ist `p_pb` hoch, der Mixed Score `-affinity *
    p_pb^4` ebenfalls, und der Ranker kann sie nicht von einer richtigen
    unterscheiden. Ein Arm, dessen Fehler zugleich UNGUELTIG sind, wird sie
    durch den p^4-Faktor von selbst los.

    Vorhersage, falls die Hypothese stimmt: unter den FALSCHEN Posen ist der
    PB-valide Anteil bei SigmaDock hoeher als bei den Flow-Armen.

DREI MASSE

    Praezision@k    Anteil richtiger Posen unter den k bestbewerteten.
                    Das ist die Spitze der Liste, also genau das, was AUC
                    nicht aufloest. Nur Komplexe, die ueberhaupt eine richtige
                    Pose enthalten -- sonst waere @k trivial null und der
                    Vergleich verzerrt zugunsten des Arms mit besserem Orakel.

    Koederquote     Anteil PB-valider Posen UNTER DEN FALSCHEN. Hoch = gut
                    getarnte Fehler. Daneben dieselbe Quote unter den
                    richtigen, als Bezugsgroesse.

    Trenngroesse    (Mittelwert Score richtig - Mittelwert Score falsch)
                    geteilt durch die Standardabweichung INNERHALB des
                    Komplexes. Ein Effektmass, damit Arme mit verschieden
                    skalierten Scores vergleichbar sind. Je Komplex gerechnet,
                    dann Median -- nicht ueber alle Posen gepoolt, das waere
                    von Komplexunterschieden dominiert.

Aufruf:
    python SigmaFlow_Variants/koeder.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, alle_zellen  # noqa: E402

KS = (1, 3, 5, 10)

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--out", default=os.path.join(_HIER, "koeder.csv"))
a = p.parse_args()

print("Zellen:")
daten = alle_zellen(satz=a.satz)

zeilen = []
for (arm, nfe), m in daten.items():
    k_seeds = m["seed"].nunique()
    prec = {k: [] for k in KS}
    trenn = []
    n_falsch = n_falsch_valid = 0
    n_richtig = n_richtig_valid = 0
    p_pb_falsch, p_pb_richtig = [], []
    aff_falsch, aff_richtig = [], []

    for c, g in m.groupby("complex"):
        richtig = g["acc"].to_numpy(bool)
        # Koederquote ueber ALLE Komplexe -- auch die ohne richtige Pose,
        # denn gerade dort entstehen die Koeder.
        n_falsch += int((~richtig).sum())
        n_falsch_valid += int(g.loc[~richtig, "valid"].sum())
        n_richtig += int(richtig.sum())
        n_richtig_valid += int(g.loc[richtig, "valid"].sum())
        p_pb_falsch.append(g.loc[~richtig, "p_pb"].mean())
        p_pb_richtig.append(g.loc[richtig, "p_pb"].mean())
        aff_falsch.append(g.loc[~richtig, "affinity"].mean())
        aff_richtig.append(g.loc[richtig, "affinity"].mean())

        if not richtig.any():
            continue                     # @k und Trenngroesse undefiniert
        o = g.sort_values("heur", ascending=False)
        for k in KS:
            prec[k].append(float(o["acc"].to_numpy()[:k].mean()))
        if richtig.all():
            continue                     # Trenngroesse braucht beide Klassen
        s = g["heur"].to_numpy(float)
        sd = s.std(ddof=1)
        if sd > 0:
            trenn.append((s[richtig].mean() - s[~richtig].mean()) / sd)

    zeilen.append({
        "arm": arm, "nfe": nfe, "K": k_seeds,
        **{f"praez@{k}": round(100 * float(np.mean(prec[k])), 2) for k in KS},
        "koeder_quote": round(100 * n_falsch_valid / max(n_falsch, 1), 2),
        "valid_unter_richtigen": round(100 * n_richtig_valid / max(n_richtig, 1), 2),
        "p_pb_falsch": round(float(np.nanmean(p_pb_falsch)), 4),
        "p_pb_richtig": round(float(np.nanmean(p_pb_richtig)), 4),
        "aff_falsch": round(float(np.nanmean(aff_falsch)), 3),
        "aff_richtig": round(float(np.nanmean(aff_richtig)), 3),
        "trenngroesse": round(float(np.median(trenn)), 3) if trenn else np.nan,
        "n_trenn": len(trenn),
    })

t = pd.DataFrame(zeilen).sort_values(["nfe", "arm"], ascending=[False, True])
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")

print("\n=== SPITZE DER LISTE: Anteil richtiger Posen unter den k besten ===")
print("    (nur Komplexe, die ueberhaupt eine richtige Pose enthalten)")
print(f"  {'Zelle':<16}" + "".join(f"{'@'+str(k):>9}" for k in KS))
for _, r in t.iterrows():
    print(f"  {r['arm'] + ' ' + str(r['nfe']):<16}"
          + "".join(f"{r['praez@'+str(k)]:9.2f}" for k in KS))

print("\n=== WIE GUT SIND DIE FEHLER GETARNT? ===")
print(f"  {'Zelle':<16}{'PB-valid|falsch':>17}{'PB-valid|richtig':>18}"
      f"{'p_pb falsch':>13}{'p_pb richtig':>14}")
for _, r in t.iterrows():
    print(f"  {r['arm'] + ' ' + str(r['nfe']):<16}{r['koeder_quote']:16.2f}%"
          f"{r['valid_unter_richtigen']:17.2f}%"
          f"{r['p_pb_falsch']:13.4f}{r['p_pb_richtig']:14.4f}")

print("\n=== TRENNGROESSE des Mixed Score, je Komplex, Median ===")
print(f"  {'Zelle':<16}{'Trenngroesse':>14}{'n Komplexe':>12}")
for _, r in t.iterrows():
    print(f"  {r['arm'] + ' ' + str(r['nfe']):<16}{r['trenngroesse']:14.3f}"
          f"{r['n_trenn']:12d}")

print("""
LESEHILFE
  praez@k        Von den k bestbewerteten Posen: wie viele treffen? Das ist
                 die Spitze, die AUC nicht aufloest.
  PB-valid|falsch  Anteil PB-valider Posen UNTER DEN FALSCHEN. Hoch heisst:
                 gut getarnte Koeder, die der Ranker nicht aussortieren kann.
  Trenngroesse   Um wie viele Standardabweichungen liegt der Score einer
                 richtigen ueber dem einer falschen Pose, innerhalb desselben
                 Komplexes. Groesser = leichter zu ranken.""")
