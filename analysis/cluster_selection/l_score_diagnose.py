"""Teil L -- ueberschaetzt der Score Falsches, oder unterschaetzt er Richtiges?

DIE FRAGE
    Der Siegerscore ist in Fehlfaellen niedriger (AUC 0,81). Das ist mit zwei
    voellig verschiedenen Ursachen vereinbar:

      A  Der Score UEBERSCHAETZT falsche Posen. Dann liegt in Fehlfaellen der
         beste falsche Score ungewoehnlich HOCH.
      B  Der Score UNTERSCHAETZT richtige Posen. Dann liegt in Fehlfaellen der
         beste richtige Score ungewoehnlich TIEF.

    Es gibt noch eine dritte Moeglichkeit, die man nicht uebersehen darf:

      C  Es wurde gar keine gute richtige Pose GEZOGEN. Der eine Treffer im
         Ziehsatz ist knapp unter 2 Angstroem und geometrisch mittelmaessig --
         dann ist sein niedriger Score kein Fehler des Scores, sondern eine
         zutreffende Bewertung, und das Problem liegt beim Sampling.

WARUM DER NAIVE VERGLEICH NICHT GEHT
    Bedingt man auf "Fehlgriff", gilt per Konstruktion
    bester_falscher > bester_richtiger. Beide Groessen verschieben sich also
    automatisch, und aus ihrer Differenz laesst sich A nicht von B trennen.
    Man braucht einen Anker, der nicht davon abhaengt, wer gewonnen hat.

DER ANKER: DERSELBE KOMPLEX IN EINER ANDEREN ZELLE
    Jeder Komplex laeuft in fuenf Zellen. Viele sind in manchen Zellen ein
    Treffer und in anderen ein rettbarer Fehlgriff. Ligand, Protein und
    Scoreskala sind dabei identisch -- nur der Ziehsatz wechselt. Der
    Vergleich derselben zwei Groessen zwischen dem Treffer- und dem
    Fehlzustand DESSELBEN Komplexes trennt A von B sauber:

      steigt der beste falsche Score im Fehlzustand   -> A
      faellt der beste richtige Score im Fehlzustand  -> B
      faellt zugleich die Zahl/Qualitaet der richtigen Posen -> C

Aufruf:
    python analysis/cluster_selection/l_score_diagnose.py --satz pb308
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_VAR = os.path.join(os.path.dirname(os.path.dirname(_HIER)), "SigmaFlow_Variants")
sys.path.insert(0, _VAR)
import posencache  # noqa: E402
from zellen import SAETZE, lade_zelle  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]

p = argparse.ArgumentParser()
p.add_argument("--satz", default="pb308")
p.add_argument("--ziel", default="beides", choices=["beides", "acc"])
a = p.parse_args()

rmsd_map = {}
for nfe in (25, 5):
    f = os.path.join(_VAR, f"rmsd_{a.satz}_nfe{nfe}.csv")
    if os.path.isfile(f):
        r = pd.read_csv(f)
        for arm in ("SigmaDock", "Minimal", "Separate"):
            s = r[r.arm == arm]
            if len(s):
                # 41 der 307 Komplexe haben in einem Seedverzeichnis eine
                # zweite SDF ohne Tabellenzeile. Ohne die Entdopplung liefert
                # .loc mehr Zeilen als Posen und die Maske passt nicht mehr.
                s = s.drop_duplicates(subset=["complex", "seed"], keep="first")
                rmsd_map[(arm, nfe)] = s.set_index(["complex", "seed"])["rmsd"]

zeilen = []
for arm, nfe in ZELLEN:
    z = next((z for z in SAETZE[a.satz]["zellen"]
              if z["arm"] == arm and z["nfe"] == nfe), None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    for code, v in posencache.hole(a.satz, arm, nfe, leise=True).items():
        idx = [(code, int(s)) for s in v["seed"]]
        heur = tab.loc[idx, "heur"].to_numpy(float)
        ok = np.asarray(v[a.ziel], bool)
        K = len(heur)
        if not ok.any() or ok.all():
            # nicht loesbar bzw. trivial -- traegt zur Frage nichts bei
            if not ok.any():
                zeilen.append({"arm": arm, "nfe": nfe, "complex": code,
                               "typ": "unloesbar", "K": K, "n_ok": 0})
            continue
        gn = -tab.loc[idx, "affinity"].to_numpy(float)
        ppb = tab.loc[idx, "p_pb"].to_numpy(float)
        i_ok = int(np.where(ok)[0][np.argmax(heur[ok])])
        i_no = int(np.where(~ok)[0][np.argmax(heur[~ok])])
        s_ok, s_no = float(heur[i_ok]), float(heur[i_no])
        rk = int(1 + (heur[~ok] > s_ok).sum())    # Rang der besten richtigen
        rm = rmsd_map.get((arm, nfe))
        r_ok = (float(rm.loc[idx].to_numpy(float)[ok].min())
                if rm is not None else np.nan)
        zeilen.append({
            "arm": arm, "nfe": nfe, "complex": code,
            "typ": "Treffer" if s_ok > s_no else "Fehlgriff",
            "K": K, "n_ok": int(ok.sum()), "anteil_ok": float(ok.mean()),
            "s_ok": s_ok, "s_no": s_no, "s_top": float(heur.max()),
            "s_med": float(np.median(heur)), "marge": s_ok - s_no,
            "rang_ok": rk, "rmsd_ok": r_ok,
            # Zerlegung des Mixed Scores in seine beiden Faktoren:
            # heur = (-affinity) * p_pb**4
            "gn_ok": float(gn[i_ok]), "gn_no": float(gn[i_no]),
            "pp_ok": float(ppb[i_ok]), "pp_no": float(ppb[i_no])})

T = pd.DataFrame(zeilen)
T.to_csv(os.path.join(_HIER, f"l_score_{a.satz}.csv"), index=False)
L = T[T.typ.isin(["Treffer", "Fehlgriff"])].copy()   # nur loesbare
F, H = L[L.typ == "Fehlgriff"], L[L.typ == "Treffer"]

print(f"########## {a.satz.upper()}, Ziel {a.ziel} ##########")
print(f"  loesbare Faelle {len(L)}  (Treffer {len(H)}, Fehlgriff {len(F)}), "
      f"unloesbar {int((T.typ == 'unloesbar').sum())}\n")

# ---- 1. Wie knapp verliert die richtige Pose? --------------------------
print("=== 1. Auf welchem Platz landet die beste RICHTIGE Pose im Fehlfall? ===")
for lo, hi, nm in ((2, 2, "Platz 2"), (3, 3, "Platz 3"), (4, 5, "Platz 4-5"),
                   (6, 10, "Platz 6-10"), (11, 25, "Platz 11-25"),
                   (26, 10**9, "Platz 26 oder schlechter")):
    m = (F.rang_ok >= lo) & (F.rang_ok <= hi)
    print(f"  {nm:<26}{int(m.sum()):5d}  {100 * m.mean():5.1f} %")
print(f"  Median Platz {F.rang_ok.median():.0f}, "
      f"oberes Quartil {F.rang_ok.quantile(.75):.0f}")
print("  -> Platz 2 haette der Score fast getroffen; Platz 26+ heisst, dass")
print("     dutzende falsche Posen besser bewertet wurden.")

# ---- 2. Der Anker: derselbe Komplex, andere Zelle -----------------------
print("\n=== 2. Derselbe Komplex, einmal Treffer und einmal Fehlgriff ===")
print("  Gleicher Ligand, gleiches Protein, gleiche Scoreskala. Nur so lassen")
print("  sich A und B trennen. Gepaart, Wilcoxon.\n")
paare = []
for code, g in L.groupby("complex"):
    h, f = g[g.typ == "Treffer"], g[g.typ == "Fehlgriff"]
    if len(h) and len(f):
        paare.append({
            "complex": code,
            "s_ok_H": h.s_ok.mean(), "s_ok_F": f.s_ok.mean(),
            "s_no_H": h.s_no.mean(), "s_no_F": f.s_no.mean(),
            "n_ok_H": h.anteil_ok.mean(), "n_ok_F": f.anteil_ok.mean(),
            "rmsd_H": h.rmsd_ok.mean(), "rmsd_F": f.rmsd_ok.mean(),
            "gn_ok_H": h.gn_ok.mean(), "gn_ok_F": f.gn_ok.mean(),
            "gn_no_H": h.gn_no.mean(), "gn_no_F": f.gn_no.mean(),
            "pp_ok_H": h.pp_ok.mean(), "pp_ok_F": f.pp_ok.mean(),
            "pp_no_H": h.pp_no.mean(), "pp_no_F": f.pp_no.mean()})
P = pd.DataFrame(paare)
print(f"  {len(P)} Komplexe erscheinen in beiden Zustaenden\n")
kopf = "Groesse"
print(f"  {kopf:<40}{'Treffer':>10}{'Fehlgriff':>11}{'Differenz':>12}{'p':>10}")
for nm, hh, ff, deut in (
        ("bester RICHTIGER Score", "s_ok_H", "s_ok_F",
         "faellt -> B: Richtiges unterschaetzt"),
        ("bester FALSCHER Score", "s_no_H", "s_no_F",
         "steigt -> A: Falsches ueberschaetzt"),
        ("Anteil richtiger Posen", "n_ok_H", "n_ok_F",
         "faellt -> C: schlechter gezogen"),
        ("bester RMSD im Ziehsatz", "rmsd_H", "rmsd_F",
         "steigt -> C: schlechter gezogen")):
    d = P[[hh, ff]].dropna()
    if len(d) < 5:
        continue
    diff = d[ff] - d[hh]
    try:
        pv = wilcoxon(diff).pvalue
    except ValueError:
        pv = np.nan
    print(f"  {nm:<40}{d[hh].mean():10.3f}{d[ff].mean():11.3f}"
          f"{diff.mean():+12.3f}{pv:10.5f}")
print()
for nm, deut in (("bester RICHTIGER Score", "faellt -> B"),
                 ("bester FALSCHER Score", "steigt -> A"),
                 ("Anteil richtiger Posen", "faellt -> C"),
                 ("bester RMSD im Ziehsatz", "steigt -> C")):
    print(f"    {nm:<32}{deut}")

# ---- 3. Wie gross ist der Effekt relativ? ------------------------------
print("\n=== 3. Welcher Anteil der Verschiebung geht auf welches Konto? ===")
d = P.dropna(subset=["s_ok_H", "s_ok_F", "s_no_H", "s_no_F"])
d_ok = (d.s_ok_F - d.s_ok_H).mean()
d_no = (d.s_no_F - d.s_no_H).mean()
gesamt = abs(d_ok) + abs(d_no)
print(f"  Verschiebung des besten richtigen Scores : {d_ok:+.3f}")
print(f"  Verschiebung des besten falschen Scores  : {d_no:+.3f}")
if gesamt > 0:
    print(f"  Anteil am Umschwung: Richtiges faellt {100*abs(d_ok)/gesamt:.0f} %, "
          f"Falsches steigt {100*abs(d_no)/gesamt:.0f} %")

# ---- 4. Ist die richtige Pose ueberhaupt gut? --------------------------
print("\n=== 4. Wie gut ist die beste richtige Pose in beiden Zustaenden? ===")
print("  Wenn sie im Fehlfall nur knapp unter 2 A liegt, ist ihr niedriger")
print("  Score kein Fehler des Scores, sondern eine zutreffende Bewertung.\n")
kopf2 = "Zustand"
print(f"  {kopf2:<20}{'Anteil richtig':>16}{'bester RMSD':>14}"
      f"{'RMSD unter 1 A':>17}{'n':>7}")
for nm, s in (("Treffer", H), ("Fehlgriff", F)):
    u1 = 100 * float((s.rmsd_ok < 1.0).mean())
    print(f"  {nm:<20}{100*s.anteil_ok.mean():15.1f}%{s.rmsd_ok.median():14.2f}"
          f"{u1:16.1f}%{len(s):7d}")

# ---- 5. Absolutes Scoreniveau -----------------------------------------
print("\n=== 5. Liegt im Fehlfall der ganze Komplex tiefer? ===")
kopf3 = "Zustand"
print(f"  {kopf3:<20}{'Siegerscore':>14}{'Median aller':>14}"
      f"{'bester richtig':>16}{'bester falsch':>15}")
for nm, s in (("Treffer", H), ("Fehlgriff", F)):
    print(f"  {nm:<20}{s.s_top.mean():14.3f}{s.s_med.mean():14.3f}"
          f"{s.s_ok.mean():16.3f}{s.s_no.mean():15.3f}")
print("\n  Faellt der Median mit, ist der ganze Komplex schwach bewertet;")
print("  bleibt er gleich, betrifft es gezielt die richtige Pose.")

# ---- 6. Den Sampling-Anteil herausrechnen ------------------------------
print("\n" + "=" * 92)
print("  6. Wie viel des Absturzes auf der richtigen Seite ist VERDIENT?")
print("=" * 92)
print("""
  Im Fehlzustand ist die beste richtige Pose nicht dieselbe Pose: sie liegt
  bei 1,02 statt 0,59 Angstroem. Ein Teil ihres niedrigeren Scores ist damit
  eine ZUTREFFENDE Bewertung einer schlechteren Pose -- kein Fehler des
  Scores, sondern eine Folge des Ziehens.

  Herausgerechnet wird das ueber die gepaarten Differenzen: die Aenderung des
  Scores wird auf die Aenderung des RMSD regressiert. Die Steigung sagt, wie
  viele Scorepunkte ein Angstroem kostet; der Achsenabschnitt ist der Rest,
  der bei UNVERAENDERTER Posenqualitaet bleibt -- und nur der ist Hypothese B.
""")
d = P.dropna(subset=["s_ok_H", "s_ok_F", "rmsd_H", "rmsd_F"])
dx = (d.rmsd_F - d.rmsd_H).to_numpy()
dy = (d.s_ok_F - d.s_ok_H).to_numpy()
A_ = np.vstack([dx, np.ones_like(dx)]).T
steig, abschn = np.linalg.lstsq(A_, dy, rcond=None)[0]
# Bootstrap ueber Komplexe auf den Achsenabschnitt
bs = []
r_ = np.random.default_rng(20260909)
for _ in range(4000):
    j = r_.integers(0, len(dx), len(dx))
    bs.append(np.linalg.lstsq(np.vstack([dx[j], np.ones_like(dx[j])]).T,
                              dy[j], rcond=None)[0][1])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  Steigung        {steig:+.3f} Scorepunkte je Angstroem RMSD")
print(f"  Achsenabschnitt {abschn:+.3f}  [{lo:+.3f},{hi:+.3f}]")
print(f"  Gesamtabsturz   {dy.mean():+.3f}")
erklaert = steig * dx.mean()
print(f"  davon durch schlechtere Posen erklaert: {erklaert:+.3f} "
      f"({100 * erklaert / dy.mean():.0f} %)")
print(f"  ungeklaerter Rest (Hypothese B)       : {abschn:+.3f} "
      f"({100 * abschn / dy.mean():.0f} %)")

# ---- 7. Welcher Faktor des Mixed Scores traegt den Umschwung? ---------
print("\n" + "=" * 92)
print("  7. gnina oder PoseBusters? Der Mixed Score ist (-affinity) * p_pb^4")
print("=" * 92)
print("""
  Beide Faktoren koennen den Umschwung erzeugen. Das ist keine akademische
  Unterscheidung: liegt es an gnina, ist die Affinitaetsvorhersage im
  falschen Bindungsmodus zu optimistisch; liegt es an p_pb, waehlt der
  Exponent 4 eine geometrisch saubere, aber falsch platzierte Pose.
""")
kopf4 = "Groesse"
print(f"  {kopf4:<34}{'Treffer':>10}{'Fehlgriff':>11}{'Differenz':>12}{'p':>10}")
for nm, hh, ff in (("gnina der besten RICHTIGEN", "gn_ok_H", "gn_ok_F"),
                   ("gnina der besten FALSCHEN", "gn_no_H", "gn_no_F"),
                   ("p_pb der besten RICHTIGEN", "pp_ok_H", "pp_ok_F"),
                   ("p_pb der besten FALSCHEN", "pp_no_H", "pp_no_F")):
    dd = P[[hh, ff]].dropna()
    diff = dd[ff] - dd[hh]
    try:
        pv = wilcoxon(diff).pvalue
    except ValueError:
        pv = np.nan
    print(f"  {nm:<34}{dd[hh].mean():10.3f}{dd[ff].mean():11.3f}"
          f"{diff.mean():+12.3f}{pv:10.5f}")
print("""
  Lesehilfe: p_pb ist der Anteil bestandener PB-Pruefungen und geht in der
  vierten Potenz ein -- eine Aenderung von 0,05 wirkt dort etwa so stark wie
  20 Prozent Aenderung im Score.""")
