"""Ist SigmaDock "confidently wrong" und SigmaFlow "unsicher, aber rettbar"?

DIE BEHAUPTUNG, DIE GEPRUEFT WIRD
    Ein Modell mit sehr konzentrierter Dichte trifft oft, liegt aber
    gelegentlich SICHER daneben. Ein Modell, das etwas breiter streut, legt
    auch ein paar Posen in unwahrscheinlichere Moden -- und gibt dem Ranker
    damit die Gelegenheit, sich fuer eine davon zu entscheiden.

    Daraus folgen zwei pruefbare Aussagen:
      (A) SigmaFlow streut breiter.
      (B) In den Faellen, wo SigmaDock falsch und SigmaFlow richtig liegt,
          ist SigmaDock KONZENTRIERT und auf dem falschen Modus.

ZWEI ZUGAENGE MIT SEHR UNTERSCHIEDLICHER FALLZAHL
    STRUKTURELL, aus den Posen: Moden, Konzentration, Entropie. Verfuegbar
    fuer die 67 lokal vorliegenden PB308-Komplexe. Die diskordanten Gruppen
    darin sind acht gegen acht -- genug zum Hinsehen, nicht zum Schliessen.

    UEBER DIE SCORES, aus den Tabellen: wie deutlich hebt sich die
    bestbewertete Pose vom Rest ab? Das braucht keine Posen und ist deshalb
    fuer ALLE 307 beziehungsweise 85 Komplexe verfuegbar. Es misst nicht
    dieselbe Groesse wie die Modenzahl -- ein Modell kann raeumlich breit
    streuen und trotzdem einen klaren Favoriten haben -- aber es misst
    genau das, was "confident" im Sinne der Auswahl bedeutet.

DAS KONFIDENZMASS
    z = (bester Score - Mittel) / Standardabweichung, INNERHALB des Komplexes.
    Es sagt, um wie viele Standardabweichungen der Favorit heraussticht.
    Innerhalb des Komplexes, weil die Scores zwischen Komplexen verschieden
    skaliert sind -- ein Vergleich ueber alle Posen hinweg waere von
    Komplexunterschieden dominiert und sagte nichts.

    "Confidently wrong" ist dann: hohes z UND Top-1 falsch.

Aufruf:
    python SigmaFlow_Variants/konfidenz.py
    python SigmaFlow_Variants/konfidenz.py --satz astex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import SAETZE, alle_zellen  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--ziel", default="acc", choices=["acc", "beides"])
p.add_argument("--struktur", default=os.path.join(_HIER, "struktur_67.csv"))
a = p.parse_args()

daten = alle_zellen(leise=True, satz=a.satz)
ARME = ["SigmaDock", "Minimal", "Separate"]

# --- je Komplex: trifft Top-1, und wie deutlich war die Wahl? -------------
je_arm = {}
for arm in ARME:
    m = daten[(arm, 25)]
    zeilen = []
    for c, g in m.groupby("complex"):
        s = g["heur"].to_numpy(float)
        sd = s.std(ddof=1)
        i = g["heur"].idxmax()
        zweitbester = np.sort(s)[-2] if len(s) > 1 else s[0]
        zeilen.append({
            "complex": c,
            "trifft": bool(g.loc[i, a.ziel]),
            "orakel": bool(g[a.ziel].any()),
            "je_zug": float(g[a.ziel].mean()),
            # Wie weit steht der Favorit ueber dem Mittel, in
            # Standardabweichungen des Komplexes?
            "z_bester": float((s.max() - s.mean()) / sd) if sd > 0 else np.nan,
            # Und wie weit ueber dem Zweitbesten? Das ist die eigentliche
            # Entscheidungsmarge -- z sagt etwas ueber die ganze Verteilung,
            # die Marge nur ueber die Spitze.
            "marge": float((s.max() - zweitbester) / sd) if sd > 0 else np.nan,
        })
    je_arm[arm] = pd.DataFrame(zeilen).set_index("complex")

sd_, mi_ = je_arm["SigmaDock"], je_arm["Minimal"]
idx = sd_.index.intersection(mi_.index)
gruppe = pd.Series("beide daneben", index=idx)
gruppe[sd_.trifft[idx] & mi_.trifft[idx]] = "beide treffen"
gruppe[~sd_.trifft[idx] & mi_.trifft[idx]] = "nur Minimal"
gruppe[sd_.trifft[idx] & ~mi_.trifft[idx]] = "nur SigmaDock"

print(f"########## {a.satz.upper()}, Ziel {a.ziel}, K = 40 ##########")
print(f"{len(idx)} Komplexe\n")
print(gruppe.value_counts().to_string())

# --- (B) Konfidenz: ist der Verlierer der jeweiligen Gruppe sicher? -------
print(f"\n=== KONFIDENZ: wie deutlich stand der Favorit heraus? ===")
print("    z = (bester Score - Mittel) / Standardabweichung, je Komplex")
print(f"\n  {'Gruppe':<16}{'n':>4}   {'z SigmaDock':>13}{'z Minimal':>11}   "
      f"{'Marge SD':>10}{'Marge MIN':>11}")
for g in ("beide treffen", "nur Minimal", "nur SigmaDock", "beide daneben"):
    k = idx[gruppe == g]
    if not len(k):
        continue
    print(f"  {g:<16}{len(k):4d}   {sd_.z_bester[k].mean():13.3f}"
          f"{mi_.z_bester[k].mean():11.3f}   {sd_.marge[k].mean():10.3f}"
          f"{mi_.marge[k].mean():11.3f}")

print(f"\n=== CONFIDENTLY WRONG: z getrennt nach richtig und falsch ===")
print(f"  {'Arm':<12}{'z | trifft':>12}{'z | daneben':>13}{'Differenz':>11}"
      f"{'p (gepaart)':>13}")
rng = np.random.default_rng(20260908)
for arm in ARME:
    d = je_arm[arm]
    r, f = d.z_bester[d.trifft], d.z_bester[~d.trifft]
    # Ungepaart, zwei Gruppen von Komplexen -- Bootstrap ueber die Differenz
    # der Mittelwerte.
    if len(r) > 2 and len(f) > 2:
        b = np.array([r.sample(len(r), replace=True, random_state=int(s)).mean()
                      - f.sample(len(f), replace=True, random_state=int(s)).mean()
                      for s in rng.integers(0, 2**31 - 1, 2000)])
        pv = min(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0)
    else:
        pv = np.nan
    print(f"  {arm:<12}{r.mean():12.3f}{f.mean():13.3f}"
          f"{r.mean() - f.mean():+11.3f}{pv:13.4f}")
print("\n  Ein Modell, das bei Fehlgriffen GENAUSO sicher ist wie bei Treffern,")
print("  hat ein Konfidenzsignal ohne Wert. Je groesser die Differenz, desto")
print("  besser trennt die Sicherheit des Modells richtig von falsch.")

# --- (A) Struktur, nur wo Posen vorliegen --------------------------------
if not os.path.isfile(a.struktur):
    print(f"\n[Struktur uebersprungen: {a.struktur} fehlt]")
    sys.exit(0)
st = pd.read_csv(a.struktur)
st = st[st["nfe"] == 25]
w = st.pivot_table(index="complex", columns="arm",
                   values=["n_moden", "groesster_modus", "entropie",
                           "groesster_richtig", "abdeckung_richtig"])
w.columns = [f"{b}_{c}" for b, c in w.columns]
gemeinsam = w.index.intersection(idx)
print(f"\n=== STRUKTUR, {len(gemeinsam)} Komplexe mit Posen ===")
print(f"  {'Gruppe':<16}{'n':>4}   {'Moden SD':>9}{'Moden MIN':>10}   "
      f"{'groesst SD':>11}{'groesst MIN':>12}   "
      f"{'gr.richtig SD':>14}{'gr.richtig MIN':>15}")
for g in ("beide treffen", "nur Minimal", "nur SigmaDock", "beide daneben"):
    k = gemeinsam[gruppe[gemeinsam] == g]
    if not len(k):
        continue
    d = w.loc[k]
    print(f"  {g:<16}{len(k):4d}   {d.n_moden_SigmaDock.mean():9.1f}"
          f"{d.n_moden_Minimal.mean():10.1f}   "
          f"{100 * d.groesster_modus_SigmaDock.mean():10.1f}%"
          f"{100 * d.groesster_modus_Minimal.mean():11.1f}%   "
          f"{100 * d.groesster_richtig_SigmaDock.mean():13.1f}%"
          f"{100 * d.groesster_richtig_Minimal.mean():14.1f}%")

# --- (C) Konfidenz als DICHTE statt als Score-Marge ------------------------
#
# Der z-Test oben misst, wie deutlich der RANKER sich entschieden hat. Das ist
# eine Eigenschaft des Rankers, nicht des generativen Modells -- und es stellt
# sich heraus, dass es nichts traegt.
#
# Die Konfidenz des MODELLS ist eine andere Groesse: wie stark buendelt es
# seine Ziehungen? Ein Modell, das vierzig von vierzig Posen in denselben
# Cluster legt, hat sich festgelegt; eines, das sie auf zwoelf Moden verteilt,
# nicht. Diese Groesse liegt in struktur_<n>.csv als `groesster_modus` vor und
# braucht keine zweite Rechnung.
#
# Sie kostet allerdings die Fallzahl: sie braucht die Posen, existiert also nur
# fuer die lokal vorliegenden Komplexe.


def _auc(x, y):
    """Wahrscheinlichkeit, dass ein Treffer ein hoeheres x hat als ein Fehlgriff.

    Das ist die Flaeche unter der ROC-Kurve, hier ueber die Rangsumme gerechnet
    (Mann-Whitney-U geteilt durch n1*n0). 0.5 heisst: das Mass sagt nichts.
    1.0 hiesse: es trennt perfekt. Gegenueber einer blossen Mittelwert-
    differenz hat es den Vorteil, skalenfrei zu sein -- z und Clusteranteil
    sind damit direkt vergleichbar, obwohl das eine in Standardabweichungen
    und das andere in Prozent gemessen wird.
    """
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print(f"\n=== KONFIDENZ ALS DICHTE: Anteil der Ziehungen im groessten Cluster ===")
print(f"  {'Arm':<12}{'n':>4}{'konz | trifft':>15}{'konz | daneben':>16}"
      f"{'Diff':>8}{'p':>9}{'AUC':>7}")
for arm in ARME:
    d = w.loc[gemeinsam, f"groesster_modus_{arm}"]
    t = je_arm[arm].trifft.reindex(gemeinsam)
    x, y = d.to_numpy(float), t.to_numpy(bool)
    r, f = x[y], x[~y]
    b = np.array([rng.choice(r, len(r)).mean() - rng.choice(f, len(f)).mean()
                  for _ in range(4000)])
    pv = min(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0)
    print(f"  {arm:<12}{len(x):4d}{100 * r.mean():14.1f}%{100 * f.mean():15.1f}%"
          f"{100 * (r.mean() - f.mean()):+7.1f}{pv:9.4f}{_auc(x, y):7.3f}")

print("\n  Zum Vergleich dieselbe AUC fuer die Score-Marge z, auf allen Komplexen:")
for arm in ARME:
    d = je_arm[arm].dropna(subset=["z_bester"])
    print(f"    {arm:<12}{len(d):4d}   AUC {_auc(d.z_bester, d.trifft):.3f}")

# "Confidently wrong" heisst mit diesem Mass: das Modell hat sich festgelegt
# UND liegt daneben. Die Schwelle 0.5 ist willkuerlich, aber lesbar -- mehr
# als die Haelfte aller Ziehungen im selben Cluster.
print(f"\n=== CONFIDENTLY WRONG, mit der Dichte gemessen ===")
print("    festgelegt := mehr als die Haelfte der Ziehungen im groessten Cluster")
print(f"\n  {'Arm':<12}{'festgelegt':>11}{'davon falsch':>21}"
      f"{'zerstreut':>11}{'davon falsch':>21}")
for arm in ARME:
    x = w.loc[gemeinsam, f"groesster_modus_{arm}"].to_numpy(float)
    t = je_arm[arm].trifft.reindex(gemeinsam).to_numpy(bool)
    k = x > 0.5
    print(f"  {arm:<12}{k.sum():11d}{(~t & k).sum():14d}"
          f" ({100 * (~t[k]).mean():5.1f}%){(~k).sum():11d}"
          f"{(~t & ~k).sum():14d} ({100 * (~t[~k]).mean():5.1f}%)")
print("\n  Waere ein Arm 'confidently wrong', stuende in seiner Spalte 'festgelegt")
print("  / davon falsch' ein hoher Anteil. Steht dort ein niedriger, ist die")
print("  Festlegung des Modells ein brauchbares Konfidenzsignal.")
