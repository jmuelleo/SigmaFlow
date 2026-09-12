"""Posenvarianz: wie verschieden sind die Ziehungen, ohne zu clustern?

DIE IDENTITAET, AUF DER ALLES BERUHT
    Fuer jede Punktwolke gilt exakt

        mittlerer quadratischer paarweiser Abstand  =  2 x Gesamtvarianz

    Die paarweisen RMSD, die ohnehin gerechnet werden, enthalten die Varianz
    also bereits -- man muss sie quadriert mitteln statt linear. Damit gibt es
    ein Streuungsmass in Angstroem, das WEDER eine Clusterschwelle NOCH eine
    Verkettungsregel braucht.

    Konkret, mit m als Mittel ueber die n(n-1)/2 Eintraege des oberen Dreiecks:

        V = m * (n-1) / (2n)        Streuung = sqrt(V)

    Der Faktor (n-1)/(2n) statt 1/2 korrigiert dafuer, dass das obere Dreieck
    die Diagonale (Abstand null) auslaesst. Bei n = 40 macht er 1,3 Prozent
    aus -- klein, aber es gibt keinen Grund, ihn wegzulassen.

DIE ZERLEGUNG, DIE DEN EIGENTLICHEN INHALT TRAEGT
    Eine einzelne Streuungszahl sagt nicht, WORIN die Posen sich unterscheiden.
    Drei Beitraege lassen sich trennen:

        LAGE          Streuung der Ligandenschwerpunkte. "Sitzt der Ligand
                      ueberhaupt an derselben Stelle?"
        ORIENTIERUNG  Was nach Abzug der Lage bleibt und durch eine Drehung
                      erklaert werden kann. "Gleiche Stelle, andere Richtung."
        FORM          Was auch nach optimaler Ueberlagerung bleibt: die innere
                      Konformation. "Gleiche Lage und Richtung, andere Torsion."

    Die erste Trennung ist EXAKT orthogonal. Schreibt man die Pose als
    Schwerpunkt plus zentrierte Pose, verschwindet der Kreuzterm, weil die
    zentrierte Pose per Konstruktion die Atomsumme null hat. Also gilt genau

        V_total = V_lage + V_rest

    Die zweite Trennung, V_rest in Orientierung und Form, ist es NICHT: die
    optimale Drehung ist nichtlinear, und "Orientierung" ist hier definiert
    als das, was die Ueberlagerung wegnimmt. Das ist eine brauchbare
    Aufteilung, keine Zerlegung im strengen Sinn -- und genau so gehoert es
    in den Text.

    Fuer SigmaFlow ist die Zerlegung besonders passend, weil das Modell je
    starrem Fragment eine Rotation und eine Translation erzeugt. Lage und
    Orientierung sind die Gesamtplatzierung, Form ist die relative Anordnung
    der Fragmente zueinander.

EFFEKTIVE DIMENSION
    Wieviele RICHTUNGEN traegt die Streuung? Die Hauptkomponenten der
    Posenwolke haben Eigenwerte lambda_k; das Partizipationsverhaeltnis

        (sum lambda)^2 / sum lambda^2

    ist 1, wenn alle Streuung auf einer Achse liegt, und gleich der Zahl der
    Achsen, wenn sie sich gleich verteilt. Das ist etwas anderes als die
    Modenzahl: eine Verteilung kann zwei weit getrennte Moden haben (2 Moden,
    aber effektiv 1 Dimension) oder eine einzige diffuse Wolke (1 Modus,
    viele Dimensionen).

WARUM VARIANZ NICHT DASSELBE IST WIE KONZENTRATION
    Die Varianz wird von den WEITESTEN Posen dominiert -- sie geht quadratisch
    ein. Der Anteil im groessten Cluster interessiert sich dagegen nur fuer
    den Kern. Eine Verteilung mit engem Kern und drei wilden Ausreissern hat
    hohe Varianz UND hohe Konzentration. Welche der beiden Groessen mehr ueber
    Richtigkeit sagt, ist deshalb eine empirische Frage; hier wird sie
    beantwortet, samt einer getrimmten Variante als Gegenprobe.

Aufruf:
    python SigmaFlow_Variants/varianz.py
    python SigmaFlow_Variants/varianz.py --nfe 5 --arme Minimal,Separate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
import posencache  # noqa: E402
from zellen import SAETZE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--nfe", type=int, default=25)
p.add_argument("--arme", default="SigmaDock,Minimal,Separate")
p.add_argument("--ziel", default="acc", choices=["acc", "beides"])
p.add_argument("--trimm", type=float, default=0.10,
               help="Anteil der entferntesten Posen, der fuer die getrimmte "
                    "Streuung wegfaellt")
p.add_argument("--out", default=None)
a = p.parse_args()
ARME = [s.strip() for s in a.arme.split(",")]
AUS = a.out or os.path.join(_HIER, f"varianz_{a.satz}_nfe{a.nfe}.csv")


def varianz_aus_d(d):
    """Gesamtvarianz je Atom aus der RMSD-Matrix. Einheit: Angstroem^2."""
    n = len(d)
    iu = np.triu_indices(n, 1)
    m = float((d[iu] ** 2).mean())
    return m * (n - 1) / (2 * n)


def kabsch(A, B):
    """Rotation, die das zentrierte A optimal auf das zentrierte B legt.

    Der Standardweg: Kreuzkovarianz, SVD, und eine Vorzeichenkorrektur, die
    eine SPIEGELUNG verhindert. Ohne sie kann det(R) = -1 herauskommen; das
    passte die Punkte zwar besser an, waere aber keine Drehung mehr, und ein
    gespiegeltes Molekuel ist ein anderes Molekuel.
    """
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def zerlegen(x, runden=3):
    """(V_total, V_lage, V_rest, V_form) je Atom, in Angstroem^2."""
    n, N, _ = x.shape
    c = x.mean(axis=1)                       # Schwerpunkte, [n, 3]
    V_lage = float(((c - c.mean(0)) ** 2).sum(-1).mean())
    z = x - c[:, None, :]                    # zentrierte Posen
    V_rest = float((((z - z.mean(0)) ** 2).sum(-1).sum(-1) / N).mean())

    # Generalisierte Prokrustes-Analyse: jede zentrierte Pose auf das
    # laufende Mittel drehen, Mittel neu bilden, wiederholen. Drei Runden
    # genuegen; das Verfahren konvergiert schnell und monoton.
    ref = z.mean(0)
    for _ in range(runden):
        gedreht = np.einsum("nij,jk->nik", z, np.eye(3))
        gedreht = np.stack([zi @ kabsch(zi, ref) for zi in z])
        ref = gedreht.mean(0)
    V_form = float((((gedreht - ref) ** 2).sum(-1).sum(-1) / N).mean())
    return V_lage + V_rest, V_lage, V_rest, V_form


def eff_dim(x):
    """Partizipationsverhaeltnis der Hauptkomponenten der Posenwolke."""
    n = len(x)
    y = x.reshape(n, -1)
    y = y - y.mean(0)
    # Eigenwerte der Gram-Matrix statt der Kovarianz: n ist viel kleiner als
    # die Zahl der Koordinaten, also ist die n x n Matrix billiger und hat
    # dieselben von null verschiedenen Eigenwerte.
    lam = np.linalg.eigvalsh(y @ y.T)
    lam = lam[lam > 1e-9]
    if not len(lam):
        return 1.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


zeilen = []
for arm in ARME:
    print(f"{arm}, {a.nfe} Schritte ...", flush=True)
    for code, v in posencache.hole(a.satz, arm, a.nfe, leise=True).items():
        if a.ziel not in v:
            continue
        d, x = v["d"].astype(float), v["xyz"].astype(float)
        heur, ziel = v["heur"], v[a.ziel]
        n = len(d)

        # Getrimmt: die entferntesten Posen weg. "Entfernt" heisst hier: die
        # groesste mittlere Distanz zu allen anderen. Das ist robuster als
        # der Abstand zur Mittelpose, weil die Mittelpose selbst von den
        # Ausreissern verschoben wird.
        k = max(3, int(round(n * (1 - a.trimm))))
        behalten = np.argsort(d.mean(axis=1))[:k]
        V_tot, V_lage, V_rest, V_form = zerlegen(x)
        zeilen.append({
            "arm": arm, "complex": code, "n": n,
            "trifft": bool(ziel[int(np.argmax(heur))]),
            "streuung": np.sqrt(V_tot),
            "streuung_trimm": np.sqrt(varianz_aus_d(d[np.ix_(behalten, behalten)])),
            "lage": np.sqrt(V_lage),
            "orientierung": np.sqrt(max(V_rest - V_form, 0.0)),
            "form": np.sqrt(V_form),
            "anteil_lage": V_lage / V_tot,
            "eff_dim": eff_dim(x),
        })

t = pd.DataFrame(zeilen)
t.to_csv(AUS, index=False)

print(f"\n########## {a.satz}, {a.nfe} Schritte, {int(t['n'].median())} Posen "
      f"je Komplex, {len(t)//len(ARME)} Komplexe je Arm ##########")
print("\n=== WIE VERSCHIEDEN SIND DIE POSEN? (RMS-Streuung in Angstroem) ===")
print(f"  {'Arm':<12}{'gesamt':>9}{'getrimmt':>10}{'Lage':>8}"
      f"{'Orient.':>9}{'Form':>7}{'Anteil Lage':>13}{'eff. Dim':>10}")
for arm in ARME:
    d = t[t.arm == arm]
    print(f"  {arm:<12}{d.streuung.mean():9.2f}{d.streuung_trimm.mean():10.2f}"
          f"{d.lage.mean():8.2f}{d.orientierung.mean():9.2f}{d['form'].mean():7.2f}"
          f"{100*d.anteil_lage.mean():12.0f}%{d.eff_dim.mean():10.1f}")

print("\n=== SAGT DIE VARIANZ ETWAS UEBER RICHTIGKEIT? (AUC, negativ genommen) ===")
print(f"  {'Groesse':<26}{'SigmaDock':>11}{'Minimal':>10}{'Separate':>10}{'Mittel':>9}")
for feld, lab in (("streuung", "-Gesamtstreuung"),
                  ("streuung_trimm", f"-getrimmt ({int(100*a.trimm)}% weg)"),
                  ("lage", "-Streuung der Lage"),
                  ("orientierung", "-Streuung Orientierung"),
                  ("form", "-Streuung der Form"),
                  ("eff_dim", "-effektive Dimension")):
    r = [auc(-t[t.arm == arm][feld], t[t.arm == arm]["trifft"]) for arm in ARME]
    print(f"  {lab:<26}" + "".join(f"{v:10.3f} " for v in r)
          + f"{np.nanmean(r):8.3f}")
print(f"\n{len(t)} Zeilen nach {AUS}")
