"""Paarweise Tests zwischen den Armen, mit Holm-Bonferroni.

DAS VERFAHREN
    Je Komplex liefert jeder Arm eine Zahl in [0, 1]: die Wahrscheinlichkeit,
    dass die vom Ranker gewaehlte Pose das Kriterium erfuellt, gemittelt ueber
    zufaellige K-Teilmengen der gezogenen Posen. Beide Arme sehen dieselben
    Komplexe, die Beobachtungen sind also GEPAART.

    Getestet wird die mittlere Differenz dieser Paare. Die Verteilung dieser
    Statistik kommt aus einem Bootstrap ueber KOMPLEXE: 8000-mal werden
    Komplexe mit Zuruecklegen gezogen, jedes Mal die mittlere Differenz
    gebildet. Daraus das 95-Prozent-Intervall und ein zweiseitiger p-Wert als
    Anteil der Bootstrap-Mittel jenseits der Null.

WARUM NICHT WILCOXON
    Wilcoxon rangiert die Differenzen. Bei Erwartungswerten sind sehr viele
    Paare fast gleich, aber nicht exakt gleich; diese Beinahe-Gleichstaende
    bekommen echte Raenge und verwaessern die Statistik. Am Endpunkt gab
    Wilcoxon p = 0,35, wo der Bootstrap 1,3e-4 liefert. Es gilt der Bootstrap.

WAS DER TEST NICHT ABDECKT
    Gestreut wird ueber Komplexe und ueber die Ziehung der Posen. NICHT ueber
    Trainingslaeufe: es gibt einen Lauf je Arm. Ein Unterschied kann daher von
    der Initialisierung stammen und nicht vom Mechanismus.
"""
import itertools

import numpy as np

import final_tables as ft

MATCHED = {"PB308": 140, "AX85": 101}
KRIT = [("both", "RMSD<2 und PB-valid"), ("acc", "RMSD<2")]
B = 8000


def reihe(data, bench, arm, nfe, K, target):
    return ft.rule(data[(bench, arm, nfe)], target, K, "heuristic")


def test(a, b, rng):
    """b minus a, gepaart ueber die gemeinsamen Komplexe."""
    idx = a.index.intersection(b.index)
    d = (b.reindex(idx) - a.reindex(idx)).to_numpy()
    zieh = rng.integers(0, len(d), size=(B, len(d)))
    mittel = d[zieh].mean(axis=1)
    p = min(1.0, max(2 * min((mittel <= 0).mean(), (mittel >= 0).mean()),
                  1.0 / B))
    return (100 * d.mean(), 100 * np.percentile(mittel, 2.5),
            100 * np.percentile(mittel, 97.5), p, len(idx))


def holm(pwerte):
    """Holm-Bonferroni: korrigierte Schranken bei familienweisem Niveau 5 %."""
    n = len(pwerte)
    ordnung = sorted(range(n), key=lambda i: pwerte[i])
    haltbar, grenze_erreicht = [False] * n, False
    for rang, i in enumerate(ordnung):
        grenze = 0.05 / (n - rang)
        if not grenze_erreicht and pwerte[i] <= grenze:
            haltbar[i] = True
        else:
            grenze_erreicht = True
    return haltbar


def main():
    data = ft.build()
    rng = np.random.default_rng(11)

    for bench in ("PB308", "AX85"):
        mk = MATCHED[bench]
        # (Beschreibung, Arm A mit Konfiguration, Arm B mit Konfiguration)
        paare = []
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 25/40 gegen SigmaDock 25/40",
                          ("SigmaDock", 25, 40), (arm, 25, 40)))
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 5/{mk} gegen SigmaDock 25/40  [gematcht]",
                          ("SigmaDock", 25, 40), (arm, 5, mk)))
        for arm in ("SigmaFlow-Minimal", "SigmaFlow-Separate"):
            paare.append((f"{arm} 5/200 gegen SigmaDock 25/40",
                          ("SigmaDock", 25, 40), (arm, 5, 200)))
        for nfe, K in ((25, 40), (5, mk), (5, 200)):
            paare.append((f"Separate gegen Minimal, {nfe}/{K}",
                          ("SigmaFlow-Minimal", nfe, K),
                          ("SigmaFlow-Separate", nfe, K)))

        for target, kname in KRIT:
            print(f"\n{'=' * 92}\n{bench}, {kname}\n{'=' * 92}")
            print(f"{'Vergleich':<52}{'Diff':>8}{'95%-Intervall':>20}"
                  f"{'p':>10}  Holm")
            zeilen, ps = [], []
            for name, (a1, n1, k1), (a2, n2, k2) in paare:
                a = reihe(data, bench, a1, n1, k1, target)
                b = reihe(data, bench, a2, n2, k2, target)
                zeilen.append((name,) + test(a, b, rng))
                ps.append(zeilen[-1][4])
            for (name, d, lo, hi, p, n), h in zip(zeilen, holm(ps)):
                print(f"{name:<52}{d:>+8.2f}   [{lo:>+6.2f},{hi:>+6.2f}]"
                      f"{p:>10.4g}  {'ja' if h else 'nein'}")
            print(f"  Familie aus {len(ps)} Tests, familienweises Niveau 5 %, "
                  f"{sum(holm(ps))} halten.")


if __name__ == "__main__":
    main()
