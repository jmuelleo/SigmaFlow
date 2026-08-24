"""Eine Gesamttabelle ueber alle vorhandenen Auswertungszellen.

Die Zellen des Versuchsplans sind:

    Schrittzahl   x   Konformerquelle   x   Arm
    {5, 25}           {bound, sampled}      {Minimal, Separate, SigmaDock}

Vorhanden sind drei der vier Kombinationen. sampled x 5 Schritte ist auf ARC
gesampelt, aber weder heruntergeladen noch mit PoseBusters geprueft -- diese
Zeilen tragen den Vermerk "(rechnet auf ARC)" statt einer Zahl.

Warum das explizit dasteht statt weggelassen zu werden: eine fehlende Zeile
sieht in einer Tabelle aus wie eine nicht gestellte Frage. Eine Zeile mit
Vermerk sagt, dass die Frage gestellt ist und die Antwort noch aussteht.

AUSGABE   Markdown auf stdout.
AUFRUF    python gesamttabelle.py > gesamttabelle.md
"""
from __future__ import annotations

import csv
import os

DATA = os.path.join("..", "..", "Thesis Visualisierungen", "data")

# (Anzeigename, Schritte, Konformer, per_draw-Datei, selection-Datei, Seeds, gnina?)
ZELLEN = [
    ("25 Schritte, bound", 25, "bound",
     "per_draw_80seeds.csv", "selection_curves_80seeds.csv", 80, True),
    ("25 Schritte, sampled", 25, "sampled",
     "per_draw_papersetup40.csv", "selection_curves_papersetup40.csv", 40, True),
    ("5 Schritte, bound", 5, "bound",
     "per_draw_nfe5_40seeds.csv", "selection_curves_nfe5_40seeds.csv", 40, False),
    ("5 Schritte, sampled", 5, "sampled", None, None, 40, None),
]
ARME = ("Minimal", "Separate", "SigmaDock")
METRIKEN = [
    ("pb_valid_no_protein", "PB-valid ohne Protein"),
    ("pb_valid_with_protein", "PB-valid mit Protein"),
    ("rmsd_lt_2.0A", "RMSD < 2 A"),
    ("lt2A_and_valid_no_protein", "< 2 A und PB-valid ohne Protein"),
    ("lt2A_and_valid_with_protein", "< 2 A und PB-valid mit Protein"),
]
KS = (1, 2, 3, 5, 10, 20, 40, 80)
FEHLT = "(rechnet auf ARC)"


def lade_per_draw(f):
    d = {}
    for r in csv.DictReader(open(os.path.join(DATA, f), encoding="utf-8")):
        d[(r["arm"], r["metric"])] = (float(r["pct"]), float(r["ci_lo"]),
                                      float(r["ci_hi"]))
    return d


def lade_kurven(f):
    d = {}
    for r in csv.DictReader(open(os.path.join(DATA, f), encoding="utf-8")):
        t = r.get("top1_affinity_pct")
        d[(r["arm"], r["metric"], int(r["k"]))] = (
            float(r["random_pct"]), float(r["oracle_pct"]),
            float(t) if t else None)
    return d


def z(x, n=2):
    """Deutsche Dezimalschreibweise."""
    return f"{x:.{n}f}".replace(".", ",")


def main() -> int:
    PD, SC = {}, {}
    for name, _, _, f1, f2, _, _ in ZELLEN:
        if f1:
            PD[name], SC[name] = lade_per_draw(f1), lade_kurven(f2)

    print("## Block A — Anteil je Ziehung, mit 95-%-Wilson-Intervall\n")
    print("| Schritte | Konformer | Arm | Kenngröße | Anteil | 95-%-Intervall | Posen |")
    print("|---|---|---|---|---:|---|---:|")
    for name, ns, kf, f1, _, seeds, _ in ZELLEN:
        for arm in ARME:
            for key, lab in METRIKEN:
                if f1 is None:
                    print(f"| {ns} | {kf} | {arm} | {lab} | {FEHLT} | — | — |")
                    continue
                p, lo, hi = PD[name][(arm, key)]
                print(f"| {ns} | {kf} | {arm} | {lab} | **{z(p)} %** | "
                      f"[{z(lo)}; {z(hi)}] | {209*seeds} |")

    for key, lab in METRIKEN:
        print(f"\n## Block B — Oracle@k, {lab}\n")
        print("| Schritte | Konformer | Arm | " +
              " | ".join(f"k={k}" for k in KS) + " |")
        print("|---|---|---|" + "---:|" * len(KS))
        for name, ns, kf, f1, _, seeds, _ in ZELLEN:
            for arm in ARME:
                if f1 is None:
                    print(f"| {ns} | {kf} | {arm} | " +
                          " | ".join([FEHLT] + ["—"] * (len(KS) - 1)) + " |")
                    continue
                zellen = []
                for k in KS:
                    v = SC[name].get((arm, key, k))
                    zellen.append(z(v[1]) if v else "—")
                print(f"| {ns} | {kf} | {arm} | " + " | ".join(zellen) + " |")

    for key, lab in METRIKEN:
        print(f"\n## Block C — Auswahl nach gnina/Vinardo (Top-1@k), {lab}\n")
        print("| Schritte | Konformer | Arm | Zufall | " +
              " | ".join(f"k={k}" for k in KS) + " |")
        print("|---|---|---|---:|" + "---:|" * len(KS))
        for name, ns, kf, f1, _, seeds, gn in ZELLEN:
            for arm in ARME:
                if f1 is None:
                    print(f"| {ns} | {kf} | {arm} | " +
                          " | ".join([FEHLT] + ["—"] * len(KS)) + " |")
                    continue
                if not gn:
                    print(f"| {ns} | {kf} | {arm} | " +
                          " | ".join(["(nie gescort)"] + ["—"] * len(KS)) + " |")
                    continue
                ran = SC[name][(arm, key, KS[0])][0]
                zellen = []
                for k in KS:
                    v = SC[name].get((arm, key, k))
                    zellen.append(z(v[2]) if v and v[2] is not None else "—")
                print(f"| {ns} | {kf} | {arm} | {z(ran)} | " +
                      " | ".join(zellen) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
