"""Ergebnis- und Testtabelle zusammengefuehrt, je Benchmark eine.

EIN SCHAETZER FUER ALLES
    Vier Spalten Auswahlregel und drei Spalten Vergleich in einer Tabelle
    verlangen denselben Schaetzer, sonst stuenden Zahlen nebeneinander, die
    verschieden entstanden sind. Genommen werden die 1000 gezogenen
    Teilmengen. Am Ende wird gedruckt, wie weit die Regelspalten dadurch von
    der geschlossenen Form abweichen.

    Ausnahme ist "per draw": das ist K=1 und damit definitionsgemaess der
    Anteil ueber alle Posen, dort gibt es nichts zu ziehen.

DIE VERGLEICHSSPALTEN
    Differenz, Intervall und p-Wert beziehen sich auf die Spalte "mixed
    score", also auf den Ranker des Papers, und immer gegen SigmaDock bei
    25 Schritten und 40 Ziehungen.
"""
import io
import pathlib

import numpy as np
import pandas as pd

import final_tables as ft
from arm_tests import MATCHED, KRIT, test
from arm_tests_mc import MC, SEED

BEZUG = ("SigmaDock", 25, 40)
REGELN = [("draw", "per draw"), ("vinardo", "Vinardo"),
          ("heuristic", "mixed score"), ("oracle", "oracle")]
PUBLIZIERT = {("PB308", "both"): "79.9", ("PB308", "acc"): "80.5",
              ("AX85", "both"): "90.6", ("AX85", "acc"): "90.6"}
abweichung = []


def mc_zellen(m, ks, targets, rng):
    """{(target, K, regel): Serie}. Eine Permutationsmatrix je Komplex."""
    out = {(t, K, r): {} for t in targets for K in ks for r, _ in REGELN}
    for cid, g in m.groupby("complex"):
        h = g["heur"].to_numpy()
        v = g["affinity"].to_numpy()
        ys = {t: g[t].to_numpy().astype(bool) for t in targets}
        n = len(h)
        perm = np.argsort(rng.random((MC, n)), axis=1)
        z = np.arange(MC)
        for K in ks:
            idx = perm[:, :min(K, n)]
            best_h = idx[z, np.argmax(h[idx], axis=1)]
            best_v = idx[z, np.argmin(v[idx], axis=1)]   # Vinardo: kleiner ist besser
            for t in targets:
                y = ys[t]
                out[(t, K, "heuristic")][cid] = y[best_h].mean()
                out[(t, K, "vinardo")][cid] = y[best_v].mean()
                out[(t, K, "oracle")][cid] = y[idx].any(axis=1).mean()
                out[(t, K, "draw")][cid] = y.mean()
    return {k: pd.Series(v) for k, v in out.items()}


def bau(bench, titel, label):
    data = ft.build()
    mk = MATCHED[bench]
    rng = np.random.default_rng(SEED)
    konf = [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
            ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
            ("SigmaFlow-Minimal", 5, mk), ("SigmaFlow-Separate", 5, mk),
            ("SigmaFlow-Minimal", 5, 200), ("SigmaFlow-Separate", 5, 200)]

    noetig = {}
    for arm, nfe, K in konf:
        noetig.setdefault((arm, nfe), set()).add(K)
    zellen = {}
    for (arm, nfe), ks in noetig.items():
        zellen[(arm, nfe)] = mc_zellen(data[(bench, arm, nfe)], sorted(ks),
                                       [t for t, _ in KRIT], rng)
        for K in ks:                       # Gegenprobe zur Formel
            for r, _ in REGELN:
                e = 100 * ft.rule(data[(bench, arm, nfe)],
                                  "both", K, r).mean()
                m = 100 * zellen[(arm, nfe)][("both", K, r)].mean()
                abweichung.append(abs(e - m))

    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + titel + "}",
         r"\label{" + label + "}", r"\scriptsize",
         r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}lcccc rrr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{Top-1, four selection rules}"
         r" & \multicolumn{3}{c}{Mixed score, minus SigmaDock $25/40$} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
         r"Arm, steps / draws & " +
         " & ".join(n for _, n in REGELN) +
         r" & Pts. & $95\%$ interval & $p$ \\"]

    for target, _ in KRIT:
        kopf = ("RMSD $<2$\\,\\AA{} and PB-valid with protein"
                if target == "both" else "RMSD $<2$\\,\\AA")
        L += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{" + kopf + r"}}\\"]
        for arm, nfe, K in konf:
            werte = [100 * zellen[(arm, nfe)][(target, K, r)].mean()
                     for r, _ in REGELN]
            zeile = f"{arm}, ${nfe}$ / ${K}$"
            if nfe == 5 and K == mk:
                zeile = r"\textbf{" + zeile + r"}"
            zahlen = " & ".join(f"${w:.2f}$" for w in werte)
            if (arm, nfe, K) == BEZUG:
                rest = r"\multicolumn{3}{c}{\emph{reference}}"
            else:
                d, lo, hi, p, _ = test(
                    zellen[BEZUG[:2]][(target, BEZUG[2], "heuristic")],
                    zellen[(arm, nfe)][(target, K, "heuristic")],
                    np.random.default_rng(11))
                ps = "$<0.001$" if p < 1e-3 else f"${p:.3f}$"
                rest = (f"${d:+.2f}$ & $[{lo:+.2f},\,{hi:+.2f}]$ & {ps}")
            L.append(f"{zeile} & {zahlen} & {rest}" + r" \\")
        L.append(r"SigmaDock, published, $25$ / $40$ & -- & -- & $"
                 + PUBLIZIERT[(bench, target)] + r"$ & -- & "
                 r"\multicolumn{3}{c}{--} \\")
        L += [r"\addlinespace",
              r"\multicolumn{8}{@{}l}{\emph{SigmaFlow-Separate minus "
              r"SigmaFlow-Minimal, mixed score}}\\"]
        for nfe, K in ((25, 40), (5, mk), (5, 200)):
            d, lo, hi, p, _ = test(
                zellen[("SigmaFlow-Minimal", nfe)][(target, K, "heuristic")],
                zellen[("SigmaFlow-Separate", nfe)][(target, K, "heuristic")],
                np.random.default_rng(11))
            ps = "$<0.001$" if p < 1e-3 else f"${p:.3f}$"
            L.append(f"at ${nfe}$ / ${K}$ & -- & -- & -- & -- & "
                     f"${d:+.2f}$ & $[{lo:+.2f},\,{hi:+.2f}]$ & {ps}" + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


T_PB = (
    "PB308 at the $72$-hour endpoints, in percent over $307$ complexes. The "
    "left half gives Top-1 under four selection rules, the right half compares "
    "the mixed score against SigmaDock at twenty-five steps and $40$ draws, so "
    "a positive number favours the row. The compute-matched comparison is the "
    "bold row, five integration steps at $140$ draws; the $200$-draw rows are "
    "the amortised regime of Section~\\ref{sec:minimal-nfe}. Every value is "
    "estimated from one thousand random draw subsets per complex, and the "
    "interval and $p$ value come from a paired bootstrap over $8000$ resamples "
    "of the complexes. That bootstrap resamples complexes and draws, not "
    "training runs, of which there is one per arm. The published row is "
    "SigmaDock as reported by \\citet{prat2026sigmadock} at $40$ draws, "
    "reproduced for orientation and not measured here.")

T_AX = (
    "The Astex diverse set, otherwise as in Table~\\ref{tab:pb308}. The "
    "compute-matched count here is $101$ draws at five steps, and the "
    "bootstrap resamples the $85$ complexes.")

pb = bau("PB308", T_PB, "tab:pb308")
ax = bau("AX85", T_AX, "tab:ax85")
io.open("merged_tables.tex", "w", encoding="utf-8", newline="").write(
    pb + "\n\n" + ax + "\n")
print(pb + "\n\n" + ax)
print(f"\ngeschrieben: merged_tables.tex")
print(f"groesste Abweichung der Regelspalten zur geschlossenen Form: "
      f"{max(abweichung):.3f} Punkte")
