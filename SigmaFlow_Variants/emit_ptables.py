"""Zwei LaTeX-Tabellen mit den p-Werten, je Benchmark eine.

AUFBAU SYMMETRISCH ZU tab:pb308
    Dieselben acht Zeilen in derselben Reihenfolge und derselben Gruppierung
    nach Kriterium. Bezug ist SigmaDock bei 25 Schritten und 40 Ziehungen, die
    Konfiguration, gegen die die Arbeit vergleicht; diese Zeile ist deshalb der
    Bezug und traegt keine Differenz. Darunter ein kurzer Block Arm gegen Arm,
    weil sich diese Frage nicht gegen SigmaDock stellen laesst.

SCHAETZER
    Ausschliesslich 1000 gezogene Teilmengen je Komplex, wie in den
    Ergebnistabellen. Intervall und p-Wert aus einem gepaarten Bootstrap ueber
    8000 Neuziehungen der Komplexmenge.
"""
import io
import pathlib

import numpy as np

import final_tables as ft
from arm_tests import MATCHED, KRIT, test
from arm_tests_mc import SEED, mc_reihen

BEZUG = ("SigmaDock", 25, 40)


def zeilen_fuer(bench):
    """Die acht Konfigurationen von tab:pb308, in derselben Reihenfolge."""
    mk = MATCHED[bench]
    return [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
            ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
            ("SigmaFlow-Minimal", 5, mk), ("SigmaFlow-Separate", 5, mk),
            ("SigmaFlow-Minimal", 5, 200), ("SigmaFlow-Separate", 5, 200)]


def name(arm, nfe, K):
    return f"{arm}, ${nfe}$ / ${K}$"


def pfmt(p):
    return "$<0.001$" if p < 1e-3 else f"${p:.3f}$"


def bau(bench, titel, label):
    data = ft.build()
    mk = MATCHED[bench]
    rng = np.random.default_rng(SEED)

    konf = zeilen_fuer(bench)
    arm_gegen_arm = [(25, 40), (5, mk), (5, 200)]

    noetig = {}
    for arm, nfe, K in konf:
        noetig.setdefault((arm, nfe), set()).add(K)
    reihen = {(arm, nfe): mc_reihen(data, bench, arm, nfe, sorted(ks), rng)
              for (arm, nfe), ks in noetig.items()}

    def vergleich(a, b, target):
        """a minus b."""
        return test(reihen[(b[0], b[1])][(target, b[2])],
                    reihen[(a[0], a[1])][(target, a[2])],
                    np.random.default_rng(11))

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{" + titel + "}", r"\label{" + label + "}", r"\small",
         r"\begin{tabular}{@{}lrrrc@{}}", r"\toprule",
         r"Arm, steps / draws & Points & \multicolumn{2}{c}{$95\%$ interval}"
         r" & $p$ \\", r"\cmidrule(lr){3-4}"]

    for target, _ in KRIT:
        kopf = ("RMSD $<2$\\,\\AA{} and PB-valid with protein"
                if target == "both" else "RMSD $<2$\\,\\AA")
        L += [r"\midrule",
              r"\multicolumn{5}{@{}l}{\emph{" + kopf +
              r", against SigmaDock, $25$ / $40$}}\\"]
        for arm, nfe, K in konf:
            if (arm, nfe, K) == BEZUG:
                L.append(name(arm, nfe, K) +
                         r" & \multicolumn{4}{c}{\emph{reference}} \\")
                continue
            d, lo, hi, p, _ = vergleich((arm, nfe, K), BEZUG, target)
            zeile = name(arm, nfe, K)
            if nfe == 5 and K == mk:      # der walltime-gematchte Vergleich
                zeile = r"\textbf{" + zeile + r"}"
            L.append(f"{zeile} & ${d:+.2f}$ & ${lo:+.2f}$ & ${hi:+.2f}$ & "
                     f"{pfmt(p)}" + r" \\")
        L += [r"\addlinespace",
              r"\multicolumn{5}{@{}l}{\emph{SigmaFlow-Separate minus "
              r"SigmaFlow-Minimal}}\\"]
        for nfe, K in arm_gegen_arm:
            d, lo, hi, p, _ = vergleich(("SigmaFlow-Separate", nfe, K),
                                        ("SigmaFlow-Minimal", nfe, K), target)
            L.append(f"at ${nfe}$ / ${K}$ & ${d:+.2f}$ & ${lo:+.2f}$ & "
                     f"${hi:+.2f}$ & {pfmt(p)}" + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


TITEL_PB = (
    "Paired comparisons on PB308 at the $72$-hour endpoints, laid out row for "
    "row as Table~\\ref{tab:pb308}. Within each criterion the upper block "
    "gives each configuration minus SigmaDock at twenty-five steps and $40$ "
    "draws, so a positive number favours the row; the lower block gives "
    "SigmaFlow-Separate minus SigmaFlow-Minimal at matching settings. "
    "Per-complex values are estimated from one thousand random draw subsets, "
    "and the interval and $p$ value come from a paired bootstrap over $8000$ "
    "resamples of the $307$ complexes. The rows in bold are the "
    "compute-matched comparisons. Those two rows, their counterpart under the "
    "other criterion and the four on Astex form the pre-specified family of "
    "eight, of which seven hold under Holm at a family-wise $5\\%$, the "
    "exception being SigmaFlow-Separate here under the combined criterion. "
    "Every other row is secondary and reported uncorrected. The bootstrap "
    "resamples complexes and draws, not training runs, of which there is one "
    "per arm.")

TITEL_AX = (
    "Paired comparisons on the Astex diverse set, otherwise as in "
    "Table~\\ref{tab:ptests-pb308}. The compute-matched count here is $101$ "
    "draws at five steps against $40$ at twenty-five, and the bootstrap "
    "resamples the $85$ complexes.")

pb = bau("PB308", TITEL_PB, "tab:ptests-pb308")
ax = bau("AX85", TITEL_AX, "tab:ptests-ax85")
io.open("ptables.tex", "w", encoding="utf-8", newline="").write(
    pb + "\n\n" + ax + "\n")
print(pb + "\n\n" + ax)
print("\ngeschrieben: ptables.tex")
