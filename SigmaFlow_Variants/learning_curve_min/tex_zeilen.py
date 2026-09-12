"""Erzeugt die LaTeX-Zeilen der beiden Kurventabellen aus den CSV.

Von Hand abgetippte Zahlen sind die haeufigste Fehlerquelle in einer Arbeit
und die am schwersten zu findende: nichts bricht, die Tabelle sieht richtig
aus. Deshalb kommen die Zeilen aus derselben Datei, aus der auch die
Abbildung gezeichnet wird.

JEDES Literal hier ist ein Rohstring. LaTeX aus Python ohne r"" zu erzeugen
ist eine Falle: \a ist das Klingelzeichen, \t ein Tabulator, \f ein
Seitenvorschub -- aus \addlinespace wird lautlos "ddlinespace", und Python
warnt nur, statt zu scheitern.
"""
import pathlib

import pandas as pd

HIER = pathlib.Path(__file__).resolve().parent
ARM = "SigmaFlow-Minimal"
d = pd.read_csv(HIER / "kurve_sigmaflow_minimal_tidy.csv")
# Seit dem Umbau auf mehrere Arme traegt kurve_paare.csv eine arm-Spalte.
# Ohne diesen Filter liefen zwei Arme in dieselben Tabellenzeilen, sobald
# SigmaDock dazukommt -- und zwar lautlos, weil die Zeilenzahl stimmt.
p = pd.read_csv(HIER / "kurve_paare.csv")
p = p[p["arm"] == ARM]
B = 8000


def z(v, lo, hi):
    return rf"${v:.2f}$ \small$[{lo:.2f},{hi:.2f}]$"


print(r"% ---- tab:curve-minimal ----")
for nfe in (25, 5):
    print(rf"\multicolumn{{6}}{{@{{}}l}}{{\itshape {nfe} integration steps}}"
          r" \\[1pt]")
    for _, r in d[d.nfe == nfe].sort_values("epoch").iterrows():
        print(rf"${int(r.epoch)}$ & ${r.walltime_h:g}$ & "
              rf"{z(r.u2, r.u2_lo, r.u2_hi)} & "
              rf"{z(r.pb_lig, r.pb_lig_lo, r.pb_lig_hi)} & "
              rf"{z(r.pb_prot, r.pb_prot_lo, r.pb_prot_hi)} & "
              rf"{z(r.u2_prot, r.u2_prot_lo, r.u2_prot_hi)} \\")
    if nfe == 25:
        print(r"\addlinespace")

NAME = {"u2": r"RMSD $<2$\,\AA",
        "pb_prot": r"PB-valid, with protein",
        "u2_prot": r"$<2$\,\AA{} and PB-valid"}
print("\n" + r"% ---- tab:curve-paired ----")
for nfe in (25, 5):
    print(rf"\multicolumn{{4}}{{@{{}}l}}{{\itshape {nfe} integration steps}}"
          r" \\[1pt]")
    for metrik in ("u2", "pb_prot", "u2_prot"):
        q = p[(p.nfe == nfe) & (p.metrik == metrik)].sort_values("von_epoch")
        zellen = []
        for _, r in q.iterrows():
            fett = r"\mathbf" if (r.lo > 0 or r.hi < 0) else r"\mathrm"
            ptxt = (r"<\!1.3\text{e-}4" if r.p <= 1 / B
                    else f"{r.p:.4f}".rstrip("0"))
            zellen.append(rf"${fett}{{{r.delta_pp:+.2f}}}$ \small$({ptxt})$")
        print(f"{NAME[metrik]} & " + " & ".join(zellen) + r" \\")
    if nfe == 25:
        print(r"\addlinespace")
