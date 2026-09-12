"""Alle drei Ergebnistabellen mit dem FESTGELEGTEN 1000-Ziehungen-Schaetzer.

Loest tab_pb308_korrigiert.py / tab_ax85_korrigiert.py / tab_traj_korrigiert.py
ab, die versehentlich ft.rule (geschlossene Form) benutzten. Zellen kommen aus
final_tables.build(), also mit dem symmetriekorrigierten Erfolgsetikett.
"""
import numpy as np
import final_tables as ft
from zieh import top1_je_komplex

B, rng = 8000, np.random.default_rng(11)
d = ft.build()
NE, TR, SD = "SigmaFlow-Minimal", "SigmaFlow-Separate", "SigmaDock"
LAB = {SD: "SigmaDock", NE: "SigmaFlow-NE", TR: "SigmaFlow-TR"}
BS = chr(92)
REGEL = {"draw": "zug", "vinardo": "gnina", "mixed": "mixed", "oracle": "orakel"}


def r(bench, arm, nfe, K, ziel, how):
    return top1_je_komplex(d[(bench, arm, nfe)], ziel, K, REGEL[how])


def test(a, b):
    ix = a.index.intersection(b.index)
    x = (b.reindex(ix) - a.reindex(ix)).to_numpy()
    mi = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    p = min(1.0, max(2 * min((mi <= 0).mean(), (mi >= 0).mean()), 1.0 / B))
    return 100 * x.mean(), 100 * np.percentile(mi, 2.5), 100 * np.percentile(mi, 97.5), p


def pf(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def block(bench, ziel, titel, zeilen, kontraste, pub, letzte):
    ref = r(bench, SD, 25, 40, ziel, "mixed")
    print(BS + "multicolumn{10}{@{}l}{" + BS + "emph{" + titel + "}}" + BS * 2)
    for arm, nfe, K, fett in zeilen:
        mix = r(bench, arm, nfe, K, ziel, "mixed")
        nam = (BS + "textbf{" + LAB[arm] + "}") if fett else LAB[arm]
        if arm == SD and nfe == 25:
            rest = BS + "multicolumn{3}{c}{" + BS + "emph{reference}}"
        else:
            pt, lo, hi, p = test(ref, mix)
            rest = f"${pt:+.2f}$ & $[{lo:+.2f}," + BS + f",{hi:+.2f}]$ & {pf(p)}"
        print(f"{nam} & ${nfe}$ & ${K}$ & "
              f"${100*r(bench,arm,nfe,K,ziel,'draw').mean():.2f}$ & "
              f"${100*r(bench,arm,nfe,K,ziel,'vinardo').mean():.2f}$ & "
              f"${100*mix.mean():.2f}$ & "
              f"${100*r(bench,arm,nfe,K,ziel,'oracle').mean():.2f}$ & {rest} " + BS * 2)
    print(f"SigmaDock, published & $25$ & $40$ & -- & -- & ${pub}$ & -- & "
          + BS + "multicolumn{3}{c}{--} " + BS * 2)
    print(BS + "addlinespace")
    print(BS + "multicolumn{10}{@{}l}{" + BS + "emph{SigmaFlow-TR minus "
          "SigmaFlow-NE, mixed score}}" + BS * 2)
    for nfe, K in kontraste:
        pt, lo, hi, p = test(r(bench, NE, nfe, K, ziel, "mixed"),
                             r(bench, TR, nfe, K, ziel, "mixed"))
        print(f"at ${nfe}$ / ${K}$ & -- & -- & -- & -- & -- & -- & "
              f"${pt:+.2f}$ & $[{lo:+.2f}," + BS + f",{hi:+.2f}]$ & {pf(p)} " + BS * 2)
    print(BS + ("bottomrule" if letzte else "midrule"))


print("%%%%%%%%%%%%%%%%  tab:pb308  %%%%%%%%%%%%%%%%")
ZP = [(SD,25,40,0),(NE,25,40,0),(TR,25,40,0),(NE,5,140,1),(TR,5,140,1),
      (NE,5,200,0),(TR,5,200,0)]
ZPa = ZP[:3] + [(SD,5,40,0)] + ZP[3:]
KP = [(25,40),(5,140),(5,200)]
block("PB308","both","RMSD $<2$"+BS+","+BS+"AA{} and PB-valid with protein",ZP,KP,"79.9",False)
block("PB308","acc","RMSD $<2$"+BS+","+BS+"AA",ZPa,KP,"80.5",True)

print()
print("%%%%%%%%%%%%%%%%  tab:ax85  %%%%%%%%%%%%%%%%")
ZA = [(SD,25,40,0),(NE,25,40,0),(TR,25,40,0),(SD,5,40,0),
      (NE,5,101,1),(TR,5,101,1),(NE,5,200,0),(TR,5,200,0)]
KA = [(25,40),(5,101),(5,200)]
block("AX85","both","RMSD $<2$"+BS+","+BS+"AA{} and PB-valid with protein",ZA,KA,"90.6",False)
block("AX85","acc","RMSD $<2$"+BS+","+BS+"AA",ZA,KA,"90.6",True)

print()
print("%%%%%%%%%%%%%%%%  tab:traj  %%%%%%%%%%%%%%%%")
KS = [1,5,10,20,40,100,140,200]
for bench in ("PB308","AX85"):
    for ziel, titel in (("acc","RMSD $<2$"+BS+","+BS+"AA"),
                        ("both","RMSD $<2$"+BS+","+BS+"AA{} and PB-valid with protein")):
        print(BS+"multicolumn{10}{@{}l}{"+BS+"emph{"+titel+"}}"+BS*2)
        for arm, nfe, kmax in ((SD,25,40),(NE,25,40),(TR,25,40),(SD,5,40),
                               (NE,5,200),(TR,5,200)):
            zellen = [f"${100*r(bench,arm,nfe,K,ziel,'mixed').mean():.1f}$"
                      if K <= kmax else "--" for K in KS]
            print(f"{LAB[arm]} & ${nfe}$ & " + " & ".join(zellen) + " " + BS*2)
        print(BS + ("midrule" if ziel=="acc" else "bottomrule"))
    print("%%%% ---- naechste Tabelle ----")
