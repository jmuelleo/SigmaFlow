import runpy, sys, io, contextlib
import pandas as pd

# vergleich_rezept.py laufen lassen, aber die Ausgabe unterdruecken --
# wir wollen nur das geladene `daten`-Dictionary.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path("vergleich_rezept.py")
daten, wahl = ns["daten"], ns["wahl"]

for name, m in daten.items():
    a = wahl(m, "acc", "heuristik")
    v = wahl(m, "valid", "heuristik")
    b = wahl(m, "beides", "heuristik")
    n = len(a)
    print(f"=== {name}   ({n} Komplexe) ===")
    print(f"  P(RMSD<2)        = {100*a.mean():6.2f}")
    print(f"  P(PB-valid)      = {100*v.mean():6.2f}")
    print(f"  P(beides)        = {100*b.mean():6.2f}")
    print(f"  Gegenprobe a&v   = {100*(a & v).mean():6.2f}   (muss gleich P(beides) sein)")
    print()
    print("  Vierfeldertafel der GEWAEHLTEN Pose:")
    print(f"    genau & gueltig : {100*(a & v).mean():6.2f}")
    print(f"    genau, ungueltig: {100*(a & ~v).mean():6.2f}")
    print(f"    gueltig, ungenau: {100*(~a & v).mean():6.2f}")
    print(f"    keines von beiden:{100*(~a & ~v).mean():6.2f}")
    print(f"    Summe            :{100*((a&v)|(a&~v)|(~a&v)|(~a&~v)).mean():6.2f}")
    print()
    lo = max(0.0, a.mean() + v.mean() - 1)
    print(f"  Frechet-Schranken: [{100*lo:.2f}, {100*min(a.mean(), v.mean()):.2f}]")
    print(f"  bei Unabhaengigkeit: {100*a.mean()*v.mean():.2f}")
    print(f"  Phi-Korrelation    : {pd.crosstab(a, v).pipe(lambda t: (t.iloc[1,1]*t.iloc[0,0]-t.iloc[1,0]*t.iloc[0,1])/ (t.sum(1).prod()*t.sum(0).prod())**0.5):.3f}")
    print()
