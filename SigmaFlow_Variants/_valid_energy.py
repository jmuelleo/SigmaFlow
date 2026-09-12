"""Innerhalb der PB-gueltigen Posen nach reiner Energie ranken.

Vier Regeln im Vergleich, alle auf denselben Posensaetzen:
  A  gnina allein            argmin(affinity) ueber alle Posen
  B  Mixed Score (Paper)     argmax(-b * p^4) ueber alle Posen
  C  Filter + gnina          argmin(affinity) NUR unter PB-gueltigen
  D  Filter + Mixed Score    argmax(-b * p^4) NUR unter PB-gueltigen

Rueckfall bei C und D: bleibt fuer einen Komplex keine gueltige Pose, wird
ungefiltert geranked. Sonst verschwaende der Komplex und die Quote stiege
kuenstlich.
"""
import glob
import pathlib

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle = ns["zelle"]


def g1(m):
    t = glob.glob(m)
    if not t:
        raise SystemExit(m)
    return t[0]


C = "_cmp"
ZELLEN = {
    "SigmaDock 72h": (f"{C}/sd_endpunkt_40seeds/*/posebusters_redock_curve/*nfe25*/rd_*_seed*.csv",
                      g1(f"{C}/sd_endpunkt_40seeds/*/learning_curve_cpu/*nfe25*/gnina_scores.csv")),
    "Minimal 72h":   (f"{C}/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
                      g1(f"{C}/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    "Minimal Rezept": ("_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
                       g1("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")),
}

for ziel, titel in (("beides", "RMSD<2 UND PB-valid"), ("acc", "RMSD<2")):
    print(f"\n########## {titel}")
    print(f"{'Modell':<16} {'A gnina':>9} {'B mixed':>9} "
          f"{'C Filter+gnina':>15} {'D Filter+mixed':>15} {'Orakel':>8}")
    for name, (rg, gc) in ZELLEN.items():
        m = zelle(rg, gc)
        hat = m["valid"].groupby(m["complex"]).transform("any")
        benutzt = m["valid"] | ~hat
        sub = m[benutzt]

        A = m.loc[m.groupby("complex")["affinity"].idxmin()][ziel].mean()
        B = m.loc[m.groupby("complex")["heur"].idxmax()][ziel].mean()
        Cq = sub.loc[sub.groupby("complex")["affinity"].idxmin()][ziel].mean()
        D = sub.loc[sub.groupby("complex")["heur"].idxmax()][ziel].mean()
        orc = m.groupby("complex")[ziel].any().mean()
        print(f"{name:<16} {100*A:9.2f} {100*B:9.2f} {100*Cq:15.2f} "
              f"{100*D:15.2f} {100*orc:8.2f}")
    print(f"   (Komplexe ohne gueltige Pose: "
          + ", ".join(f"{n}={int((~zelle(rg, gc)['valid'].groupby(zelle(rg, gc)['complex']).any()).sum())}"
                      for n, (rg, gc) in ZELLEN.items()) + ")")
