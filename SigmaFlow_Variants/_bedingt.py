"""Wie gut sagt PB-Validitaet die Platzierung voraus?

Zwei Ebenen: ueber ALLE erzeugten Posen (was die Validitaet als Signal wert
ist) und ueber die vom Mixed Score gewaehlte Pose je Komplex.
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

print("UEBER ALLE ERZEUGTEN POSEN")
print(f"{'Modell':<16} {'P(valid)':>9} {'P(<2A)':>8} {'P(<2A|valid)':>13} "
      f"{'P(<2A|ungueltig)':>17} {'Hub':>7}")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    v, a = m["valid"], m["acc"]
    p_av = a[v].mean()
    p_au = a[~v].mean()
    print(f"{name:<16} {100*v.mean():9.2f} {100*a.mean():8.2f} {100*p_av:13.2f} "
          f"{100*p_au:17.2f} {p_av/p_au:7.2f}x")

print()
print("FUER DIE VOM MIXED SCORE GEWAEHLTE POSE")
print(f"{'Modell':<16} {'P(valid)':>9} {'P(<2A)':>8} {'P(<2A|valid)':>13} "
      f"{'P(<2A|ungueltig)':>17} {'Hub':>7}")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    sel = m.loc[m.groupby("complex")["heur"].idxmax()]
    v, a = sel["valid"], sel["acc"]
    p_av = a[v].mean()
    p_au = a[~v].mean() if (~v).any() else float("nan")
    print(f"{name:<16} {100*v.mean():9.2f} {100*a.mean():8.2f} {100*p_av:13.2f} "
          f"{100*p_au:17.2f} {p_av/p_au:7.2f}x")
