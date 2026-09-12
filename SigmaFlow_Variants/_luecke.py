"""Wie viel geht beim Uebergang von RMSD<2 auf das kombinierte Kriterium verloren?

Fuer die nach dem Mixed Score gewaehlte Pose je Komplex. Die Ladefunktionen
kommen aus vergleich_rezept.py, damit es keine zweite Fassung gibt -- nur der
Teil VOR der Zellendefinition wird ausgefuehrt.
"""
import glob
import pathlib

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, wahl = ns["zelle"], ns["wahl"]


def g1(muster):
    t = glob.glob(muster)
    if not t:
        raise SystemExit(f"nicht gefunden: {muster}")
    return t[0]


C = "_cmp"
ZELLEN = {
    "SigmaDock 72h": (
        f"{C}/sd_endpunkt_40seeds/*/posebusters_redock_curve/*nfe25*/rd_*_seed*.csv",
        g1(f"{C}/sd_endpunkt_40seeds/*/learning_curve_cpu/*nfe25*/gnina_scores.csv")),
    "Minimal 72h": (
        f"{C}/endpunkt_min_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
        g1(f"{C}/endpunkt_min_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    "Separate 72h": (
        f"{C}/endpunkt_sep_nfe25/*/posebusters_redock_curve/*/rd_*_seed*.csv",
        g1(f"{C}/endpunkt_sep_nfe25/*/learning_curve_cpu/*/gnina_scores.csv")),
    "Minimal Rezept": (
        "_recipe/posebusters_redock/sigmaflow_minimal__recipe__confsampled/rd_*_seed*.csv",
        g1("_recipe/GNINA-SCORE-*/gnina_scores_*.csv")),
}

print(f"{'Modell':<16} {'RMSD<2':>8} {'PB-valid':>9} {'beides':>8} "
      f"{'Verlust':>8} {'gueltig|genau':>14} {'phi':>7}")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    a = wahl(m, "acc", "heuristik")
    v = wahl(m, "valid", "heuristik")
    b = a & v
    verlust = 100 * (a.mean() - b.mean())
    bedingt = 100 * b.sum() / a.sum()
    n11, n10 = int((a & v).sum()), int((a & ~v).sum())
    n01, n00 = int((~a & v).sum()), int((~a & ~v).sum())
    phi = (n11 * n00 - n10 * n01) / (((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5)
    print(f"{name:<16} {100*a.mean():8.2f} {100*v.mean():9.2f} {100*b.mean():8.2f} "
          f"{verlust:8.2f} {bedingt:13.1f}% {phi:7.3f}")

print(f"{'Paper SigmaDock':<16} {80.5:8.2f} {'--':>9} {79.9:8.2f} "
      f"{0.6:8.2f} {100*79.9/80.5:13.1f}% {'--':>7}")
