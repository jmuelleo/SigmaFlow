"""Was passiert, wenn der Mixed Score mehr PoseBusters-Pruefungen sieht?

Die Paperfassung mittelt fuenf Pruefungen zu p_pb. Die beiden Kofaktor-
Abstaende fehlen darin, verursachen aber einen grossen Teil der genauen,
aber ungueltigen Auswahlen. Hier wird p_pb ueber erweiterte Mengen gebildet,
alles andere bleibt gleich: s = -b * p^4.
"""
import glob
import pathlib

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, LADE, BETA, PB_CHECKS = ns["zelle"], ns["LADE"], ns["BETA"], ns["PB_CHECKS"]

KOFAKTOR = ["minimum_distance_to_organic_cofactors",
            "minimum_distance_to_inorganic_cofactors"]


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

print(f"{'Modell':<16} {'Paper (5)':>10} {'+Kofakt (7)':>12} {'alle Checks':>12} {'Orakel':>8}")
for name, (rg, gc) in ZELLEN.items():
    m = zelle(rg, gc)
    rmsd_sp = next(c for c in m.columns if c.startswith("rmsd"))
    alle_ch = [c for c in m.columns
               if c not in LADE | {"file", "molecule", "position", "seed", "complex",
                                   rmsd_sp, "valid", "p_pb", "acc", "beides",
                                   "affinity", "heur"}]
    zeile = []
    for satz in (list(PB_CHECKS), list(PB_CHECKS) + KOFAKTOR, alle_ch):
        p = m[satz].mean(axis=1)
        s = -m["affinity"] * (p ** BETA)
        gew = m.loc[s.groupby(m["complex"]).idxmax()]
        zeile.append(100 * gew["beides"].mean())
    orakel = 100 * m.groupby("complex")["beides"].any().mean()
    print(f"{name:<16} {zeile[0]:10.2f} {zeile[1]:12.2f} {zeile[2]:12.2f} {orakel:8.2f}")
