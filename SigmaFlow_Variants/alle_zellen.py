"""Alle Endpunkt-Zellen nebeneinander, gruppiert nach Trainingsregime.

Zwei Regime, unterschieden durch EINEN Schalter:
    72 h    -- kein --config, also pb_check = False (config.py:64).
               Gemessen: 18,37 % der Trainingsziele je Epoche nicht PB-valide.
    Rezept  -- conf_flow_paper.yaml:103 setzt pb_check = true.
               Gemessen: 12,60 % nicht PB-valide.
Beide trainierten auf `pdbbind-general` allein: die Kommandozeile in
train_final_72h.slurm:616 uebergibt --train_exps auch im Rezeptfall, und CLI
schlaegt YAML. Die Zweifachzaehlung von refined war in keinem Lauf aktiv.

SigmaDock gibt es nur im 72-h-Regime -- es wurde nie mit dem Rezept
nachtrainiert.

Alle Zellen werden auf die Seeds 0-39 beschraenkt: die nfe5-Zellen der
72-h-Laeufe haben 140, und Top-1 bei mehr Ziehungen waere nicht vergleichbar.
"""
import glob
import pathlib
import sys

quelle = pathlib.Path("vergleich_rezept.py").read_text(encoding="utf-8")
ns = {"__name__": "_defs", "__file__": "vergleich_rezept.py"}
exec(compile(quelle[:quelle.index("ZELLEN = {")], "vergleich_rezept.py", "exec"), ns)  # noqa: S102
zelle, wahl, boot = ns["zelle"], ns["wahl"], ns["boot"]


def g1(muster: str) -> str:
    # rekursiv: manche Zellen wurden mit vollem ARC-Pfad entpackt und liegen
    # deshalb mehrere Ebenen tiefer als die uebrigen.
    t = sorted(glob.glob(muster, recursive=True))
    if not t:
        sys.exit(f"nicht gefunden: {muster}")
    return t[0]


def paar(basis: str, nfe: int) -> tuple:
    """(Redock-Glob, gnina-CSV) fuer eine 72-h-Zelle."""
    rd = g1(f"{basis}/**/posebusters_redock_curve/*__nfe{nfe}__sampled")
    gn = g1(f"{basis}/**/learning_curve_cpu/*__nfe{nfe}__sampled/gnina_scores.csv")
    return f"{rd}/rd_*seed*.csv", gn


def rezept(tag: str) -> tuple:
    return (f"_recipe_all/posebusters_redock/{tag}/rd_*seed*.csv",
            g1(f"_recipe_all/GNINA-SCORE-{tag}_*/gnina_scores_{tag}.csv"))


ZELLEN = {}
for nfe in (25, 5):
    ZELLEN[("72 h", "SigmaDock", nfe)] = paar("_cmp/sd_endpunkt_40seeds", nfe)
    ZELLEN[("72 h", "Minimal", nfe)] = paar(f"_cmp/endpunkt_min_nfe{nfe}", nfe)
    ZELLEN[("72 h", "Separate", nfe)] = paar(f"_cmp/endpunkt_sep_nfe{nfe}", nfe)
    s = "__nfe5" if nfe == 5 else ""
    ZELLEN[("Rezept", "Minimal", nfe)] = rezept(f"sigmaflow_minimal{s}__recipe__confsampled")
    ZELLEN[("Rezept", "Separate", nfe)] = rezept(f"exp110{s}__recipe__confsampled")

daten = {}
print("=== Laden (auf Seeds 0-39 beschraenkt) ===")
for k, (rg, gn) in ZELLEN.items():
    m = zelle(rg, gn)
    m = m[m["seed"] < 40]
    daten[k] = m
    print(f"{k[0]:<8} {k[1]:<10} {k[2]:>3} Schr.  {len(m):6d} Posen, "
          f"{m['complex'].nunique():3d} Komplexe, {m['seed'].nunique():3d} Seeds")

REGIME = {"72 h": "pb_check aus, 18,4 % der Ziele ungueltig",
          "Rezept": "pb_check an,  12,6 % der Ziele ungueltig"}

for nfe in (25, 5):
    print(f"\n{'=' * 78}")
    print(f"  {nfe} INTEGRATIONSSCHRITTE, K = 40 ZIEHUNGEN, PB308")
    print("=" * 78)
    for ziel, titel in (("acc", "RMSD < 2 A"), ("valid", "PB-valid"),
                        ("beides", "RMSD<2 UND PB-valid")):
        print(f"\n--- {titel}")
        print(f"{'Regime':<9} {'Arm':<10} {'je Zug':>8} {'gnina':>8} "
              f"{'Mixed':>8} {'Orakel':>8}")
        for reg in ("72 h", "Rezept"):
            for arm in ("SigmaDock", "Minimal", "Separate"):
                k = (reg, arm, nfe)
                if k not in daten:
                    continue
                m = daten[k]
                z = [100 * m[ziel].mean()] + [100 * wahl(m, ziel, w).mean()
                                              for w in ("gnina", "heuristik", "orakel")]
                print(f"{reg:<9} {arm:<10} " + " ".join(f"{v:8.2f}" for v in z))

print(f"\n{'=' * 78}")
print("  REZEPT MINUS 72 h, gepaart je Komplex, 8000 Bootstrap-Ziehungen")
print("=" * 78)
print(f"{'Schritte':<9} {'Arm':<10} {'Kriterium':<12} {'Differenz':>11} "
      f"{'95%-Intervall':>20} {'p':>8}")
for nfe in (25, 5):
    for arm in ("Minimal", "Separate"):
        for ziel, t in (("acc", "RMSD<2"), ("valid", "PB-valid"), ("beides", "kombiniert")):
            a = wahl(daten[("72 h", arm, nfe)], ziel, "heuristik")
            b = wahl(daten[("Rezept", arm, nfe)], ziel, "heuristik")
            d, lo, hi, p = boot(a, b)
            print(f"{nfe:<9} {arm:<10} {t:<12} {d:>+10.2f} pp  "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {p:>8.4f}")
