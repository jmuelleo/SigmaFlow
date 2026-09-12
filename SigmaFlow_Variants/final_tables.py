"""Final tables and figure: everything at 200 draws where available.

Cells at twenty-five steps keep their forty draws, which is the reference
budget. The five-step cells of the two flow arms now hold two hundred, so the
table can be read at forty, at the compute-matched count for each benchmark,
at one hundred and forty, and at the amortised asymptote of two hundred.
"""
import glob
import importlib
import math
import pathlib
import re

import numpy as np
import pandas as pd

pr = importlib.import_module("plot_ranking_lib")
from pb_bool import als_bool

HERE = pathlib.Path(".").resolve()
FIN = HERE / "final200"
AX = HERE / "astex"
CELL = "sched255ep_emergency__nfe5__sampled"
RNG = np.random.default_rng(0)


def redock(pattern):
    parts = []
    for f in sorted(glob.glob(pattern)):
        t = pd.read_csv(f)
        t["seed"] = int(re.search(r"seed(\d+)\.csv$", f).group(1))
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d["complex"] = d["file"].map(lambda p: pathlib.Path(p).name.split("__")[0])
    rmsd_col = next(c for c in d.columns if c.startswith("rmsd"))
    checks = [c for c in d.columns
              if c not in pr.LOADED | {"file", "molecule", "position", "seed",
                                       "complex", rmsd_col}]
    for c in checks:
        d[c] = als_bool(d[c])
    d["valid"] = d[checks].all(axis=1)
    d["p_pb"] = d[list(pr.PB_CHECKS)].mean(axis=1)
    d["acc_redock"] = als_bool(d[rmsd_col])   # symmetriekorrigiert, s. _finish
    d = d.sort_values(["seed", "file"]).drop_duplicates(["complex", "seed"],
                                                        keep="first")
    return d[["complex", "seed", "valid", "p_pb", "acc_redock"]]


def cell200(run, tree, redock_glob, gnina):
    base = FIN / run / tree / CELL
    return pr._finish(redock(redock_glob),
                      pd.read_csv(base / "per_pose_200.csv"),
                      pd.read_csv(gnina))


def build():
    d = {}
    for k, v in pr.CELLS.items():
        if k[2] == 25 or k[1] == "SigmaDock":
            d[k] = pr.load_pb(*v)
    for k, v in pr.AX_CELLS.items():
        if k[2] == 25 or k[1] == "SigmaDock":
            d[k] = pr.load_ax(*v)
    d[("AX85", "SigmaFlow-Minimal", 5)] = cell200(
        "SF_MIN_72H_s0_8653824", "learning_curve_cpu_astex",
        str(FIN / "astex_redock/sigmaflow_minimal__nfe5/rd_*_seed*.csv"),
        FIN / "GNINA-SCORE-sigmaflow_minimal__nfe5__astex_8686311"
            / "gnina_scores_sigmaflow_minimal__nfe5__astex.csv")
    d[("AX85", "SigmaFlow-Separate", 5)] = cell200(
        "SF_2H_72H_s0_8668713", "learning_curve_cpu_astex",
        str(FIN / "astex_redock/exp110__nfe5/rd_*_seed*.csv"),
        FIN / "GNINA-SCORE-exp110__nfe5__astex_8686313"
            / "gnina_scores_exp110__nfe5__astex.csv")
    for arm, run in (("SigmaFlow-Minimal", "SF_MIN_72H_s0_8653824"),
                     ("SigmaFlow-Separate", "SF_2H_72H_s0_8668713")):
        d[("PB308", arm, 5)] = cell200(
            run, "learning_curve_cpu",
            str(FIN / run / "posebusters_redock_curve" / CELL / "rd_*_seed*.csv"),
            FIN / run / "learning_curve_cpu" / CELL / "gnina_scores_200.csv")
    # Jede Zelle traegt ihren Schluessel, damit rule() die eingefrorenen
    # Zahlen aus ergebnisse_72h_*.csv nachschlagen kann statt neu zu rechnen.
    for schl, m in d.items():
        m.attrs["zelle"] = schl
    return d


def rule(m, target, K, how, exakt=False):
    """Top-1 je Komplex bei K Ziehungen.

    VORGABE IST DER GEZOGENE SCHAETZER (1000 Teilmengen je Komplex), weil in
    diesem Projekt genau der festgelegt ist -- Tabellen, Abbildungen und Tests
    sollen auf derselben Groesse sitzen. `exakt=True` liefert die geschlossene
    Form; sie schaetzt dieselbe Groesse ohne Monte-Carlo-Fehler und dient nur
    noch der Gegenprobe (final_comparison.py --exact).
    """
    if not exakt:
        schl = m.attrs.get("zelle")
        if schl is None:
            raise RuntimeError(
                "Zelle ohne Schluessel: rule() kann die eingefrorenen Zahlen "
                "nicht nachschlagen. Zellen immer ueber final_tables.build() "
                "beziehen.")
        import ergebnisse
        return ergebnisse.je_komplex(*schl, K, target, how)
    return _rule_geschlossen(m, target, K, how)


def _rule_geschlossen(m, target, K, how):
    out = {}
    for cid, g in m.groupby("complex"):
        y_all = g[target].to_numpy().astype(float)
        n = len(y_all)
        k = min(K, n)
        if how == "draw":
            out[cid] = y_all.mean()
            continue
        if how == "oracle":
            s = int(y_all.sum())
            out[cid] = 1.0 - (math.exp(pr.logC(n - s, k) - pr.logC(n, k))
                              if n - s >= k else 0.0)
            continue
        if how == "filter":
            # Harter PB-Filter, danach gnina. Das ist KEINE Score-Ordnung,
            # sondern eine lexikographische: erst alle gueltigen Posen nach
            # Bindungsenergie, dann alle ungueltigen nach Bindungsenergie.
            # Eine lexikographische Ordnung ist aber immer noch eine TOTALE
            # Ordnung, und die geschlossene Form fuer Top-1 bei K Ziehungen
            # braucht nichts weiter als das -- sie fragt nur, welche Pose in
            # einer Teilmenge die vorderste ist.
            #
            # Faellt alles durch den Filter, waehlt die Regel aus allen; das
            # kann nur schaden, nie helfen, und haelt sie mit den uebrigen
            # vergleichbar.
            h = g.sort_values(["valid", "affinity"], ascending=[False, True])
        else:
            col = "affinity" if how == "vinardo" else "heur"
            h = g.sort_values(col, ascending=(how == "vinardo"))
        y = h[target].to_numpy().astype(float)
        den = pr.logC(n, k)
        w = np.array([math.exp(pr.logC(n - 1 - i, k - 1) - den)
                      for i in range(n - k + 1)])
        out[cid] = float(y[:len(w)] @ w)
    return pd.Series(out)


def boot(a, b, n=8000):
    d = (b - a).to_numpy()
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, [2.5, 97.5]), min(1.0, max(
        2 * min((means <= 0).mean(), (means >= 0).mean()), 1 / n))


ROWS = {
    "PB308": [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
              ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
              ("SigmaFlow-Minimal", 5, 140), ("SigmaFlow-Separate", 5, 140),
              ("SigmaFlow-Minimal", 5, 200), ("SigmaFlow-Separate", 5, 200)],
    "AX85": [("SigmaDock", 25, 40), ("SigmaFlow-Minimal", 25, 40),
             ("SigmaFlow-Separate", 25, 40), ("SigmaDock", 5, 40),
             ("SigmaFlow-Minimal", 5, 101), ("SigmaFlow-Separate", 5, 101),
             ("SigmaFlow-Minimal", 5, 140), ("SigmaFlow-Separate", 5, 140),
             ("SigmaFlow-Minimal", 5, 200), ("SigmaFlow-Separate", 5, 200)],
}

if __name__ == "__main__":
    data = build()
    for k, m in sorted(data.items()):
        print(f"{k[0]:6} {k[1]:19} {k[2]:2} steps  {m.seed.nunique():3} seeds, "
              f"{len(m):6} poses")
    for bench, rows in ROWS.items():
        for target, name in (("both", "RMSD<2 and PB-valid with protein"),
                             ("acc", "RMSD<2")):
            print(f"\n=== {bench}, {name} ===")
            print(f"{'row':<34}{'per draw':>10}{'Vinardo':>10}"
                  f"{'mixed':>10}{'oracle':>10}")
            for arm, nfe, K in rows:
                m = data[(bench, arm, nfe)]
                vals = [100 * rule(m, target, K, r).mean()
                        for r in ("draw", "vinardo", "heuristic", "oracle")]
                print(f"{arm + ', ' + str(nfe) + '/' + str(K):<34}"
                      + "".join(f"{v:10.2f}" for v in vals))
            a = rule(data[(bench, "SigmaDock", 25)], target, 40, "heuristic")
            print("  -- mixed score, paired bootstrap vs SigmaDock 25/40 --")
            for arm, nfe, K in rows:
                if (arm, nfe) == ("SigmaDock", 25):
                    continue
                b = rule(data[(bench, arm, nfe)], target, K,
                         "heuristic").reindex(a.index)
                diff, ci, p = boot(a, b)
                print(f"     {arm + ', ' + str(nfe) + '/' + str(K):<32}"
                      f"{100 * diff:+7.2f} pp  "
                      f"[{100 * ci[0]:+6.2f},{100 * ci[1]:+6.2f}]  p={p:.3g}")
