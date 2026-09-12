"""Astex at 200 draws, while the PoseBusters cells are still being evaluated."""
import importlib

import final_tables as ft

pr = importlib.import_module("plot_ranking_lib")

data = {}
for k, v in pr.AX_CELLS.items():
    if k[2] == 25 or k[1] == "SigmaDock":
        data[k] = pr.load_ax(*v)
data[("AX85", "SigmaFlow-Minimal", 5)] = ft.cell200(
    "SF_MIN_72H_s0_8653824", "learning_curve_cpu_astex",
    str(ft.FIN / "astex_redock/sigmaflow_minimal__nfe5/rd_*_seed*.csv"),
    ft.FIN / "GNINA-SCORE-sigmaflow_minimal__nfe5__astex_8686311"
           / "gnina_scores_sigmaflow_minimal__nfe5__astex.csv")
data[("AX85", "SigmaFlow-Separate", 5)] = ft.cell200(
    "SF_2H_72H_s0_8668713", "learning_curve_cpu_astex",
    str(ft.FIN / "astex_redock/exp110__nfe5/rd_*_seed*.csv"),
    ft.FIN / "GNINA-SCORE-exp110__nfe5__astex_8686313"
           / "gnina_scores_exp110__nfe5__astex.csv")

for k, m in sorted(data.items()):
    print(f"{k[1]:19} {k[2]:2} steps  {m.seed.nunique():3} seeds, {len(m):6} poses")

for target, name in (("both", "RMSD<2 and PB-valid with protein"),
                     ("acc", "RMSD<2")):
    print(f"\n=== Astex, {name} ===")
    print(f"{'row':<34}{'per draw':>10}{'Vinardo':>10}{'mixed':>10}{'oracle':>10}")
    for arm, nfe, K in ft.ROWS["AX85"]:
        m = data[("AX85", arm, nfe)]
        vals = [100 * ft.rule(m, target, K, r).mean()
                for r in ("draw", "vinardo", "heuristic", "oracle")]
        print(f"{arm + ', ' + str(nfe) + '/' + str(K):<34}"
              + "".join(f"{v:10.2f}" for v in vals))
    a = ft.rule(data[("AX85", "SigmaDock", 25)], target, 40, "heuristic")
    print("  -- mixed score, paired bootstrap vs SigmaDock 25/40 --")
    for arm, nfe, K in ft.ROWS["AX85"]:
        if (arm, nfe) == ("SigmaDock", 25):
            continue
        b = ft.rule(data[("AX85", arm, nfe)], target, K,
                    "heuristic").reindex(a.index)
        diff, ci, p = ft.boot(a, b)
        print(f"     {arm + ', ' + str(nfe) + '/' + str(K):<32}"
              f"{100 * diff:+7.2f} pp  [{100 * ci[0]:+6.2f},{100 * ci[1]:+6.2f}]"
              f"  p={p:.3g}")
