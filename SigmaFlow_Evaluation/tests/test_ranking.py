"""Unit-Tests fuer den Ranking-Evaluator.

Der Anlass ist konkret: eine frueherer Oracle@K-Fassung mittelte RMSDs und
schwellte danach. Das verfaelschte K=1 von 3.9 % auf 0.0 % - ein Fehler, der
plausibel aussah und keine Ausnahme warf. Diese Tests fixieren die Definition,
damit er nicht wiederkommt.

    python SigmaFlow_Evaluation/tests/test_ranking.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SigmaFlow_Evaluation.metrics.ranking import (  # noqa: E402
    PoseRecord, calibration, evaluate_ranking, oracle_at_k, ranking_quality, top1_at_k,
)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def rec(cid, pid, rmsd, score=None):
    return PoseRecord(complex_id=cid, pose_id=pid, rmsd=rmsd, score=score)


print("=== 1. Oracle@K: Definition erst minimieren, dann schwellen ===")
# Ein Komplex, 10 Posen, genau eine unter 2 A.
recs = [rec("A", str(i), 5.0) for i in range(9)] + [rec("A", "9", 1.0)]
o1 = oracle_at_k(recs, 1, n_resamples=2000, seed=0)
o10 = oracle_at_k(recs, 10)
check("K=1 trifft die gute Pose in ~10 % der Faelle",
      0.05 < o1["success_2A"] < 0.15, f"{100*o1['success_2A']:.1f} %")
check("K=10 findet sie immer", o10["success_2A"] == 1.0)
check("Mittelung-dann-Schwellen waere FALSCH: Mittel-RMSD = 4.6 A > 2 A",
      abs(np.mean([r.rmsd for r in recs]) - 4.6) < 1e-9,
      "genau dieser Wert erzeugte frueher 0.0 %")

print("\n=== 2. Monotonie in K ===")
rng = np.random.default_rng(0)
recs = [rec(f"C{c}", str(i), float(rng.gamma(2.0, 1.5)))
        for c in range(60) for i in range(40)]
succ = [oracle_at_k(recs, k, seed=1)["success_2A"] for k in (1, 5, 10, 20, 40)]
med = [oracle_at_k(recs, k, seed=1)["median_rmsd"] for k in (1, 5, 10, 20, 40)]
check("Erfolgsrate monoton steigend", all(a <= b + 1e-9 for a, b in zip(succ, succ[1:])),
      " -> ".join(f"{100*s:.1f}%" for s in succ))
check("Median-RMSD monoton fallend", all(a >= b - 1e-9 for a, b in zip(med, med[1:])),
      " -> ".join(f"{m:.2f}" for m in med))

print("\n=== 3. K=1 reproduziert die Einzelsample-Leistung ===")
single = np.mean([r.rmsd < 2.0 for r in recs])
o1 = oracle_at_k(recs, 1, n_resamples=200, seed=2)
check("K=1 == Anteil aller Einzelposen unter 2 A",
      abs(o1["success_2A"] - single) < 0.02,
      f"{100*o1['success_2A']:.2f} % vs {100*single:.2f} %")

print("\n=== 4. Keine Vermischung ueber Komplexe hinweg ===")
# Komplex X hat nur schlechte, Y nur gute Posen. Oracle@K darf X nicht retten.
recs = [rec("X", str(i), 9.0) for i in range(5)] + [rec("Y", str(i), 0.5) for i in range(5)]
o = oracle_at_k(recs, 5)
check("Erfolgsrate genau 50 % (Y ja, X nein)", abs(o["success_2A"] - 0.5) < 1e-9,
      f"{100*o['success_2A']:.1f} %")
check("beide Komplexe gezaehlt", o["n_complexes"] == 2)
check("X bleibt schlecht", abs(o["per_complex"]["X"] - 9.0) < 1e-9)

print("\n=== 5. Fehlende Posen explizit, nicht still ===")
recs = [rec("A", "0", 1.0), rec("A", "1", float("nan")), rec("B", "0", 3.0)]
o = oracle_at_k(recs, 2)
check("NaN-RMSD faellt aus der Statistik, Komplex bleibt", o["n_complexes"] == 2)
recs_s = [rec("A", "0", 1.0, 0.9), rec("A", "1", 3.0, None)]
t = top1_at_k(recs_s, 2)
check("Posen ohne Score werden GEZAEHLT", t["n_missing_score"] == 1)

print("\n=== 6. Perfektes und invertiertes Ranking ===")
rng = np.random.default_rng(3)
base = [(f"C{c}", i, float(rng.gamma(2.0, 1.5))) for c in range(50) for i in range(10)]
perfect = [rec(c, str(i), r, score=-r) for c, i, r in base]      # Score = -RMSD
inverted = [rec(c, str(i), r, score=+r) for c, i, r in base]
tp = top1_at_k(perfect, 10)
ti = top1_at_k(inverted, 10)
op = oracle_at_k(perfect, 10)
check("perfektes Ranking erreicht Oracle@K",
      abs(tp["success_2A"] - op["success_2A"]) < 1e-9,
      f"Top1 {100*tp['success_2A']:.1f} % == Oracle {100*op['success_2A']:.1f} %")
check("perfektes Ranking hat Regret 0", abs(tp["mean_regret"]) < 1e-9)
check("invertiertes Ranking hat positiven Regret", ti["mean_regret"] > 0,
      f"{ti['mean_regret']:.2f} A")
qp, qi = ranking_quality(perfect), ranking_quality(inverted)
check("Spearman ~ +1 bei perfektem Ranking", qp["spearman_median"] > 0.99,
      f"{qp['spearman_median']:.3f}")
check("Spearman ~ -1 bei invertiertem", qi["spearman_median"] < -0.99,
      f"{qi['spearman_median']:.3f}")
check("Recall der besten Pose = 100 % bei perfektem Ranking",
      abs(qp["recall_best_in_top1"] - 1.0) < 1e-9)

print("\n=== 7. Top-1 liegt immer zwischen Oracle und Zufall ===")
rng = np.random.default_rng(4)
noisy = [rec(c, str(i), r, score=-r + float(rng.normal(0, 1.0))) for c, i, r in base]
o10 = oracle_at_k(noisy, 10, seed=5)
t10 = top1_at_k(noisy, 10, seed=5)
o1 = oracle_at_k(noisy, 1, seed=5)
check("Oracle@10 >= Top1@10 >= Einzelsample",
      o10["success_2A"] >= t10["success_2A"] >= o1["success_2A"] - 0.02,
      f"{100*o10['success_2A']:.1f} >= {100*t10['success_2A']:.1f} >= {100*o1['success_2A']:.1f}")

print("\n=== 8. Kalibrierung ===")
rng = np.random.default_rng(6)
probs = rng.uniform(0, 1, 4000)
labels = rng.uniform(0, 1, 4000) < probs          # perfekt kalibriert
cal = [rec(f"C{i//10}", str(i % 10), 1.0 if labels[i] else 5.0, score=float(probs[i]))
       for i in range(4000)]
c = calibration(cal)
check("AUROC deutlich ueber 0.5", c["auroc"] > 0.7, f"{c['auroc']:.3f}")
check("ECE klein bei perfekter Kalibrierung", c["ece"] < 0.05, f"{c['ece']:.3f}")
check("Brier plausibel", 0.1 < c["brier"] < 0.25, f"{c['brier']:.3f}")

print("\n=== 9. Gesamtlauf ohne Score ===")
res = evaluate_ranking([rec(c, str(i), r) for c, i, r in base])
check("laeuft ohne Score durch", res["has_score"] is False)
check("Oracle fuer alle K < max vorhanden", set(res["oracle"]) == {1, 5, 10})
check("kein Top-1 ohne Score", res["top1"] == {})

print("\n" + "=" * 70)
if FAILS:
    print(f"FEHLGESCHLAGEN: {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("ALLE RANKING-TESTS BESTANDEN")
