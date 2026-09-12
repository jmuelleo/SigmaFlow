"""Tests der paper-konformen SigmaDock-Heuristik.

Die Heuristik ist die BASELINE, gegen die eine gelernte Confidence antreten
muss. Wenn sie falsch reproduziert ist, ist der ganze Vergleich wertlos --
und zwar in beide Richtungen: eine zu schwache Baseline laesst die gelernte
Variante gut aussehen, eine zu starke laesst sie unnoetig scheitern.

    python SigmaFlow_Evaluation/tests/test_heuristic_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from SigmaFlow_Evaluation.ranking.heuristic_score import (  # noqa: E402
    BETA,
    PB_CHECKS,
    mixed_score,
    pb_fraction_from_row,
    score_table,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:  # noqa: C901
    print("\n1. Die Formel s = -b * p^4")
    check("beta ist 4 wie im Paper", BETA == 4.0, str(BETA))
    check("fuenf PB-Checks wie in Appendix F.2", len(PB_CHECKS) == 5,
          ", ".join(PB_CHECKS))

    # Handgerechnetes Beispiel: b = -8.0, p = 0.5 -> s = 8.0 * 0.0625 = 0.5
    s = mixed_score(-8.0, 0.5)
    check("Beispiel -8.0 und p=0.5 ergibt 0.5", abs(s - 0.5) < 1e-12, f"{s}")
    # p = 1 -> s = -b, also die negierte Energie
    check("p = 1 gibt die negierte Energie", abs(mixed_score(-9.3, 1.0) - 9.3) < 1e-12)
    # p = 0 -> s = 0, unabhaengig von der Energie. Eine chemisch unmoegliche
    # Pose wird also auf denselben Wert gedrueckt, egal wie gut sie bindet.
    check("p = 0 loescht die Energie vollstaendig",
          mixed_score(-99.0, 0.0) == 0.0)

    print("\n2. Richtung: besser bindend UND plausibler gewinnt")
    check("bei gleichem p gewinnt die niedrigere Energie",
          mixed_score(-9.0, 0.8) > mixed_score(-6.0, 0.8))
    check("bei gleicher Energie gewinnt das hoehere p",
          mixed_score(-8.0, 0.9) > mixed_score(-8.0, 0.6))
    # Der Exponent 4 ist scharf: p=0.8 gegen p=1.0 kostet Faktor 0.41.
    ratio = mixed_score(-8.0, 0.8) / mixed_score(-8.0, 1.0)
    check("beta=4 bestraft p=0.8 um Faktor ~0.41", abs(ratio - 0.8 ** 4) < 1e-12,
          f"{ratio:.4f}")
    # Gegenprobe zur Scharfe: mit beta=1 waere die Strafe viel milder.
    ratio1 = mixed_score(-8.0, 0.8, beta=1.0) / mixed_score(-8.0, 1.0, beta=1.0)
    check("mit beta=1 waere die Strafe deutlich milder (Test kann scheitern)",
          ratio1 > ratio + 0.3, f"beta=1: {ratio1:.2f} gegen beta=4: {ratio:.2f}")

    print("\n3. Ungueltige Eingaben werden nicht stillschweigend geklemmt")
    for bad_p in (-0.1, 1.2):
        try:
            mixed_score(-8.0, bad_p)
            check(f"p={bad_p} wird abgelehnt", False)
        except ValueError:
            check(f"p={bad_p} wird abgelehnt", True)
    try:
        mixed_score(float("nan"), 0.5)
        check("nicht-endliche Energie wird abgelehnt", False)
    except ValueError:
        check("nicht-endliche Energie wird abgelehnt", True)

    print("\n4. PB-Anteil aus einer Zeile")
    full = {c: "True" for c in PB_CHECKS}
    p, miss = pb_fraction_from_row(full)
    check("alle Checks bestanden -> p = 1", p == 1.0 and not miss)
    half = {c: ("True" if i < 3 else "False") for i, c in enumerate(PB_CHECKS)}
    p, _ = pb_fraction_from_row(half)
    check("3 von 5 bestanden -> p = 0.6", abs(p - 0.6) < 1e-12, f"{p}")
    check("Zahlen und Wahrheitswerte werden beide verstanden",
          abs(pb_fraction_from_row({c: "1" for c in PB_CHECKS})[0] - 1.0) < 1e-12)

    # DER entscheidende Fall: ein FEHLENDER Check darf nicht als bestanden
    # zaehlen. Sonst bekaeme eine Datei, in der ein Check nicht berechnet wurde,
    # einen besseren Score als eine, in der er berechnet und nicht bestanden
    # wurde -- der Ranker wuerde fehlende Information belohnen.
    partial = {c: "False" for c in PB_CHECKS[:4]}      # fuenfter fehlt ganz
    p_partial, miss = pb_fraction_from_row(partial)
    p_falsefive = pb_fraction_from_row({c: "False" for c in PB_CHECKS})[0]
    check("fehlender Check wird gemeldet, nicht als bestanden gewertet",
          p_partial == 0.0 and PB_CHECKS[4] in miss,
          f"p={p_partial}, fehlend={miss}")
    check("fehlender Check macht den Score nicht besser",
          p_partial <= p_falsefive + 1e-12)

    print("\n5. Tabelle end-to-end")
    rows = [
        {"complex_id": "AAAA", "pose_id": "seed0", "vinardo": "-9.0",
         **{c: "True" for c in PB_CHECKS}},
        {"complex_id": "AAAA", "pose_id": "seed1", "vinardo": "-9.5",
         **{c: "False" for c in PB_CHECKS}},
        {"complex_id": "BBBB", "pose_id": "seed0", "vinardo": "",
         **{c: "True" for c in PB_CHECKS}},        # gnina fehlt
    ]
    scores, diag = score_table(rows)
    check("zwei Posen bewertet, eine ohne Energie uebersprungen",
          diag["n_scored"] == 2 and diag["n_no_energy"] == 1, str(diag))
    check("die chemisch plausible Pose gewinnt trotz schlechterer Energie",
          scores["AAAA"]["seed0"] > scores["AAAA"]["seed1"],
          f"{scores['AAAA']['seed0']:.2f} > {scores['AAAA']['seed1']:.2f}")
    check("Abdeckung wird beziffert", diag["coverage"] == round(2 / 3, 4),
          str(diag["coverage"]))
    check("Format passt zu evaluate_run --scores",
          isinstance(scores, dict) and isinstance(scores["AAAA"], dict))

    print("\n6. Anschluss an den Ranking-Evaluator")
    from SigmaFlow_Evaluation.metrics.ranking import PoseRecord, evaluate_ranking
    from dataclasses import replace
    recs = [PoseRecord("AAAA", "seed0", 1.0, None, "m"),
            PoseRecord("AAAA", "seed1", 5.0, None, "m"),
            PoseRecord("BBBB", "seed0", 0.5, None, "m"),
            PoseRecord("BBBB", "seed1", 4.0, None, "m")]
    sc = {"AAAA": {"seed0": 10.0, "seed1": 1.0}, "BBBB": {"seed0": 10.0, "seed1": 1.0}}
    scored = [replace(r, score=sc[r.complex_id][r.pose_id]) for r in recs]
    res = evaluate_ranking(scored, k_values=(1, 2))
    check("perfekt korrelierter Score erreicht das Oracle",
          abs(res["top1"][2]["success_2A"] - res["oracle"][2]["success_2A"]) < 1e-9,
          f"top1={res['top1'][2]['success_2A']:.2f} oracle={res['oracle'][2]['success_2A']:.2f}")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


def test_heuristic_score() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
