"""Read the raw outputs of both completion tests and print the verdict.

Applies the decision logic agreed in advance, so the classification is not
chosen after seeing the numbers.

Usage: python summarize.py <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Thresholds fixed BEFORE the run.
EQ_TOL = 1e-3          # relative equivariance error
NEG_ALARM = 0.55       # fraction of negative rotation cosines that means "systematically negative"
WEAK = 0.15            # |cos| below this counts as "not learned"
STRONG = 0.35          # above this counts as "clearly learned"


def main(out_dir: Path) -> None:
    lines = ["=" * 100, "SIGMAFLOW — ABSCHLUSSDIAGNOSTIK DER MINIMAL-KONVERSION", "=" * 100, ""]

    # ---------- TEST 1 ----------
    raw = out_dir / "cosine_by_t_raw.csv"
    outcome = "UNBEKANNT"
    if not raw.exists():
        lines.append("TEST 1: RAW-DATEI FEHLT — Test nicht gelaufen.")
    else:
        df = pd.read_csv(raw)
        df["bin"] = np.clip((df["t"] * 10).astype(int), 0, 9)
        lines += ["TEST 1 — KOSINUS UEBER t", "-" * 100,
                  f"{'t-bin':<12}{'n':>7}{'trans mean':>12}{'trans med':>11}"
                  f"{'rot mean':>11}{'rot med':>10}{'rot neg %':>11}{'rot ratio':>11}"]
        per_bin = []
        for b in range(10):
            d = df[df["bin"] == b]
            if not len(d):
                continue
            per_bin.append((b, d.cos_rot.mean()))
            lines.append(
                f"{f'[{b/10:.1f},{(b+1)/10:.1f})':<12}{len(d):>7}"
                f"{d.cos_trans.mean():>12.3f}{d.cos_trans.median():>11.3f}"
                f"{d.cos_rot.mean():>11.3f}{d.cos_rot.median():>10.3f}"
                f"{100*(d.cos_rot < 0).mean():>10.1f}%{d.ratio_rot.median():>11.3f}")

        ct, cr = df.cos_trans.mean(), df.cos_rot.mean()
        neg_r = (df.cos_rot < 0).mean()
        lo = df[df.t < 0.35].cos_rot.mean()
        hi = df[df.t > 0.65].cos_rot.mean()
        lines += ["", f"GESAMT  cos_trans={ct:.3f}   cos_rot={cr:.3f}   "
                      f"Anteil cos_rot<0 = {100*neg_r:.1f}%",
                  f"        cos_rot bei kleinem t (<0.35) = {lo:.3f}   "
                  f"bei grossem t (>0.65) = {hi:.3f}   Anstieg = {hi-lo:+.3f}"]

        if neg_r > NEG_ALARM or cr < -WEAK:
            outcome = "C"
            lines.append("\n  -> OUTCOME C: SYSTEMATISCH NEGATIV. Versteckter Frame-/Vorzeichenfehler.")
            lines.append("     STOP. Nicht als vollstaendig einstufen.")
        elif ct > STRONG and lo < WEAK and (hi - lo) > 0.15:
            outcome = "A"
            lines.append("\n  -> OUTCOME A: FRUEHZEITIGE ROTATIONSBLINDHEIT.")
            lines.append("     Rotation bei kleinem t nicht aufgeloest, bei grossem t schon.")
            lines.append("     Naechster Hebel: Zeitverteilung in _sample_time, nicht die Architektur.")
        elif WEAK <= cr <= STRONG:
            outcome = "B"
            lines.append("\n  -> OUTCOME B: ROTATION DURCHGEHEND SCHWACH, ABER POSITIV.")
            lines.append("     Die Repraesentation lernt Rotation nur schwach.")
            lines.append("     Rechtfertigt einen dedizierten Rotationskopf — aber erst nach Test 2.")
        elif cr > STRONG:
            outcome = "D"
            lines.append("\n  -> OUTCOME D: ROTATIONSVORHERSAGE IST GUT.")
            lines.append("     Der Fehler entsteht DOWNSTREAM: Konversion, Indizierung,")
            lines.append("     Batching, Integration, Posenaufbau oder Auswertung.")
        else:
            outcome = "B?"
            lines.append("\n  -> Zwischenfall: cos_rot nahe 0, aber nicht negativ. Wie B behandeln.")

    # ---------- TEST 2 ----------
    eqf = out_dir / "equivariance_results.json"
    eq_ok = None
    lines += ["", "=" * 100, "TEST 2 — GLOBALE SE(3)-AEQUIVARIANZ", "-" * 100]
    if not eqf.exists():
        lines.append("ERGEBNISDATEI FEHLT — Test nicht gelaufen.")
    else:
        eq = json.loads(eqf.read_text())
        res = eq["results"]
        lines.append(f"{'Variante':<26}{'trans rel med':>15}{'rot rel med':>14}   Urteil")
        for k, v in res.items():
            t_med = v["trans_rel_err"]["median"]
            r_med = v["rot_rel_err"]["median"]
            passed = (t_med < EQ_TOL) and (r_med < EQ_TOL)
            expect = (k == "MAIN")
            flag = "" if passed == expect else "   <-- UNERWARTET"
            lines.append(f"{k:<26}{t_med:>15.2e}{r_med:>14.2e}   "
                         f"{'BESTANDEN' if passed else 'FEHLER'}{flag}")
        main_ok = (res["MAIN"]["trans_rel_err"]["median"] < EQ_TOL and
                   res["MAIN"]["rot_rel_err"]["median"] < EQ_TOL)
        ctrl_fail = all(not (res[k]["trans_rel_err"]["median"] < EQ_TOL and
                             res[k]["rot_rel_err"]["median"] < EQ_TOL)
                        for k in res if k != "MAIN")
        eq_ok = main_ok and ctrl_fail
        lines.append("")
        if not ctrl_fail:
            lines.append("  WARNUNG: mindestens eine Kontrolle hat BESTANDEN. Der Test ist blind;")
            lines.append("  ein bestandenes MAIN-Ergebnis ist dann nicht aussagekraeftig.")
        lines.append(f"  Aequivarianz insgesamt: {'BESTANDEN' if eq_ok else 'NICHT BESTANDEN'}")

    # ---------- VERDICT ----------
    lines += ["", "=" * 100, "GESAMTURTEIL", "=" * 100]
    if eq_ok and outcome in ("A", "B", "B?", "D"):
        lines += ["MINIMAL SIGMAFLOW CONVERSION COMPLETE",
                  "",
                  "  Aequivarianz besteht, die Kontrollen schlagen fehl (Test ist scharf),",
                  "  und der Kosinus ist nicht systematisch negativ. Zusammen mit den bereits",
                  "  bestandenen Oracle-, Vorzeichen- und Rundreisetests heisst das:",
                  "  die Konversion ist implementierungsseitig korrekt. Verbleibende",
                  "  Leistungsluecken sind Modellierungs-/Trainingsfragen, keine Defekte."]
    elif outcome == "C":
        lines += ["MINIMAL SIGMAFLOW CONVERSION NOT YET COMPLETE",
                  "  Grund: systematisch negativer Rotationskosinus (Outcome C)."]
    elif eq_ok is False:
        lines += ["MINIMAL SIGMAFLOW CONVERSION NOT YET COMPLETE",
                  "  Grund: Aequivarianztest nicht bestanden."]
    else:
        lines += ["UNENTSCHIEDEN — mindestens ein Test fehlt. Beide Tests wiederholen."]

    txt = "\n".join(lines)
    print(txt)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
