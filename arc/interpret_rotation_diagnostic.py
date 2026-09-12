"""Liest `cosine_by_t_raw.csv` und stellt die Weichenfrage automatisch.

WOZU
Die Rotationsdiagnose entscheidet, welcher Forschungsast als Naechstes sinnvoll
ist (SIGMAFLOW_RESEARCH_ROADMAP.md, G8). Diese Entscheidung soll nicht davon
abhaengen, wer die CSV gerade anschaut und worauf er zuerst blickt.

DIE BEIDEN KONKURRIERENDEN ERKLAERUNGEN

  H-SHRINK   Bei kleinem t ist R_1 aus dem Zustand nicht bestimmbar. Der
             L2-optimale Ausgang ist der bedingte Mittelwert, und Mittelung von
             Rotationsvektoren schrumpft gegen null. Ein geschrumpftes Feld
             erzeugt beim Integrieren kaum Drehung.
             Signatur:  ||v_R|| << ||u_R||,  Kosinus aber > 0.
             Konsequenz: konditionierte Quelle (A1) oder Endpunkt-
             Parametrisierung (G2) sind die richtigen Hebel.

  H-NOISE    Die Magnitude stimmt ungefaehr, die Richtung ist Rauschen. Das
             passt zur Architektur: das Drehmoment ist das inertianormierte
             erste Moment eines Kraftfeldes und verstaerkt Rauschen auf realen
             Fragmentgeometrien 8.8x staerker als die Translation.
             Signatur:  ||v_R|| ~ ||u_R||,  Kosinus ~ 0.
             Konsequenz: direkter aequivarianter Rotationskopf (G1).

Die Schwellen unten sind DIAGNOSTISCHE HEURISTIKEN, keine wissenschaftlich
begruendeten Grenzwerte. Sie stehen hier explizit, damit man sie diskutieren
kann, statt sie im Kopf zu haben. Die Rohzahlen werden immer mit ausgegeben.

    python arc/interpret_rotation_diagnostic.py --csv .../cosine_by_t_raw.csv
    python arc/interpret_rotation_diagnostic.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --- diagnostische Heuristiken, bewusst konservativ ----------------------
RATIO_SHRINK = 0.5      # ||v||/||u|| darunter gilt als deutlich geschrumpft
RATIO_OK_LO = 0.7       # dazwischen: Magnitude im Wesentlichen richtig
RATIO_OK_HI = 1.5
COS_SIGNAL = 0.15       # darueber traegt die Richtung Signal
COS_NOISE = 0.05        # darunter ist sie praktisch Rauschen
EARLY_T = 0.35          # "frueh" - dort entscheidet sich die Lernbarkeit
MIN_N = 30              # weniger Beobachtungen je Bin -> nicht interpretieren


def ci95(x: np.ndarray) -> float:
    """Halbe Breite des 95-%-Intervalls des Mittelwerts."""
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def bin_summary(df, col_cos: str, col_ratio: str, edges) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["t"] >= lo) & (df["t"] < hi)
        n = int(m.sum())
        if n == 0:
            continue
        c = df.loc[m, col_cos].to_numpy(dtype=float)
        r = df.loc[m, col_ratio].to_numpy(dtype=float)
        c, r = c[np.isfinite(c)], r[np.isfinite(r)]
        out.append({
            "t_lo": lo, "t_hi": hi, "n": n,
            "cos_mean": float(c.mean()) if c.size else float("nan"),
            "cos_ci": ci95(c) if c.size else float("nan"),
            "cos_median": float(np.median(c)) if c.size else float("nan"),
            "ratio_median": float(np.median(r)) if r.size else float("nan"),
            "ratio_mean": float(r.mean()) if r.size else float("nan"),
            "frac_negative": float((c < 0).mean()) if c.size else float("nan"),
        })
    return out


def classify(early_ratio: float, early_cos: float, n_early: int) -> tuple[str, str]:
    """Die Weichenstellung. Gibt (Label, Begruendung) zurueck."""
    if n_early < MIN_N or not np.isfinite(early_ratio) or not np.isfinite(early_cos):
        return "INCONCLUSIVE", f"zu wenige verwertbare Beobachtungen bei t < {EARLY_T} (n={n_early})"

    shrunk = early_ratio < RATIO_SHRINK
    mag_ok = RATIO_OK_LO <= early_ratio <= RATIO_OK_HI
    dir_signal = early_cos > COS_SIGNAL
    dir_noise = early_cos < COS_NOISE

    if shrunk and dir_signal:
        return "SHRINKAGE-LIKE", (
            f"Magnitude stark unterschaetzt (Verhaeltnis {early_ratio:.2f} < {RATIO_SHRINK}), "
            f"Richtung traegt aber Signal (cos {early_cos:.3f} > {COS_SIGNAL})")
    if mag_ok and dir_noise:
        return "DIRECTION-NOISE-LIKE", (
            f"Magnitude im Wesentlichen richtig (Verhaeltnis {early_ratio:.2f}), "
            f"Richtung praktisch Rauschen (cos {early_cos:.3f} < {COS_NOISE})")
    if shrunk and dir_noise:
        return "BOTH", (
            f"Magnitude geschrumpft (Verhaeltnis {early_ratio:.2f}) UND Richtung Rauschen "
            f"(cos {early_cos:.3f}) - der Kanal traegt gar nichts")
    return "INCONCLUSIVE", (
        f"Verhaeltnis {early_ratio:.2f}, cos {early_cos:.3f} passen in kein Muster; "
        "Schwellen und Rohdaten pruefen")


BRANCH = {
    "SHRINKAGE-LIKE": [
        "Das Ziel ist bei kleinem t nicht bestimmbar, das Modell weicht auf den",
        "bedingten Mittelwert aus. Die passenden Hebel sind:",
        "  1. A1-A  heuristische konditionierte Quelle (FM-spezifisch)",
        "  2. G2    Endpunkt-Parametrisierung R_1 statt Vektorfeld",
        "EXP-101 sollte VOR A1-A laufen (Abbruchkriterium).",
    ],
    "DIRECTION-NOISE-LIKE": [
        "Die Magnitude kommt an, die Richtung nicht. Das passt zur Architektur:",
        "Rotation ist das inertianormierte erste Moment eines Kraftfeldes.",
        "  1. G1    direkter aequivarianter Rotationskopf (allgemein, nicht FM)",
        "Die konditionierte Quelle waere hier NICHT der erste Schritt - sie",
        "wuerde einen kaputten Kanal kaschieren statt ihn zu reparieren.",
    ],
    "BOTH": [
        "Der Rotationskanal traegt weder Magnitude noch Richtung.",
        "Zuerst die Repraesentation reparieren (G1), erst danach die Quelle (A1).",
        "Eine Quelle vor der Reparatur waere nicht interpretierbar: jeder Gewinn",
        "waere ihr allein zuzuschreiben.",
    ],
    "INCONCLUSIVE": [
        "Keine klare Signatur. Moegliche Ursachen:",
        "  - zu wenige Komplexe in der Diagnose (--max_complexes erhoehen)",
        "  - Checkpoint zu frueh im Training",
        "  - die Schwellen oben passen nicht zu diesem Modell",
        "Vor einer Astentscheidung die Rohtabelle unten ansehen.",
    ],
}


def report(df, label: str = "") -> str:
    edges = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.01]
    rot = bin_summary(df, "cos_rot", "ratio_rot", edges)
    tra = bin_summary(df, "cos_trans", "ratio_trans", edges)

    L = ["=" * 100, f"ROTATIONSDIAGNOSE  {label}", "=" * 100,
         f"{len(df)} Beobachtungen, {df['t'].nunique()} Zeitpunkte, "
         f"{df['global_complex'].nunique() if 'global_complex' in df else '?'} Komplexe", ""]

    L.append("ROTATION")
    L.append(f"  {'t-Bereich':<14}{'n':>7}{'cos mean':>11}{'+-95%':>9}"
             f"{'cos med':>10}{'|v|/|u| med':>13}{'cos<0':>8}")
    for b in rot:
        L.append(f"  [{b['t_lo']:.2f}, {b['t_hi']:.2f})  {b['n']:>6} "
                 f"{b['cos_mean']:>10.3f}{b['cos_ci']:>9.3f}{b['cos_median']:>10.3f}"
                 f"{b['ratio_median']:>13.3f}{100*b['frac_negative']:>7.0f}%")

    L += ["", "TRANSLATION (Vergleichsmassstab - dieser Kanal funktioniert)"]
    L.append(f"  {'t-Bereich':<14}{'n':>7}{'cos mean':>11}{'+-95%':>9}"
             f"{'cos med':>10}{'|v|/|u| med':>13}{'cos<0':>8}")
    for b in tra:
        L.append(f"  [{b['t_lo']:.2f}, {b['t_hi']:.2f})  {b['n']:>6} "
                 f"{b['cos_mean']:>10.3f}{b['cos_ci']:>9.3f}{b['cos_median']:>10.3f}"
                 f"{b['ratio_median']:>13.3f}{100*b['frac_negative']:>7.0f}%")

    early = df[df["t"] < EARLY_T]
    er = early["ratio_rot"].to_numpy(dtype=float)
    ec = early["cos_rot"].to_numpy(dtype=float)
    er, ec = er[np.isfinite(er)], ec[np.isfinite(ec)]
    e_ratio = float(np.median(er)) if er.size else float("nan")
    e_cos = float(ec.mean()) if ec.size else float("nan")

    et = early["ratio_trans"].to_numpy(dtype=float)
    ect = early["cos_trans"].to_numpy(dtype=float)
    et, ect = et[np.isfinite(et)], ect[np.isfinite(ect)]

    L += ["", "=" * 100, f"FRUEHES REGIME  (t < {EARLY_T})  — hier entscheidet sich die Lernbarkeit",
          "=" * 100]
    L.append(f"  Rotation    : |v|/|u| median {e_ratio:6.3f}   cos mean {e_cos:+.3f} "
             f"+- {ci95(ec) if ec.size else float('nan'):.3f}   n={len(ec)}")
    if et.size:
        L.append(f"  Translation : |v|/|u| median {np.median(et):6.3f}   "
                 f"cos mean {ect.mean():+.3f} +- {ci95(ect):.3f}   n={len(ect)}")

    verdict, why = classify(e_ratio, e_cos, len(ec))
    L += ["", "=" * 100, f"BEFUND: {verdict}", "=" * 100, f"  {why}", ""]
    L += ["  " + ln for ln in BRANCH[verdict]]
    L += ["", "  Die Schwellen sind diagnostische Heuristiken, keine Messgroessen:",
          f"    Schrumpfung wenn |v|/|u| < {RATIO_SHRINK}",
          f"    Magnitude ok  wenn {RATIO_OK_LO} <= |v|/|u| <= {RATIO_OK_HI}",
          f"    Richtung Signal wenn cos > {COS_SIGNAL}, Rauschen wenn cos < {COS_NOISE}"]
    return "\n".join(L)


def _self_test() -> int:
    """Erzeugt synthetische CSVs mit bekannter Signatur und prueft die Einordnung."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 4000
    t = rng.uniform(0, 1, n)
    cases = {
        # ratio, cos, erwartetes Label
        "SHRINKAGE-LIKE": (0.15, 0.45),
        "DIRECTION-NOISE-LIKE": (1.0, 0.01),
        "BOTH": (0.2, 0.0),
    }
    fails = []
    for expect, (ratio, cosv) in cases.items():
        df = pd.DataFrame({
            "t": t, "global_complex": rng.integers(0, 50, n),
            "cos_rot": rng.normal(cosv, 0.25, n),
            "ratio_rot": np.abs(rng.normal(ratio, ratio * 0.2 + 0.02, n)),
            "cos_trans": rng.normal(0.6, 0.2, n),
            "ratio_trans": np.abs(rng.normal(0.9, 0.1, n)),
        })
        early = df[df["t"] < EARLY_T]
        got, _ = classify(float(np.median(early["ratio_rot"])),
                          float(early["cos_rot"].mean()), len(early))
        ok = got == expect
        print(f"  [{'ok ' if ok else 'FAIL'}] {expect:<22} -> {got}")
        if not ok:
            fails.append(expect)
    # Zu wenig Daten muss INCONCLUSIVE geben
    got, _ = classify(0.2, 0.4, 5)
    ok = got == "INCONCLUSIVE"
    print(f"  [{'ok ' if ok else 'FAIL'}] {'zu wenige Daten':<22} -> {got}")
    if not ok:
        fails.append("wenig-daten")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="cosine_by_t_raw.csv")
    ap.add_argument("--label", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        print("Selbsttest der Einordnungslogik:")
        return _self_test()

    if not args.csv:
        ap.error("--csv oder --self-test angeben")
    p = Path(args.csv)
    if not p.exists():
        print(f"Datei nicht gefunden: {p}")
        return 1
    import pandas as pd
    print(report(pd.read_csv(p), args.label or p.parent.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
